"""Unit tests for queue worker loop helper functions."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from acestep.api.jobs.worker_loops import (
    process_queue_item,
    run_job_store_cleanup_loop,
)


class _FakeJobQueue:
    """Track task completion calls made by worker helper."""

    def __init__(self) -> None:
        """Initialize task_done call counter."""

        self.task_done_calls = 0

    def task_done(self) -> None:
        """Record one task completion."""

        self.task_done_calls += 1


class _FakeStore:
    """In-memory store stub for worker helper tests."""

    def __init__(self, record=None) -> None:
        """Initialize store with optional record."""

        self.record = record
        self.mark_failed_calls = []
        self.cleanup_returns = [0]
        self.cleanup_call_count = 0
        self.stats = {"total": 0}

    def get(self, _job_id: str):
        """Return configured record."""

        return self.record

    def mark_failed(self, job_id: str, error: str) -> None:
        """Record mark_failed invocations."""

        self.mark_failed_calls.append((job_id, error))
        if self.record is not None:
            self.record.status = "failed"
            self.record.error = error

    def cleanup_old_jobs(self) -> int:
        """Return deterministic cleanup counts for each invocation."""

        index = min(self.cleanup_call_count, len(self.cleanup_returns) - 1)
        self.cleanup_call_count += 1
        result = self.cleanup_returns[index]
        if isinstance(result, Exception):
            raise result
        return int(result)

    def get_stats(self):
        """Return deterministic stats payload."""

        return self.stats


class _DoneEvent:
    """Simple event stub tracking whether set() was called."""

    def __init__(self) -> None:
        """Initialize unset event state."""

        self.was_set = False

    def set(self) -> None:
        """Mark event as set."""

        self.was_set = True


class WorkerLoopTests(unittest.TestCase):
    """Behavior tests for extracted queue worker helper logic."""

    def test_process_queue_item_success_notifies_result_and_done(self):
        """Success path should emit result+done and cleanup queue bookkeeping."""

        async def _run() -> None:
            progress_queue = asyncio.Queue()
            done_event = _DoneEvent()
            record = SimpleNamespace(
                status="running",
                result=None,
                error=None,
                progress_queue=progress_queue,
                done_event=done_event,
            )
            store = _FakeStore(record=record)
            app_state = SimpleNamespace(
                pending_lock=asyncio.Lock(),
                pending_ids=["job-1"],
                job_queue=_FakeJobQueue(),
            )
            cleanup_calls = []

            async def _run_one_job(job_id: str, _req) -> None:
                record.status = "succeeded"
                record.result = {"job_id": job_id, "ok": True}

            async def _cleanup_job_temp_files(job_id: str) -> None:
                cleanup_calls.append(job_id)

            await process_queue_item(
                job_id="job-1",
                req=SimpleNamespace(),
                app_state=app_state,
                store=store,
                run_one_job=_run_one_job,
                cleanup_job_temp_files=_cleanup_job_temp_files,
            )

            msg1 = await progress_queue.get()
            msg2 = await progress_queue.get()
            self.assertEqual({"type": "result", "result": {"job_id": "job-1", "ok": True}}, msg1)
            self.assertEqual({"type": "done"}, msg2)
            self.assertTrue(done_event.was_set)
            self.assertEqual([], app_state.pending_ids)
            self.assertEqual(["job-1"], cleanup_calls)
            self.assertEqual(1, app_state.job_queue.task_done_calls)

        asyncio.run(_run())

    def test_process_queue_item_failure_marks_failed_and_notifies_error(self):
        """Failure path should mark failed, emit error+done, and perform cleanup."""

        async def _run() -> None:
            progress_queue = asyncio.Queue()
            done_event = _DoneEvent()
            record = SimpleNamespace(
                status="running",
                result=None,
                error=None,
                progress_queue=progress_queue,
                done_event=done_event,
            )
            store = _FakeStore(record=record)
            app_state = SimpleNamespace(
                pending_lock=asyncio.Lock(),
                pending_ids=["job-2"],
                job_queue=_FakeJobQueue(),
            )
            cleanup_calls = []

            async def _run_one_job(_job_id: str, _req) -> None:
                raise RuntimeError("boom")

            async def _cleanup_job_temp_files(job_id: str) -> None:
                cleanup_calls.append(job_id)

            await process_queue_item(
                job_id="job-2",
                req=SimpleNamespace(),
                app_state=app_state,
                store=store,
                run_one_job=_run_one_job,
                cleanup_job_temp_files=_cleanup_job_temp_files,
            )

            msg1 = await progress_queue.get()
            msg2 = await progress_queue.get()
            self.assertEqual({"type": "error", "content": "boom"}, msg1)
            self.assertEqual({"type": "done"}, msg2)
            self.assertTrue(done_event.was_set)
            self.assertEqual([("job-2", "boom")], store.mark_failed_calls)
            self.assertEqual([], app_state.pending_ids)
            self.assertEqual(["job-2"], cleanup_calls)
            self.assertEqual(1, app_state.job_queue.task_done_calls)

        asyncio.run(_run())

    def test_run_job_store_cleanup_loop_logs_cleanup_and_stops_on_cancel(self):
        """Cleanup loop should log cleanup stats and exit on cancellation."""

        async def _run() -> None:
            store = _FakeStore()
            store.cleanup_returns = [2]
            store.stats = {"total": 3, "succeeded": 2, "failed": 1}
            logs = []
            calls = {"count": 0}

            async def _sleep_fn(_seconds: float) -> None:
                calls["count"] += 1
                if calls["count"] == 1:
                    return
                raise asyncio.CancelledError

            await run_job_store_cleanup_loop(
                store=store,
                cleanup_interval_seconds=1,
                sleep_fn=_sleep_fn,
                log_fn=logs.append,
            )

            self.assertEqual(1, store.cleanup_call_count)
            self.assertEqual(1, len(logs))
            self.assertIn("Cleaned up 2 old jobs", logs[0])

        asyncio.run(_run())

    def test_run_job_store_cleanup_loop_logs_errors_and_continues(self):
        """Cleanup loop should log non-cancel exceptions and continue until cancelled."""

        async def _run() -> None:
            store = _FakeStore()
            store.cleanup_returns = [RuntimeError("cleanup-failed"), 0]
            logs = []
            calls = {"count": 0}

            async def _sleep_fn(_seconds: float) -> None:
                calls["count"] += 1
                if calls["count"] <= 2:
                    return
                raise asyncio.CancelledError

            await run_job_store_cleanup_loop(
                store=store,
                cleanup_interval_seconds=1,
                sleep_fn=_sleep_fn,
                log_fn=logs.append,
            )

            self.assertGreaterEqual(store.cleanup_call_count, 2)
            self.assertTrue(any("Job cleanup error: cleanup-failed" in line for line in logs))

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
