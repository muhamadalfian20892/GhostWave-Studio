"""
Batch Audio Sanitization Queue and Folder Watcher Automator for GhostWave Studio.

Provides queued batch processing of multiple audio files and automated background
folder monitoring for studio ingestion workflows.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set

from ghostwave.audio_processor import (
    AudioProcessor,
    AudioProcessResult,
    AudioSanitizeOptions,
    sanitize_audio,
    slice_audio_for_suno
)


@dataclass
class BatchTask:
    """Represents a single audio file processing job in the batch queue."""
    input_path: str
    output_path: str
    id: str = ""
    status: str = "queued"          # "queued", "processing", "completed", "failed", "cancelled"
    progress: int = 0
    message: str = "In queue"
    result: Optional[AudioProcessResult] = None
    error_message: Optional[str] = None
    slice_chunks: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = os.path.basename(self.input_path)

    @property
    def success(self) -> bool:
        return self.status == "completed"


class BatchProcessor:
    """
    Manages and executes an ordered queue of audio sanitization tasks.
    Supports cooperative cancellation and per-task status dispatching.
    """

    def __init__(self, processor: Optional[AudioProcessor] = None):
        self.processor = processor or AudioProcessor()
        self.queue: list[BatchTask] = []
        self._cancel_event = threading.Event()
        self._is_running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def add_file(self, input_path: str, output_path: Optional[str] = None) -> BatchTask:
        """Appends a new file to the processing queue."""
        with self._lock:
            stem = Path(input_path).stem
            ext = Path(output_path).suffix if output_path else ".wav"
            out = output_path or os.path.join(os.path.dirname(input_path), f"{stem}_ghostwave_ready{ext}")
            task = BatchTask(
                id=f"task_{len(self.queue) + 1}_{int(time.time())}",
                input_path=os.path.abspath(input_path),
                output_path=os.path.abspath(out)
            )
            self.queue.append(task)
            return task

    def clear(self) -> None:
        """Clears all tasks from the queue."""
        with self._lock:
            if not self._is_running:
                self.queue.clear()

    def cancel(self) -> None:
        """Signals active batch processing to cancel cooperatively."""
        self._cancel_event.set()

    def process_queue(
        self,
        tasks: list[BatchTask],
        options: AudioSanitizeOptions,
        progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
        auto_slice: bool = False,
        slice_duration_sec: float = 22.0
    ) -> list[BatchTask]:
        """Convenience method to populate queue and run batch synchronously."""
        with self._lock:
            self.queue = list(tasks)

        def on_task(t: BatchTask):
            if progress_callback:
                try:
                    idx = self.queue.index(t)
                except ValueError:
                    idx = 0
                progress_callback(idx, len(self.queue), t.progress, t.message)

        return self.run_batch(
            options=options,
            auto_slice=auto_slice,
            slice_duration=slice_duration_sec,
            task_callback=on_task
        )

    def run_batch(
        self,
        options: AudioSanitizeOptions,
        auto_slice: bool = False,
        slice_duration: float = 22.0,
        task_callback: Optional[Callable[[BatchTask], None]] = None,
        overall_progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> list[BatchTask]:
        """
        Executes all queued tasks sequentially.
        """
        self._cancel_event.clear()
        self._is_running = True
        total = len(self.queue)

        try:
            for idx, task in enumerate(self.queue, start=1):
                if self._cancel_event.is_set():
                    task.status = "cancelled"
                    task.message = "Cancelled by user"
                    if task_callback:
                        task_callback(task)
                    continue

                task.status = "processing"
                task.message = "Processing audio..."
                task.progress = 5
                if task_callback:
                    task_callback(task)

                if overall_progress_callback:
                    pct = int(((idx - 1) / max(1, total)) * 100)
                    overall_progress_callback(pct, f"Processing file {idx}/{total}: {os.path.basename(task.input_path)}")

                def _progress_cb(p: int, msg: str):
                    task.progress = p
                    task.message = msg
                    if task_callback:
                        task_callback(task)

                try:
                    res = sanitize_audio(
                        input_path=task.input_path,
                        output_path=task.output_path,
                        options=options,
                        progress_callback=_progress_cb,
                        cancel_event=self._cancel_event
                    )
                    task.result = res
                    if res.success:
                        task.status = "completed"
                        task.progress = 100
                        task.message = f"Completed ({res.final_duration:.1f}s)"

                        if auto_slice:
                            slice_dir = os.path.join(os.path.dirname(task.output_path), f"{Path(task.output_path).stem}_chunks")
                            chunks = slice_audio_for_suno(task.output_path, slice_dir, chunk_duration_sec=slice_duration)
                            task.slice_chunks = chunks
                            task.message += f" | Sliced into {len(chunks)} chunks"
                    else:
                        task.status = "failed"
                        task.error_message = res.error_message or "Unknown error"
                        task.message = f"Failed: {task.error_message}"

                except Exception as ex:
                    task.status = "failed"
                    task.error_message = str(ex)
                    task.message = f"Error: {str(ex)}"

                if task_callback:
                    task_callback(task)

            if overall_progress_callback:
                overall_progress_callback(100, f"Batch processing completed ({total} files).")

        finally:
            self._is_running = False

        return self.queue


class FolderWatcher:
    """
    Monitors a folder on disk and automatically sanitizes newly added audio files.
    """

    SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        options: Optional[AudioSanitizeOptions] = None,
        auto_slice: bool = True,
        slice_duration: float = 22.0,
        poll_interval: float = 3.0,
        on_processed: Optional[Callable[[str, str], None]] = None,
        audio_processor: Optional[AudioProcessor] = None,
        poll_interval_sec: Optional[float] = None
    ):
        # Support legacy positional callers passing AudioProcessor as the 3rd arg
        if isinstance(options, AudioProcessor):
            audio_processor = options
            options = None
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.options = options or AudioSanitizeOptions()
        self.auto_slice = auto_slice
        self.slice_duration = slice_duration
        self.poll_interval = poll_interval_sec if poll_interval_sec is not None else poll_interval
        self.on_processed = on_processed
        self.audio_processor = audio_processor

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed_files: Set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Starts background folder monitoring."""
        if self.is_running:
            return False
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Mark existing files as processed so we only handle newly added ones
        try:
            for f in os.listdir(self.input_dir):
                if Path(f).suffix.lower() in self.SUPPORTED_EXTS:
                    self._processed_files.add(os.path.join(self.input_dir, f))
        except Exception:
            pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GhostWaveFolderWatcher")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stops background folder monitoring."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if os.path.exists(self.input_dir):
                    for fname in os.listdir(self.input_dir):
                        if self._stop_event.is_set():
                            break
                        ext = Path(fname).suffix.lower()
                        if ext not in self.SUPPORTED_EXTS:
                            continue
                        fpath = os.path.join(self.input_dir, fname)
                        if fpath in self._processed_files:
                            continue

                        # Check if file write is complete (size stable)
                        try:
                            s1 = os.path.getsize(fpath)
                            time.sleep(0.5)
                            s2 = os.path.getsize(fpath)
                            if s1 != s2 or s1 == 0:
                                continue
                        except Exception:
                            continue

                        self._processed_files.add(fpath)
                        stem = Path(fname).stem
                        bitrate = getattr(self.options, "output_bitrate", "wav")
                        out_ext = ".mp3" if bitrate in ("320k", "192k") else ".wav"
                        out_file = os.path.join(self.output_dir, f"{stem}_ghostwave_ready{out_ext}")

                        res = sanitize_audio(
                            input_path=fpath,
                            output_path=out_file,
                            options=self.options
                        )

                        if res.success and self.auto_slice:
                            slice_dir = os.path.join(self.output_dir, f"{stem}_chunks")
                            slice_audio_for_suno(out_file, slice_dir, chunk_duration_sec=self.slice_duration)

                        if self.on_processed:
                            self.on_processed(fpath, out_file)

            except Exception:
                pass

            self._stop_event.wait(self.poll_interval)
