from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Lock management for RadixCache snapshot operations.
"""

import threading
from typing import Optional


class SnapshotInProgressError(Exception):
    """Raised when attempting a snapshot operation while another is in progress."""

    def __init__(self, message: Optional[str] = None):
        super().__init__(
            message
            or "Cannot start snapshot operation: another operation is in progress"
        )


class SnapshotLock:
    """Thread-safe lock for snapshot operations.

    This lock ensures that only one snapshot operation (create/restore) can run at a time.
    It uses a reentrant lock internally to allow the same thread to re-acquire the lock
    if needed.

    Usage:
        lock = SnapshotLock()
        with lock:
            # perform snapshot operation
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot_in_progress = False
        self._owner: Optional[int] = None

    def __enter__(self) -> SnapshotLock:
        """Acquire the lock.

        Raises:
            SnapshotInProgressError: If another snapshot operation is in progress
        """
        self._lock.acquire()
        try:
            current_thread_id = threading.get_ident()
            if self._snapshot_in_progress and self._owner != current_thread_id:
                raise SnapshotInProgressError()
            self._snapshot_in_progress = True
            self._owner = current_thread_id
            return self
        except:
            self._lock.release()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the lock."""
        try:
            if threading.get_ident() == self._owner:
                self._snapshot_in_progress = False
                self._owner = None
        finally:
            self._lock.release()

    @property
    def in_progress(self) -> bool:
        """Whether a snapshot operation is currently in progress."""
        with self._lock:
            return self._snapshot_in_progress

    @property
    def owner(self) -> Optional[int]:
        """Thread ID of the current lock owner, or None if not locked."""
        with self._lock:
            return self._owner
