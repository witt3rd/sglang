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
Mixin class for adding snapshot capabilities to RadixCache.
"""

from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from .deserializer import RadixTreeDeserializer
from .lock import SnapshotLock
from .serializer import RadixTreeSerializer
from .snapshot_types import SnapshotMetadata


@runtime_checkable
class SnapshotableCache(Protocol):
    """Protocol defining required methods for a cache to be snapshotable."""

    @property
    def root_node(self): ...

    @property
    def token_to_kv_pool(self): ...

    @property
    def req_to_token_pool(self): ...

    def evictable_size(self) -> int: ...

    def total_size(self) -> int: ...


class SnapshotMixin:
    """Mixin class that adds snapshot capabilities to a cache implementation.

    This mixin provides methods for creating and restoring snapshots of the cache state.
    The host class must implement the SnapshotableCache protocol.

    Usage:
        class MyCache(SnapshotMixin, BaseCacheClass):
            def __init__(self):
                super().__init__()
                self._init_snapshot()
    """

    def _init_snapshot(self) -> None:
        """Initialize snapshot-related state.

        Call this in the host class's __init__ after super().__init__().
        """
        self._snapshot_lock = SnapshotLock()

    def create_snapshot(
        self,
        snapshot_dir: Path,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SnapshotMetadata:
        """Create a snapshot of the current cache state.

        Args:
            snapshot_dir: Directory to store the snapshot
            name: User-provided name for the snapshot
            description: Optional description
            tags: Optional tags for categorization

        Returns:
            Metadata about the created snapshot

        Raises:
            SnapshotInProgressError: If another snapshot operation is in progress
            ValueError: If the snapshot directory already exists
            RuntimeError: If snapshot is not initialized
            TypeError: If host class doesn't implement SnapshotableCache
        """
        if not hasattr(self, "_snapshot_lock"):
            raise RuntimeError(
                "Snapshot not initialized. Did you call _init_snapshot()?"
            )

        # Type check to ensure host class implements required protocol
        if not isinstance(self, SnapshotableCache):
            raise TypeError(
                f"{self.__class__.__name__} must implement SnapshotableCache protocol"
            )

        with self._snapshot_lock:
            serializer = RadixTreeSerializer(self)  # type: ignore
            return serializer.create_snapshot(
                snapshot_dir=snapshot_dir, name=name, description=description, tags=tags
            )

    def restore_snapshot(self, snapshot_dir: Path) -> SnapshotMetadata:
        """Restore cache state from a snapshot.

        Args:
            snapshot_dir: Directory containing the snapshot

        Returns:
            Metadata about the restored snapshot

        Raises:
            SnapshotInProgressError: If another snapshot operation is in progress
            FileNotFoundError: If snapshot directory doesn't exist
            ValueError: If snapshot is invalid or incompatible
            RuntimeError: If snapshot is not initialized
            TypeError: If host class doesn't implement SnapshotableCache
        """
        if not hasattr(self, "_snapshot_lock"):
            raise RuntimeError(
                "Snapshot not initialized. Did you call _init_snapshot()?"
            )

        # Type check to ensure host class implements required protocol
        if not isinstance(self, SnapshotableCache):
            raise TypeError(
                f"{self.__class__.__name__} must implement SnapshotableCache protocol"
            )

        with self._snapshot_lock:
            deserializer = RadixTreeDeserializer(self)  # type: ignore
            return deserializer.restore_snapshot(snapshot_dir)
