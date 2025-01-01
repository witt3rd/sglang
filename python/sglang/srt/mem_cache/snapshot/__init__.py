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
RadixCache snapshot module for saving and restoring cache state.

This module provides functionality to create snapshots of the RadixCache state
and restore them later. This is useful for:
- Saving cache state between runs
- Debugging and analysis
- State migration between instances

Usage:
    # Add snapshot support to your cache
    class MyCache(SnapshotMixin, BaseCacheClass):
        def __init__(self):
            super().__init__()
            self._init_snapshot()

    # Create a snapshot
    cache = MyCache()
    metadata = cache.create_snapshot(
        snapshot_dir=Path("/path/to/snapshots/snapshot1"),
        name="pre_training",
        description="Cache state before training",
        tags=["training"]
    )

    # Restore from a snapshot
    metadata = cache.restore_snapshot(
        snapshot_dir=Path("/path/to/snapshots/snapshot1")
    )
"""

from .deserializer import RadixTreeDeserializer
from .lock import SnapshotInProgressError, SnapshotLock
from .mixin import SnapshotableCache, SnapshotMixin
from .serializer import RadixTreeSerializer
from .snapshot_types import NodeSnapshot, RadixTreeSnapshot, SnapshotMetadata

__all__ = [
    # Main functionality
    "SnapshotMixin",
    "SnapshotableCache",
    # Core types
    "NodeSnapshot",
    "RadixTreeSnapshot",
    "SnapshotMetadata",
    # Implementation classes
    "RadixTreeSerializer",
    "RadixTreeDeserializer",
    # Lock management
    "SnapshotLock",
    "SnapshotInProgressError",
]
