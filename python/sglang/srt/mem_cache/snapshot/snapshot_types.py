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
Data structures for RadixCache snapshotting.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NodeSnapshot:
    """Snapshot of a single node in the radix tree."""

    key: List[int]  # Token sequence
    value: List[int]  # KV cache indices
    children_keys: List[str]  # Keys to identify children
    lock_ref: int  # Reference count for locking
    last_access_time: float  # Last time this node was accessed
    parent_id: Optional[str]  # ID of parent node (None for root)


@dataclass
class RadixTreeSnapshot:
    """Complete snapshot of the radix tree state."""

    version: str = "1.0.0"  # Snapshot format version
    nodes: Dict[str, NodeSnapshot] = field(
        default_factory=dict
    )  # Map of node IDs to nodes
    root_node_id: str = "root"  # ID of the root node
    evictable_size: int = 0  # Size of evictable cache
    timestamp: float = field(default_factory=time.time)  # When snapshot was created
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class SnapshotMetadata:
    """Metadata for managing snapshots."""

    snapshot_id: str  # Unique snapshot identifier
    name: str  # User-provided name
    description: Optional[str]  # Optional description
    tags: List[str]  # Optional tags for categorization
    created_at: float  # Creation timestamp
    size_bytes: int  # Total size in bytes
    status: str  # "creating", "ready", "failed"
    error: Optional[str] = None  # Error message if status is "failed"
    path: Optional[Path] = None  # Path to snapshot directory
