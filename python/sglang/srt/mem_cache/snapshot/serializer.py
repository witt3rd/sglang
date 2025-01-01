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
Serialization logic for RadixCache snapshots.
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from sglang.srt.mem_cache.radix_types import TreeNode

from .snapshot_types import NodeSnapshot, RadixTreeSnapshot, SnapshotMetadata

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache


class RadixTreeSerializer:
    """Handles serialization of RadixCache to disk."""

    SNAPSHOT_VERSION = "1.0.0"
    TREE_FILE = "tree.json"
    KV_CACHE_DIR = "kv_cache"
    METADATA_FILE = "metadata.json"

    def __init__(self, radix_cache: RadixCache):
        self.radix_cache = radix_cache

    def create_snapshot(
        self,
        snapshot_dir: Path,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SnapshotMetadata:
        """Create a complete snapshot of the radix tree and KV cache.

        Args:
            snapshot_dir: Directory to store the snapshot
            name: User-provided name for the snapshot
            description: Optional description
            tags: Optional tags for categorization

        Returns:
            Metadata about the created snapshot
        """
        # Create snapshot directory structure
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        kv_cache_dir = snapshot_dir / self.KV_CACHE_DIR
        kv_cache_dir.mkdir(exist_ok=True)

        # Create snapshot
        snapshot = self._create_tree_snapshot()

        # Save tree structure
        tree_path = snapshot_dir / self.TREE_FILE
        with tree_path.open("w") as f:
            json.dump(self._serialize_snapshot(snapshot), f, indent=2)

        # Save KV cache tensors
        total_size = self._save_kv_cache(snapshot, kv_cache_dir)

        # Create and save metadata
        metadata = SnapshotMetadata(
            id=snapshot_dir.name,
            name=name,
            description=description,
            tags=tags or [],
            created_at=time.time(),
            size_bytes=total_size,
            status="ready",
            path=snapshot_dir,
        )

        metadata_path = snapshot_dir / self.METADATA_FILE
        with metadata_path.open("w") as f:
            json.dump(self._serialize_metadata(metadata), f, indent=2)

        return metadata

    def _create_tree_snapshot(self) -> RadixTreeSnapshot:
        """Create a snapshot of the tree structure."""
        snapshot = RadixTreeSnapshot(version=self.SNAPSHOT_VERSION)

        # Start with root node
        root = self.radix_cache.root_node
        snapshot.root_node_id = self._generate_node_id(root)

        # Traverse tree and add all nodes
        self._traverse_and_snapshot(root, None, snapshot.nodes)

        # Add cache statistics
        snapshot.evictable_size = self.radix_cache.evictable_size()
        snapshot.metadata["total_size"] = self.radix_cache.total_size()

        return snapshot

    def _traverse_and_snapshot(
        self, node: TreeNode, parent_id: Optional[str], nodes: Dict[str, NodeSnapshot]
    ) -> None:
        """Recursively traverse tree and create node snapshots."""
        node_id = self._generate_node_id(node)

        # Create node snapshot
        nodes[node_id] = NodeSnapshot(
            key=node.key.tolist() if isinstance(node.key, torch.Tensor) else node.key,
            value=node.value.tolist()
            if isinstance(node.value, torch.Tensor)
            else node.value,
            children_keys=list(node.children.keys()),
            lock_ref=node.lock_ref,
            last_access_time=node.last_access_time,
            parent_id=parent_id,
        )

        # Recursively process children
        for child in node.children.values():
            self._traverse_and_snapshot(child, node_id, nodes)

    def _generate_node_id(self, node: TreeNode) -> str:
        """Generate a unique ID for a node based on its position in the tree."""
        if node.parent is None:
            return "root"

        # Create path from root to node
        path = []
        current = node
        while current.parent is not None:
            for key, child in current.parent.children.items():
                if child is current:
                    path.append(str(key))
                    break
            current = current.parent

        return "_".join(reversed(path))

    def _save_kv_cache(self, snapshot: RadixTreeSnapshot, cache_dir: Path) -> int:
        """Save KV cache tensors to disk.

        Returns:
            Total size in bytes of saved tensors
        """
        total_size = 0

        for node_id, node in snapshot.nodes.items():
            if node.value:
                # Get tensors from memory pool for each layer
                tensors = []
                for layer_id in range(self.radix_cache.token_to_kv_pool.layer_num):
                    k = self.radix_cache.token_to_kv_pool.get_key_buffer(layer_id)[node.value]
                    v = self.radix_cache.token_to_kv_pool.get_value_buffer(layer_id)[node.value]
                    tensors.extend([k, v])

                # Save tensors
                tensor_path = cache_dir / f"{node_id}.pt"
                torch.save(torch.stack(tensors), tensor_path)

                total_size += tensor_path.stat().st_size

        return total_size

    def _serialize_snapshot(self, snapshot: RadixTreeSnapshot) -> Dict:
        """Convert snapshot to JSON-serializable dict."""
        return {
            "version": snapshot.version,
            "nodes": {
                node_id: {
                    "key": node.key,
                    "value": node.value,
                    "children_keys": node.children_keys,
                    "lock_ref": node.lock_ref,
                    "last_access_time": node.last_access_time,
                    "parent_id": node.parent_id,
                }
                for node_id, node in snapshot.nodes.items()
            },
            "root_node_id": snapshot.root_node_id,
            "evictable_size": snapshot.evictable_size,
            "timestamp": snapshot.timestamp,
            "metadata": snapshot.metadata,
        }

    def _serialize_metadata(self, metadata: SnapshotMetadata) -> Dict:
        """Convert metadata to JSON-serializable dict."""
        return {
            "id": metadata.id,
            "name": metadata.name,
            "description": metadata.description,
            "tags": metadata.tags,
            "created_at": metadata.created_at,
            "size_bytes": metadata.size_bytes,
            "status": metadata.status,
            "error": metadata.error,
            "path": str(metadata.path) if metadata.path else None,
        }
