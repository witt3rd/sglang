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
Deserialization logic for RadixCache snapshots.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import torch
from sglang.srt.mem_cache.radix_types import TreeNode

from .serializer import RadixTreeSerializer
from .snapshot_types import NodeSnapshot, RadixTreeSnapshot, SnapshotMetadata

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache


class RadixTreeDeserializer:
    """Handles deserialization of RadixCache from disk."""

    def __init__(self, radix_cache: RadixCache):
        self.radix_cache = radix_cache
        self.serializer = RadixTreeSerializer(radix_cache)

    def restore_snapshot(self, snapshot_dir: Path) -> SnapshotMetadata:
        """Restore the radix tree and KV cache from a snapshot.

        Args:
            snapshot_dir: Directory containing the snapshot

        Returns:
            Metadata about the restored snapshot

        Raises:
            ValueError: If snapshot is invalid or incompatible
            FileNotFoundError: If snapshot files are missing
        """
        # Load and validate metadata
        metadata = self._load_metadata(snapshot_dir)
        if metadata.status != "ready":
            raise ValueError(f"Cannot restore snapshot with status: {metadata.status}")

        # Load tree structure
        tree_path = snapshot_dir / self.serializer.TREE_FILE
        if not tree_path.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")

        with tree_path.open("r") as f:
            tree_data = json.load(f)

        # Validate version compatibility
        if tree_data["version"] != self.serializer.SNAPSHOT_VERSION:
            raise ValueError(
                f"Incompatible snapshot version: {tree_data['version']} "
                f"(expected {self.serializer.SNAPSHOT_VERSION})"
            )

        # Create snapshot object
        snapshot = self._deserialize_snapshot(tree_data)

        # Clear existing cache
        self._clear_cache()

        # Reconstruct tree
        self._reconstruct_tree(snapshot)

        # Load KV cache tensors
        self._load_kv_cache(snapshot, snapshot_dir / self.serializer.KV_CACHE_DIR)

        return metadata

    def _load_metadata(self, snapshot_dir: Path) -> SnapshotMetadata:
        """Load and validate snapshot metadata."""
        metadata_path = snapshot_dir / self.serializer.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open("r") as f:
            data = json.load(f)

        return SnapshotMetadata(
            snapshot_id=data["snapshot_id"],
            name=data["name"],
            description=data["description"],
            tags=data["tags"],
            created_at=data["created_at"],
            size_bytes=data["size_bytes"],
            status=data["status"],
            error=data["error"],
            path=Path(data["path"]) if data["path"] else None,
        )

    def _deserialize_snapshot(self, data: Dict) -> RadixTreeSnapshot:
        """Convert JSON data back to RadixTreeSnapshot."""
        nodes = {
            node_id: NodeSnapshot(
                key=node["key"],
                value=node["value"],
                children_keys=node["children_keys"],
                lock_ref=node["lock_ref"],
                last_access_time=node["last_access_time"],
                parent_id=node["parent_id"],
            )
            for node_id, node in data["nodes"].items()
        }

        return RadixTreeSnapshot(
            version=data["version"],
            nodes=nodes,
            root_node_id=data["root_node_id"],
            evictable_size=data["evictable_size"],
            timestamp=data["timestamp"],
            metadata=data["metadata"],
        )

    def _clear_cache(self) -> None:
        """Clear the existing cache state."""
        # Reset root node
        self.radix_cache.root_node = TreeNode()
        # Reset eviction tracking
        self.radix_cache.evictable_size_ = 0
        # Free all memory in pools
        self.radix_cache.token_to_kv_pool.reset()
        self.radix_cache.req_to_token_pool.reset()

    def _reconstruct_tree(self, snapshot: RadixTreeSnapshot) -> None:
        """Reconstruct the tree structure from snapshot."""
        # Start with root node
        root_snapshot = snapshot.nodes[snapshot.root_node_id]
        root = self.radix_cache.root_node
        self._reconstruct_node(root, root_snapshot)

        # Process all other nodes in dependency order
        processed = {snapshot.root_node_id}
        while len(processed) < len(snapshot.nodes):
            for node_id, node_snapshot in snapshot.nodes.items():
                if node_id in processed:
                    continue
                if node_snapshot.parent_id in processed:
                    parent_node = self._find_node(node_snapshot.parent_id)
                    new_node = TreeNode()
                    new_node.parent = parent_node
                    # Find the key in parent's children that points to this node
                    for key in node_snapshot.children_keys:
                        if str(key) in node_id:
                            parent_node.children[key] = new_node
                            break
                    self._reconstruct_node(new_node, node_snapshot)
                    processed.add(node_id)

    def _reconstruct_node(self, node: TreeNode, snapshot: NodeSnapshot) -> None:
        """Reconstruct a single node from its snapshot."""
        node.key = snapshot.key
        node.value = snapshot.value
        node.lock_ref = snapshot.lock_ref
        node.last_access_time = snapshot.last_access_time

    def _find_node(self, node_id: str) -> TreeNode:
        """Find a node in the tree by its ID."""
        if node_id == "root":
            return self.radix_cache.root_node

        # Parse path from node_id
        path = node_id.split("_")
        current = self.radix_cache.root_node
        for key in path:
            current = current.children[int(key)]
        return current

    def _load_kv_cache(self, snapshot: RadixTreeSnapshot, cache_dir: Path) -> None:
        """Load KV cache tensors from disk."""
        if not cache_dir.exists():
            raise FileNotFoundError(f"KV cache directory not found: {cache_dir}")

        for node_id, node in snapshot.nodes.items():
            if node.value:
                tensor_path = cache_dir / f"{node_id}.pt"
                if not tensor_path.exists():
                    raise FileNotFoundError(f"KV cache tensor not found: {tensor_path}")

                # Load tensors
                tensors = torch.load(tensor_path)
                num_layers = len(tensors) // 2  # Each layer has k and v tensors

                # Restore tensors to memory pool
                for layer_id in range(num_layers):
                    k_idx = layer_id * 2
                    v_idx = k_idx + 1
                    k = tensors[k_idx]
                    v = tensors[v_idx]

                    # Write to the pool's buffers
                    self.radix_cache.token_to_kv_pool.k_buffer[layer_id][node.value] = k
                    self.radix_cache.token_to_kv_pool.v_buffer[layer_id][node.value] = v
