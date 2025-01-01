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
Core data structures for the radix tree implementation.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional


class TreeNode:
    """A node in the radix tree.

    Each node represents a sequence of tokens and their corresponding KV cache values.
    The tree structure allows for efficient prefix matching and cache reuse.
    """

    def __init__(self):
        """Initialize a new TreeNode."""
        self.children: Dict[int, TreeNode] = defaultdict(
            TreeNode
        )  # Child nodes keyed by first token
        self.parent: Optional[TreeNode] = None  # Parent node reference
        self.key: Optional[List[int]] = None  # Token sequence for this node
        self.value: Optional[List[int]] = None  # KV cache indices for this node
        self.lock_ref: int = 0  # Reference count for locking
        self.last_access_time: float = time.time()  # Last time this node was accessed

    def __lt__(self, other: TreeNode) -> bool:
        """Compare nodes by last access time for heap operations."""
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List) -> int:
    """Find the length of the matching prefix between two keys.

    Args:
        key0: First key sequence
        key1: Second key sequence

    Returns:
        Length of the matching prefix
    """
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i
