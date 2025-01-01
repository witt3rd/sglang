# Radix Cache Snapshotting

Let me analyze the feasibility of implementing cache snapshotting for RadixAttention based on the provided documentation and code.

First, let's understand what we would need to snapshot:

1. **Tree Structure**

- The radix tree structure itself, including all nodes and their relationships
- Each node contains:
  - children (defaultdict of TreeNode)
  - parent reference
  - key (token sequence)
  - value (KV cache tensors)
  - lock_ref count
  - last_access_time

2. **KV Cache Data**

- The actual KV cache tensors stored in GPU memory
- These are organized in a paged layout where each page represents one token

Key Considerations:

1. **Storage Format**

- The tree structure could be serialized as a JSON/protobuf format
- KV cache tensors would need efficient tensor serialization (e.g., PyTorch's save/load mechanisms)
- Need to maintain relationships between tree nodes and their corresponding KV cache data

2. **Memory Management**

- KV cache tensors can be very large
- Need efficient streaming to/from disk
- Consider compression options for the tensor data

3. **Restoration Process**

- Need to handle GPU memory allocation during restore
- Ensure proper reconstruction of parent/child relationships
- Validate cache consistency

4. **Challenges**

- Handling concurrent access during snapshot/restore
- Managing memory efficiently during the process
- Ensuring thread safety
- Dealing with version compatibility
- Validating cache integrity after restore

Here's a potential high-level design:

```plaintext
1. Snapshot Process:
   - Acquire a global lock on the cache
   - Serialize tree structure:
     * Node relationships
     * Keys and metadata
     * References to KV cache data
   - Save KV cache tensors:
     * Stream from GPU to CPU memory
     * Write to disk with compression
   - Save metadata:
     * Cache statistics
     * Version information
     * Validation checksums
   - Release lock

2. Restore Process:
   - Validate snapshot version and integrity
   - Load tree structure
   - Allocate GPU memory for KV cache
   - Stream KV cache data to GPU
   - Reconstruct node relationships
   - Initialize cache state
```

This would be a significant but feasible enhancement to RadixAttention. The main technical challenges would be:

1. Efficient serialization of large GPU tensors
2. Managing memory during the snapshot/restore process
3. Ensuring thread safety and cache consistency
4. Handling version compatibility
5. Validating cache integrity

Would you like me to elaborate on any of these aspects or discuss specific implementation details?

## Implementation Plan

I'll create a detailed implementation plan for adding snapshot and restore capabilities to RadixAttention.

### 1. Implementation Structure

```plaintext
python/sglang/srt/mem_cache/
├── radix_cache.py        (existing)
├── snapshot/
│   ├── __init__.py
│   ├── serializer.py     (new)
│   ├── deserializer.py   (new)
│   └── snapshot_types.py (new)
```

### 2. Detailed Implementation Plan

#### Phase 1: Core Snapshot Types and Utilities

1. **Define Snapshot Data Types** (`snapshot_types.py`):

```python
@dataclass
class NodeSnapshot:
    key: List[int]                # Token sequence
    value: List[int]              # KV cache indices
    children_keys: List[str]      # Keys to identify children
    lock_ref: int
    last_access_time: float

@dataclass
class RadixTreeSnapshot:
    version: str                  # Snapshot format version
    nodes: Dict[str, NodeSnapshot]  # Map of node IDs to nodes
    root_node_id: str
    evictable_size: int
    timestamp: float
    metadata: Dict[str, Any]      # Additional metadata
```

2. **Implement Serialization** (`serializer.py`):

```python
class RadixTreeSerializer:
    def __init__(self, radix_cache: RadixCache):
        self.radix_cache = radix_cache

    def _generate_node_id(self, node: TreeNode) -> str:
        # Generate unique ID for node based on path from root
        pass

    def _serialize_node(self, node: TreeNode) -> NodeSnapshot:
        # Convert TreeNode to NodeSnapshot
        pass

    def _serialize_kv_cache(self, node_values: List[torch.Tensor], path: Path):
        # Save KV cache tensors to disk
        pass

    def create_snapshot(self, snapshot_dir: Path):
        # 1. Create snapshot directory
        # 2. Acquire lock
        # 3. Serialize tree structure
        # 4. Save KV cache data
        # 5. Save metadata
        # 6. Release lock
        pass
```

3. **Implement Deserialization** (`deserializer.py`):

```python
class RadixTreeDeserializer:
    def __init__(self, radix_cache: RadixCache):
        self.radix_cache = radix_cache

    def _deserialize_node(self, snapshot: NodeSnapshot) -> TreeNode:
        # Convert NodeSnapshot back to TreeNode
        pass

    def _load_kv_cache(self, path: Path) -> Dict[str, torch.Tensor]:
        # Load KV cache tensors from disk
        pass

    def restore_snapshot(self, snapshot_dir: Path):
        # 1. Validate snapshot
        # 2. Acquire lock
        # 3. Clear existing cache
        # 4. Load tree structure
        # 5. Load KV cache data
        # 6. Reconstruct relationships
        # 7. Release lock
        pass
```

#### Phase 2: RadixCache Integration

1. **Add Snapshot Methods to RadixCache**:

```python
class RadixCache:
    def create_snapshot(self, path: Path):
        """Create a snapshot of the current cache state."""
        serializer = RadixTreeSerializer(self)
        return serializer.create_snapshot(path)

    def restore_snapshot(self, path: Path):
        """Restore cache state from a snapshot."""
        deserializer = RadixTreeDeserializer(self)
        return deserializer.restore_snapshot(path)
```

#### Phase 3: Memory Management and Safety

1. **Lock Management**:

```python
class SnapshotLock:
    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot_in_progress = False

    def acquire_for_snapshot(self):
        with self._lock:
            if self._snapshot_in_progress:
                raise SnapshotInProgressError()
            self._snapshot_in_progress = True

    def release_snapshot(self):
        with self._lock:
            self._snapshot_in_progress = False
```

2. **Memory Streaming**:

```python
class KVCacheStreamer:
    def __init__(self, chunk_size: int = 1024*1024):
        self.chunk_size = chunk_size

    def stream_to_disk(self, tensor: torch.Tensor, path: Path):
        # Stream tensor data in chunks to avoid OOM
        pass

    def stream_from_disk(self, path: Path) -> torch.Tensor:
        # Stream tensor data in chunks from disk
        pass
```

### 3. Implementation Phases

1. **Phase 1 (Core Types - Week 1)**:

   - Implement data structures and basic serialization
   - Unit tests for serialization/deserialization
   - Validation mechanisms

2. **Phase 2 (Integration - Week 2)**:

   - Integrate with RadixCache
   - Add snapshot/restore methods
   - Integration tests

3. **Phase 3 (Memory & Safety - Week 2-3)**:

   - Implement memory streaming
   - Add locking mechanisms
   - Stress tests and performance optimization

4. **Phase 4 (Testing & Documentation - Week 4)**:
   - Comprehensive testing
   - Documentation
   - Performance benchmarks

### 4. Testing Strategy

1. **Unit Tests**:

   - Serialization/deserialization correctness
   - Memory management
   - Lock behavior
   - Error handling

2. **Integration Tests**:

   - Full snapshot/restore cycles
   - Concurrent operations
   - Error recovery

3. **Performance Tests**:
   - Snapshot creation time
   - Restore time
   - Memory usage
   - Disk I/O performance

### 5. Safety Considerations

1. **Versioning**:

   - Include version in snapshot metadata
   - Version compatibility checks
   - Migration path for format changes

2. **Validation**:

   - Checksums for data integrity
   - Tree structure validation
   - KV cache consistency checks

3. **Error Handling**:
   - Atomic operations
   - Rollback mechanisms
   - Cleanup on failure

Would you like me to elaborate on any part of this implementation plan or discuss specific aspects in more detail?

## Snapshot Management

I'll design the snapshot management system and API endpoints for controlling snapshots.

### 1. FastAPI Endpoints Design

Add these endpoints to `python/sglang/srt/server.py`:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import time

# Request/Response Models
class SnapshotRequest(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class SnapshotResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    tags: Optional[List[str]]
    created_at: float
    size_bytes: int
    status: str  # "creating", "ready", "failed"

class SnapshotListResponse(BaseModel):
    snapshots: List[SnapshotResponse]

# API Endpoints
@app.post("/snapshots", response_model=SnapshotResponse)
async def create_snapshot(
    request: SnapshotRequest,
    background_tasks: BackgroundTasks
):
    """Create a new snapshot of the current cache state."""
    snapshot_id = f"snapshot_{int(time.time())}_{request.name}"
    snapshot_dir = get_snapshot_dir() / snapshot_id

    # Register snapshot metadata
    metadata = {
        "id": snapshot_id,
        "name": request.name,
        "description": request.description,
        "tags": request.tags,
        "created_at": time.time(),
        "status": "creating"
    }
    save_snapshot_metadata(metadata)

    # Queue snapshot creation in background
    background_tasks.add_task(
        create_snapshot_task,
        snapshot_id,
        snapshot_dir
    )

    return SnapshotResponse(**metadata)

@app.get("/snapshots", response_model=SnapshotListResponse)
async def list_snapshots():
    """List all available snapshots."""
    snapshots = load_all_snapshot_metadata()
    return SnapshotListResponse(snapshots=snapshots)

@app.get("/snapshots/{snapshot_id}", response_model=SnapshotResponse)
async def get_snapshot(snapshot_id: str):
    """Get details about a specific snapshot."""
    metadata = load_snapshot_metadata(snapshot_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return SnapshotResponse(**metadata)

@app.post("/snapshots/{snapshot_id}/restore")
async def restore_snapshot(
    snapshot_id: str,
    background_tasks: BackgroundTasks
):
    """Restore the cache state from a snapshot."""
    metadata = load_snapshot_metadata(snapshot_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    if metadata["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail="Snapshot is not ready for restore"
        )

    # Queue restore in background
    background_tasks.add_task(
        restore_snapshot_task,
        snapshot_id
    )

    return {"message": "Restore initiated"}

@app.delete("/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a snapshot."""
    metadata = load_snapshot_metadata(snapshot_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    # Delete snapshot files and metadata
    delete_snapshot_files(snapshot_id)
    delete_snapshot_metadata(snapshot_id)

    return {"message": "Snapshot deleted"}
```

### 2. Snapshot Management System

Create a new module `python/sglang/srt/snapshot_manager.py`:

```python
from pathlib import Path
import json
import shutil
import time
from typing import Dict, List, Optional

class SnapshotManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.snapshots_dir = base_dir / "snapshots"
        self.metadata_dir = base_dir / "metadata"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

    def get_snapshot_path(self, snapshot_id: str) -> Path:
        """Get the directory path for a snapshot."""
        return self.snapshots_dir / snapshot_id

    def save_metadata(self, snapshot_id: str, metadata: Dict):
        """Save snapshot metadata to disk."""
        path = self.metadata_dir / f"{snapshot_id}.json"
        with path.open('w') as f:
            json.dump(metadata, f)

    def load_metadata(self, snapshot_id: str) -> Optional[Dict]:
        """Load snapshot metadata from disk."""
        path = self.metadata_dir / f"{snapshot_id}.json"
        if not path.exists():
            return None
        with path.open('r') as f:
            return json.load(f)

    def list_snapshots(self) -> List[Dict]:
        """List all available snapshots."""
        snapshots = []
        for path in self.metadata_dir.glob("*.json"):
            with path.open('r') as f:
                snapshots.append(json.load(f))
        return snapshots

    def delete_snapshot(self, snapshot_id: str):
        """Delete a snapshot and its metadata."""
        # Delete snapshot files
        snapshot_path = self.get_snapshot_path(snapshot_id)
        if snapshot_path.exists():
            shutil.rmtree(snapshot_path)

        # Delete metadata
        metadata_path = self.metadata_dir / f"{snapshot_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()

    def cleanup_old_snapshots(self, max_snapshots: int = 5):
        """Remove old snapshots if we exceed max_snapshots."""
        snapshots = self.list_snapshots()
        if len(snapshots) <= max_snapshots:
            return

        # Sort by creation time
        snapshots.sort(key=lambda x: x["created_at"])

        # Remove oldest snapshots
        for snapshot in snapshots[:-max_snapshots]:
            self.delete_snapshot(snapshot["id"])
```

### 3. Background Tasks

Create a new module `python/sglang/srt/snapshot_tasks.py`:

```python
from pathlib import Path
import time
from typing import Dict

class SnapshotTask:
    def __init__(self, snapshot_manager, radix_cache):
        self.snapshot_manager = snapshot_manager
        self.radix_cache = radix_cache

    async def create(self, snapshot_id: str):
        """Create a snapshot in the background."""
        try:
            # Update status
            metadata = self.snapshot_manager.load_metadata(snapshot_id)
            metadata["status"] = "creating"
            self.snapshot_manager.save_metadata(snapshot_id, metadata)

            # Create snapshot
            snapshot_path = self.snapshot_manager.get_snapshot_path(snapshot_id)
            self.radix_cache.create_snapshot(snapshot_path)

            # Update metadata
            metadata["status"] = "ready"
            metadata["size_bytes"] = calculate_snapshot_size(snapshot_path)
            self.snapshot_manager.save_metadata(snapshot_id, metadata)

            # Cleanup old snapshots
            self.snapshot_manager.cleanup_old_snapshots()

        except Exception as e:
            # Update metadata on failure
            metadata["status"] = "failed"
            metadata["error"] = str(e)
            self.snapshot_manager.save_metadata(snapshot_id, metadata)
            raise

    async def restore(self, snapshot_id: str):
        """Restore from a snapshot in the background."""
        try:
            snapshot_path = self.snapshot_manager.get_snapshot_path(snapshot_id)
            self.radix_cache.restore_snapshot(snapshot_path)
        except Exception as e:
            # Log error and raise
            logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            raise
```

### 4. Usage Examples

```bash
# Create a snapshot
curl -X POST http://localhost:8000/snapshots \
  -H "Content-Type: application/json" \
  -d '{"name": "pre_training", "description": "Cache state before training", "tags": ["training"]}'

# List snapshots
curl http://localhost:8000/snapshots

# Restore a snapshot
curl -X POST http://localhost:8000/snapshots/snapshot_1234567890_pre_training/restore

# Delete a snapshot
curl -X DELETE http://localhost:8000/snapshots/snapshot_1234567890_pre_training
```

### 5. Key Features

1. **Asynchronous Operations**:

   - Snapshot creation/restoration runs in background
   - Non-blocking API endpoints
   - Status tracking for operations

2. **Metadata Management**:

   - Snapshot naming and tagging
   - Creation timestamps
   - Size tracking
   - Status tracking

3. **Automatic Cleanup**:

   - Configurable maximum number of snapshots
   - Automatic removal of oldest snapshots
   - Cleanup of failed snapshots

4. **Error Handling**:

   - Graceful failure handling
   - Status updates on failures
   - Cleanup on failed operations

5. **Monitoring**:
   - List available snapshots
   - Track snapshot status
   - Size information
   - Creation time tracking

Would you like me to elaborate on any of these aspects or discuss specific implementation details?

## Affected Files

I'll list all the files that would be affected by adding the snapshot feature, including new files and modifications to existing files:

### New Files

```plaintext
python/sglang/srt/
├── mem_cache/
│   └── snapshot/
│       ├── __init__.py
│       ├── serializer.py
│       ├── deserializer.py
│       └── snapshot_types.py
├── snapshot_manager.py
└── snapshot_tasks.py
```

### Modified Files

1. `python/sglang/srt/mem_cache/radix_cache.py`

   - Add snapshot/restore methods
   - Add locking mechanisms
   - Integrate with snapshot system

2. `python/sglang/srt/server.py`

   - Add new FastAPI endpoints for snapshot management
   - Add snapshot-related request/response models
   - Integrate background tasks

3. `python/sglang/srt/managers/scheduler.py`

   - Add snapshot-aware scheduling
   - Handle scheduling during snapshot/restore operations

4. `python/sglang/srt/managers/io_struct.py`
   - Add snapshot-related request/response structures

### Test Files

```plaintext
tests/srt/
├── test_radix_cache.py
    - Add snapshot/restore tests
├── test_snapshot_manager.py
    - Test snapshot management functionality
├── test_snapshot_serialization.py
    - Test serialization/deserialization
└── test_snapshot_api.py
    - Test FastAPI endpoints
```

### Documentation Files

```plaintext
docs/
├── api/
│   └── snapshot_api.md
├── developer/
│   └── snapshot_implementation.md
└── user/
    └── snapshot_management.md
```

### Configuration Files

1. `python/sglang/srt/config.py`

   - Add snapshot-related configuration options

2. `.env.example`
   - Add snapshot-related environment variables

This feature would require careful coordination across these files to ensure proper integration and functionality. Would you like me to elaborate on any specific file's changes or discuss the integration points in more detail?

## Implementation State

- [x] Design the snapshot types and serialization/deserialization logic
- [x] Implement the snapshot creation and restoration logic
- [ ] Integrate with the RadixCache and FastAPI server
- [ ] Add background tasks for snapshot operations
- [ ] Implement snapshot management API
- [ ] Test and validate the snapshot feature
- [ ] Document the snapshot feature
