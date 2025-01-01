RadixAttention is a novel optimization technique that systematically manages and reuses KV cache across multiple LLM generation calls. Here's how it works:

## Core Mechanism

Rather than discarding KV cache after each generation request, RadixAttention maintains the cache in a radix tree data structure that enables efficient prefix sharing between different prompts[3]. The system retains both prompts and generation results in this tree structure, allowing for automatic identification and reuse of shared prefixes[4].

## Key Components

**Cache Management**

- Implements an LRU (Least Recently Used) eviction policy to manage memory constraints
- Uses a cache-aware scheduling policy to maximize cache hit rates
- Enables efficient prefix search, insertion, and eviction operations[3]

**Tree Structure Operations**

- Dynamically splits nodes when new requests share partial prefixes
- Consolidates system prompts, user messages, and LLM replies into tree edges
- Color-codes nodes to track their states (new, cached, or evicted)[3]

## Performance Benefits

RadixAttention provides several key advantages:

- Automatically handles various cache reuse patterns without manual configuration
- Shows no noticeable overhead even when cache hits are absent
- Significantly improves first-token latency when prefix cache hits occur[1]
- Achieves up to 5x higher throughput compared to traditional systems[7]

## Technical Implementation

The system processes requests through several stages:

1. Searches for matching prefixes in the radix tree
2. Reuses existing KV cache when matches are found
3. Splits or creates new nodes as needed
4. Evicts least recently used nodes when memory limits are reached[3]

This approach is particularly effective for complex LLM programs with multiple chained generation calls, where traditional systems would need to recompute the KV cache for each request[4].

Citations:
[1] https://lmsys.org/blog/2024-01-17-sglang/
[2] https://github.com/sgl-project/sglang/issues/906
[3] http://arxiv.org/pdf/2312.07104.pdf
[4] https://arxiv.org/html/2312.07104v1
[6] https://sky.cs.berkeley.edu/project/sglang/
[7] https://multiplatform.ai/sglang-transforming-large-language-model-performance/
[8] https://huggingface.co/papers/2312.07104
[9] https://www.marktechpost.com/2024/07/27/sglang-a-structured-generation-language-for-efficient-execution-of-complex-language-model-programs/
[10] https://rocm.blogs.amd.com/artificial-intelligence/sglang/README.html
