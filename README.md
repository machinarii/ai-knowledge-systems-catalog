<p align="center">
  <img src="banner.jpg" alt="Total Recall Catalog" />
</p>

# Total Recall Catalog
Catalog of AI Knowledge Retrieval, Memory & RAG Systems (Last updated 2025-04-13)

## What This Is

A reference catalog mapping of AI knowledge systems that store, retrieve, and reason. The thesis: every AI knowledge system is solving a problem that biological memory already solved, just with different tradeoffs.

Humans remember through association (vector similarity), narrative (episodic memory), relationships (knowledge graphs), consolidation (sleep/dreaming), and forgetting (interference, decay). AI systems have converged on strikingly parallel architectures: vector databases for associative recall, conversation logs for episodes, graph RAG for relational reasoning, offline consolidation for memory pruning, and TTL/auto-expiry for managed forgetting.

This catalog exists because the landscape is fragmented across 100+ projects with overlapping names, unclear boundaries, and fast-changing codebases. It answers: what does each project actually do, what layer of the stack does it occupy, what hardware does it need, and how does it map to the cognitive function it's replacing.

## Who This Is For

Anyone building or evaluating AI knowledge infrastructure — particularly local-first setups where hardware compatibility (Metal vs CUDA vs CPU) determines what's even possible.

## How To Use It

Sections are ordered from low-level infrastructure (vector DBs, embedding servers) to high-level cognition (memory management, dreaming/consolidation). If you're building a stack, read bottom-up. If you're evaluating a specific project, find its category and compare within-section. The cognition mapping table at the bottom connects everything back to the human memory mechanisms each category replaces.

---

Cross-referenced against [Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory), [Awesome-Memory-for-Agents](https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents), and [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG). GPU/platform data aligned to [GPU Compute Platforms Breakdown](./gpu-compute-platforms-breakdown.md).

### GPU/Accelerator Legend

Platform terminology follows the six-layer stack model (see companion doc):

| Tag | Meaning | Stack Layer |
|-----|---------|-------------|
| `CPU` | CPU-only, no GPU acceleration | — |
| `CUDA` | NVIDIA GPU via CUDA API + cuBLAS/cuDNN kernels | L1-L2 |
| `Metal` | Apple GPU via Metal API + native shaders (llama.cpp, Ollama) | L1 |
| `MPS` | Apple GPU via Metal Performance Shaders (PyTorch `device="mps"`) | L2 via L4 |
| `MLX` | Apple GPU via Apple's ML framework (unified memory, lazy eval) | L4 |
| `ROCm` | AMD GPU via ROCm/HIP | L1-L2 |
| `Vulkan` | Cross-platform GPU via Vulkan compute | L1 |
| `SYCL` | Intel GPU via oneAPI/SYCL | L1 |
| `Any` | Platform-agnostic (SaaS API, Docker) | — |
| `⚠` | Unverified — check repo | — |

- **⭐ Stars**: Approximate GitHub stars as of April 2026. Rounded to nearest K. Sourced from repo pages, search results, and Awesome lists. Counts change daily — treat as order-of-magnitude indicators, not exact figures.

**Key distinction**: `Metal` = native GPU shaders (fast, no PyTorch) vs `MPS` = PyTorch routing through Metal Performance Shaders (convenient, overhead). Your Ollama/llama.cpp stack uses `Metal`, not `MPS`.

### Other Columns
- **Updated**: Last release as of April 2026. `~` = approximate.
- **Deploy**: `Self` = self-hosted | `Cloud` = managed service | `Both` = either

---

## 1. VECTOR DATABASES & SIMILARITY SEARCH

> **Purpose**: Store and search high-dimensional vectors (embeddings) by similarity. The foundation layer — everything else retrieves from here.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **Milvus** | 43K | [milvus-io/milvus](https://github.com/milvus-io/milvus) | Go/C++ | CPU, CUDA | Docker/K8s | ~2026-Q1 | Apache 2.0 |
| **FAISS** | 33K | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | C++/Py | CPU, CUDA | Py 3.8+, cmake | ~2026-Q1 | MIT |
| **Qdrant** | 22K | [qdrant/qdrant](https://github.com/qdrant/qdrant) | Rust | CPU | Docker/binary, 1GB+ RAM | ~2026-Q1 | Apache 2.0 |
| **Chroma** | 18K | [chroma-core/chroma](https://github.com/chroma-core/chroma) | Rust/Py | CPU | Py 3.9+ or Docker | ~2026-Q1 | Apache 2.0 |
| **Weaviate** | 15K | [weaviate/weaviate](https://github.com/weaviate/weaviate) | Go | CPU, CUDA (modules) | Docker | ~2026-Q1 | BSD-3 |
| **pgvector** | 13K | [pgvector/pgvector](https://github.com/pgvector/pgvector) | C | CPU | PostgreSQL 12+ | ~2026-Q1 | PostgreSQL |
| **LanceDB** | 5K | [lancedb/lancedb](https://github.com/lancedb/lancedb) | Rust/Py | CPU | Py 3.9+, pip | ~2026-Q1 | Apache 2.0 |
| **Vespa** | 4K | [vespa-engine/vespa](https://github.com/vespa-engine/vespa) | Java/C++ | CPU | Docker, 8GB+ RAM | ~2026-Q1 | Apache 2.0 |
| **Pinecone** | — | [pinecone.io](https://www.pinecone.io/) | SaaS | Any | API key only | Active | Proprietary |
| **Turbopuffer** | — | [turbopuffer.com](https://turbopuffer.com/) | SaaS | Any | API key, $64/mo min | Active | Proprietary |

> **Note**: Vector DBs themselves rarely need GPU. The GPU matters for the *embedding model* that feeds vectors into them. FAISS is the exception — it can use CUDA for the similarity search itself.

---

## 2. RAG FRAMEWORKS

> **Purpose**: Orchestrate the full retrieve-then-generate pipeline. Chunking, embedding, retrieval, prompt assembly, LLM call. Most are orchestrators that call external model servers.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **LangChain** | 95K | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Py/JS | CPU (calls LLM APIs) | Py 3.9+ or Node 18+ | ~2026-Q1 | MIT |
| **RAGFlow** | 65K | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Py | CPU, CUDA | Docker, 16GB+ RAM | ~2026-Q1 | Apache 2.0 |
| **LlamaIndex** | 38K | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Py | CPU (calls LLM APIs) | Py 3.9+ | ~2026-Q1 | MIT |
| **Flowise** | 35K | [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | TS | CPU (calls LLM APIs) | Node 18+ | ~2026-Q1 | Apache 2.0 |
| **DSPy** | 34K | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | Py | CPU (calls LLM APIs) | Py 3.9+ | ~2026-Q1 | MIT |
| **Haystack** | 20K | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Py | CPU, CUDA (local models) | Py 3.9+ | ~2026-Q1 | Apache 2.0 |
| **STORM** | 11K | [stanford-oval/storm](https://github.com/stanford-oval/storm) | Py | CPU (calls LLM APIs) | Py 3.11+ | ~2026-Q1 | MIT |
| **txtai** | 10K | [neuml/txtai](https://github.com/neuml/txtai) | Py | CPU, CUDA, MPS | Py 3.9+, torch | ~2025-Q4 | Apache 2.0 |
| **LLMWare** | 8K | [llmware-ai/llmware](https://github.com/llmware-ai/llmware) | Py | CPU, CUDA, MPS | Py 3.9+ | ~2025-Q4 | Apache 2.0 |
| **R2R** | 4K | [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) | Py | CPU, CUDA | Py 3.10+, Docker | ~2025-Q4 ⚠ | MIT |
| **Pathway** | 4K | [pathwaycom/pathway](https://github.com/pathwaycom/pathway) | Py/Rust | CPU | Py 3.10+, Linux/Mac | ~2026-Q1 | BSL 1.1 |
| **Morphik** | 2K | [morphik-ai/morphik-core](https://github.com/morphik-ai/morphik-core) | Py | CPU, CUDA | Py 3.10+ | ~2025-Q4 ⚠ | Apache 2.0 |

> **Note**: Most RAG frameworks are orchestrators — they call external LLM/embedding APIs and don't run models themselves. "CPU" means the framework runs on CPU; GPU usage depends on the model server it calls (Ollama, vLLM, etc.). DSPy is distinctive: it's a programmatic optimization framework (compile prompts/weights, not hand-tune) that treats RAG as composable modules. STORM is a knowledge curation system that uses multi-agent conversations to produce long-form cited articles from web or local sources.

---

## 3. GRAPH RAG & KNOWLEDGE GRAPHS

> **Purpose**: Add entity-relationship structure on top of vector retrieval. Answers questions that require connecting dots across documents ("who reported to whom during which project").


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **GraphRAG** | 20K | [microsoft/graphrag](https://github.com/microsoft/graphrag) | Py | CPU (calls LLM APIs) | Py 3.10+ | ~2026-Q1 | MIT |
| **LightRAG** | 15K | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | Py | CPU, CUDA, MPS (needs patch) | Py 3.10+, 8GB+ RAM | 2026-02 | MIT |
| **Neo4j** | 14K | [neo4j/neo4j](https://github.com/neo4j/neo4j) | Java | CPU | JDK 17+, Docker, 2GB+ heap | ~2026-Q1 | GPL-3/Comm |
| **LangGraph** | 10K | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Py/JS | CPU | Py 3.9+ or Node 18+ | ~2026-Q1 | MIT |
| **Cognee** | 5K | [topoteretes/cognee](https://github.com/topoteretes/cognee) | Py | CPU (calls LLM APIs) | Py 3.10+, SQLite/LanceDB/Kuzu | ~2026-Q1 | Apache 2.0 |
| **nano-graphrag** | 5K | [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag) | Py | CPU | Py 3.9+, minimal deps | ~2025-Q3 | MIT |
| **LinearRAG** | — | [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) | Py | CPU, CUDA | Py 3.9+. ICLR'26 | ~2025-Q4 ⚠ | MIT |
| **LogicRAG** | — | [chensyCN/LogicRAG](https://github.com/chensyCN/LogicRAG) | Py | CPU, CUDA | Py 3.9+. AAAI'26 | ~2025-Q4 ⚠ | MIT |
| **fast-graphrag** | — | [circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | MIT |

> **Note**: LightRAG MPS issue — default HF embedding code checks `torch.cuda.is_available()` and falls back to CPU, ignoring MPS. Patch: add `elif torch.backends.mps.is_available()` check. Not needed when routing embeddings through Ollama (which uses Metal natively).

---

## 4. SPECIALIZED RETRIEVAL & SEARCH

> **Purpose**: Retrieval engines that solve specific problems — hybrid search, code search, cache-augmented generation, or token-level reranking.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **Meilisearch** | 48K | [meilisearch/meilisearch](https://github.com/meilisearch/meilisearch) | Rust | CPU | Docker/binary, low RAM | ~2026-Q1 | MIT |
| **RAGatouille** | 3K | [bclavie/RAGatouille](https://github.com/bclavie/RAGatouille) | Py | CPU, CUDA, MPS | Py 3.9+, torch (ColBERT) | ~2025-Q4 | Apache 2.0 |
| **CLaRa** | 1K | [apple/ml-clara](https://github.com/apple/ml-clara) | Py | CUDA | Py 3.10+, torch, DeepSpeed | ~2025-Q4 | Apple |
| **QMD** | — | [tobi/qmd](https://github.com/tobi/qmd) | TS | CPU (node-llama-cpp) | Node 18+, ~2GB GGUF models | ~2026-Q1 ⚠ | MIT |
| **RAG-Anything** | — | [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) | Py | CPU, CUDA | Py 3.10+ | ~2025-Q4 ⚠ | MIT |
| **FlashRAG** | — | [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Py | CPU, CUDA, MPS | Py 3.9+, torch | ~2025-Q4 ⚠ | MIT |
| **PageIndex** | — | [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | Py | CPU ⚠ | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ |
| **REFRAG** | — | [simulanics/REFRAG](https://github.com/simulanics/REFRAG) | Py | CPU ⚠ | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ |
| **CAG** | — | [hhhuang/CAG](https://github.com/hhhuang/CAG) | Py | CPU | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ |
| **GitNexus** | — | [nxpatterns/gitnexus](https://github.com/nxpatterns/gitnexus) | Py | CPU ⚠ | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ |

---

## 5. DOCUMENT PREPROCESSING & INGESTION

> **Purpose**: Convert raw files (PDFs, HTML, images, web pages) into clean, chunked text that vector DBs and RAG frameworks can ingest. Garbage in, garbage out — this layer determines retrieval quality.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **Crawl4AI** | 35K | [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai) | Py | CPU | Py 3.9+, Playwright | ~2026-Q1 | Apache 2.0 |
| **Firecrawl** | 30K | [mendableai/firecrawl](https://github.com/mendableai/firecrawl) | TS | CPU | Node 18+ or SaaS API | ~2026-Q1 | AGPL-3.0 |
| **Marker** | 20K | [VikParuchuri/marker](https://github.com/VikParuchuri/marker) | Py | CPU, CUDA, MPS | Py 3.10+, torch | ~2026-Q1 | GPL-3.0 |
| **Docling** | 18K | [DS4SD/docling](https://github.com/DS4SD/docling) | Py | CPU, CUDA, MPS | Py 3.10+, torch | ~2026-Q1 | MIT |
| **Unstructured** | 10K | [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | Py | CPU, CUDA (OCR models) | Py 3.9+, tesseract/poppler opt | ~2026-Q1 | Apache 2.0 |
| **PaperQA** | 7K | [Future-House/paper-qa](https://github.com/Future-House/paper-qa) | Py | CPU (calls LLM APIs) | Py 3.11+ | ~2026-Q1 | Apache 2.0 |

---

## 6. RERANKING & RETRIEVAL REFINEMENT

> **Purpose**: Take the top-N candidates from initial retrieval and re-score them with a more expensive model. Bridges the gap between "vaguely relevant" (vector search) and "actually useful" (cross-encoder scoring).


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **FlagEmbedding** | 8K | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | Py | CPU, CUDA, MPS | Py 3.8+, torch | ~2026-Q1 | MIT |
| **ColBERT** | 4K | [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) | Py | CUDA (strong rec), CPU | Py 3.8+, torch | ~2025-Q2 | MIT |
| **FlashRank** | 2K | [PrithivirajDamodaran/FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | Py | CPU | Py 3.8+, <100MB models | ~2025-Q3 | Apache 2.0 |
| **Cohere Rerank** | — | [cohere.com/rerank](https://cohere.com/rerank) | SaaS | Any | API key | Active | Proprietary |
| **zerank-2** | — | HuggingFace GGUF | C++ | CPU, Metal, CUDA, ROCm, Vulkan | llama.cpp server, 2-8GB model | N/A | ⚠ |

> **zerank-2 on your stack**: Runs via llama.cpp on host Mac Studio → Metal backend → M4 Max GPU. The reranker is served at `127.0.0.1:8090` and ClawRAG calls it via HTTP after ChromaDB returns initial candidates.

---

## 7. EMBEDDING MODELS & INFERENCE SERVERS

> **Purpose**: Run LLMs and embedding models locally or serve them via API. The compute layer. This is where GPU acceleration (Metal, CUDA) actually matters.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **Ollama** | 110K | [ollama/ollama](https://github.com/ollama/ollama) | Go | CPU, **Metal**, CUDA, ROCm | macOS/Linux/Win, 8GB+ RAM | ~2026-Q1 | MIT |
| **llama.cpp** | 80K | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ | CPU, **Metal**, CUDA, ROCm, Vulkan, SYCL | cmake, C++17. Multi-platform | ~2026-Q1 | MIT |
| **vLLM** | 50K | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Py | **CUDA only** | Py 3.9+, CUDA 12+, 16GB+ VRAM | ~2026-Q1 | Apache 2.0 |
| **sentence-transformers** | 16K | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Py | CPU, CUDA, MPS | Py 3.8+, torch | ~2026-Q1 | Apache 2.0 |
| **FlagEmbedding** | 8K | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | Py | CPU, CUDA, MPS | Py 3.8+, torch | ~2026-Q1 | MIT |
| **TEI** | 3K | [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) | Rust | CPU, CUDA | Docker/Rust build, Linux rec | ~2026-Q1 | Apache 2.0 |

> **Metal vs MPS here**: Ollama and llama.cpp use **Metal** (native shaders, no PyTorch, fastest path). sentence-transformers and FlagEmbedding use **MPS** (PyTorch → Metal Performance Shaders → Metal, convenient but slower). vLLM has **no Apple Silicon support** — CUDA only.
>
> **Your inference path**: `Ollama → llama.cpp → Metal shaders → M4 Max GPU cores` (no PyTorch, no MPS, no MLX in the critical path)

---

## 8. AGENT MEMORY — UNIVERSAL / MULTI-TIER

> **Purpose**: Give AI agents persistent memory across sessions. Extract facts from conversations, store them durably, retrieve when relevant. The "remember me" layer.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License | Deploy |
|---------|---:|--------|------|-------|-----|---------|---------|--------|
| **Mem0** | 48K | [mem0ai/mem0](https://github.com/mem0ai/mem0) | Py/JS | CPU (calls LLM APIs) | Py 3.9+/Node. Default: gpt-4.1-nano | 2026-Q1 | Apache 2.0 | Both |
| **Letta** | 15K | [letta-ai/letta](https://github.com/letta-ai/letta) | Py | CPU (calls LLM APIs) | Py 3.10+. Runs as server | ~2026-Q1 | Apache 2.0 | Both |
| **Supermemory** | 5K | [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) | TS | CPU (internal engine) | npm/pip. MCP server. <300ms recall | ~2026-Q1 | Source-avail | Cloud/Ent |
| **Cognee** | 5K | [topoteretes/cognee](https://github.com/topoteretes/cognee) | Py | CPU (calls LLM APIs) | Py 3.10+. SQLite+LanceDB+Kuzu | ~2026-Q1 | Apache 2.0 | Self |
| **TeleMem** | — | [TeleAI-UAGI/TeleMem](https://github.com/TeleAI-UAGI/TeleMem) | Py | CPU (calls LLM APIs) | Py 3.9+. Drop-in Mem0 replacement | ~2026-Q1 | ⚠ | Self |
| **MemOS** | — | [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | ⚠ | Self |
| **MemMachine** | — | [MemMachine/MemMachine](https://github.com/MemMachine/MemMachine) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | Apache 2.0 | Self |
| **SuperLocalMemory** | — | [qualixar/superlocalmemory](https://github.com/qualixar/superlocalmemory) | Py | CPU | Py 3.9+. Zero cloud (Mode A) | ~2026-Q1 | ⚠ | Self |
| **EverMemOS** | — | [EverMind-AI/EverMemOS](https://github.com/EverMind-AI/EverMemOS) | Py | CPU | Py 3.9+ | ~2026-Q1 ⚠ | ⚠ | Self |

> **Note**: Memory frameworks are almost universally CPU-only. They store/retrieve/manage memories and call external LLM APIs for extraction/reasoning. GPU usage is delegated to whatever model server they're configured to use (Ollama, OpenAI, etc.).

---

## 9. AGENT MEMORY — CONVERSATION & EPISODIC

> **Purpose**: Track conversation history, temporal facts, and episodic sequences. Optimized for "what happened when" rather than "find similar content."


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License | Deploy |
|---------|---:|--------|------|-------|-----|---------|---------|--------|
| **Second Me** | 10K | [mindverse/Second-Me](https://github.com/mindverse/Second-Me) | Py | CPU, CUDA ⚠ | Py 3.9+ | ~2025-Q4 ⚠ | ⚠ | Self |
| **Zep/Graphiti** | 8K | [getzep/graphiti](https://github.com/getzep/graphiti) | Py | CPU | Py 3.10+, Neo4j | ~2026-Q1 | Apache 2.0 | Both |
| **Hindsight** | 3K | [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) | Py/TS/Go | CPU | 1 Docker cmd. Embedded Postgres | ~2026-Q1 | MIT | Self |
| **MemPalace** | 2K | [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) | Py | CPU | Py 3.9+, ChromaDB, SQLite | ~2025-Q4 ⚠ | ⚠ | Self |
| **Honcho** | — | [plastic-labs/honcho](https://github.com/plastic-labs/honcho) | Py | CPU | Py 3.10+ | ~2025-Q4 ⚠ | ⚠ | Self |
| **Memobase** | — | [memodb-io/memobase](https://github.com/memodb-io/memobase) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | ⚠ | Both |
| **MIRIX** | — | [Mirix-AI/MIRIX](https://github.com/Mirix-AI/MIRIX) | Py | CPU | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ | Self |
| **MemU** | — | [NevaMind-AI/memU](https://github.com/NevaMind-AI/memU) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | ⚠ | Self |
| **ReMe** | — | [modelscope/MemoryScope](https://github.com/modelscope/MemoryScope) | Py | CPU | Py 3.9+ | ~2025-Q4 ⚠ | Apache 2.0 | Self |

---

## 10. AGENT MEMORY — CODING & WORKSPACE

> **Purpose**: Persist context for coding agents (Claude Code, OpenClaw). Remember project state, decisions, and working patterns across sessions.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **claude-mem** | — | [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem) | TS | CPU | Node 18+, Claude Code plugin | ~2026-Q1 | ⚠ |
| **Memov** | — | [memov-io/memov](https://github.com/memov-io/memov) | TS | CPU | Node 18+, Git, Claude Code | ~2026-Q1 ⚠ | ⚠ |
| **LangMem** | — | [langchain-ai/langmem](https://github.com/langchain-ai/langmem) | Py | CPU | Py 3.9+, LangGraph required | ~2026-Q1 | MIT |
| **OpenMemory** | — | [caviraoss/openmemory](https://github.com/caviraoss/openmemory) | Py | CPU | Py 3.9+, MCP-native | ~2025-Q4 ⚠ | ⚠ |
| **Memori** | — | [memorilabs/memori](https://github.com/memorilabs/memori) | Py | CPU | Py 3.9+, SQL-native | ~2025-Q4 ⚠ | ⚠ |

---

## 11. ACADEMIC MEMORY SYSTEMS (Papers with Code)

> **Purpose**: Research systems from papers — not production-ready but represent where the field is heading. Many introduce novel memory architectures that later get adopted by production frameworks.


| Project | ★ | Paper | Date | Innovation |
|---------|---:|-------|------|------------|
| **A-MEM** | — | [2502.12110](https://arxiv.org/abs/2502.12110) | 2025-02 | Agentic memory, note-taking metaphor |
| **SYNAPSE** | — | [2601.02744](https://arxiv.org/abs/2601.02744) | 2026-01 | Spreading activation episodic-semantic |
| **HiMem** | — | [2601.06377](https://arxiv.org/abs/2601.06377) | 2026-01 | Hierarchical long-term memory |
| **MAGMA** | — | [2601.03236](https://arxiv.org/abs/2601.03236) | 2026-01 | Multi-graph architecture |
| **SimpleMem** | — | [2601.02553](https://arxiv.org/abs/2601.02553) | 2026-01 | Efficient lifelong memory |
| **SwiftMem** | — | [2601.08160](https://arxiv.org/abs/2601.08160) | 2026-01 | Query-aware fast indexing |
| **AtomMem** | — | [2601.08323](https://arxiv.org/abs/2601.08323) | 2026-01 | Atomic memory operations |
| **TiMem** | — | [2601.02845](https://arxiv.org/abs/2601.02845) | 2026-01 | Temporal-hierarchical consolidation |
| **Memory-R1** | — | [2512.20092](https://arxiv.org/abs/2512.20092) | 2025-12 | RL for temporal reasoning |
| **O-Mem** | — | [2511.13593](https://arxiv.org/abs/2511.13593) | 2025-11 | Omni personalized memory |
| **MemGPT** | — | [2310.08560](https://arxiv.org/abs/2310.08560) | 2023-10 | LLMs as OS (→ Letta) |
| **MemoryBank** | — | [2305.10250](https://arxiv.org/abs/2305.10250) | 2023-05 | Long-term LLM memory (early) |

---

## 12. CONTEXT & TOKEN MANAGEMENT

> **Purpose**: Manage the LLM context window — the bottleneck. KV cache reuse, token compression, context trimming. The AI equivalent of managing working memory capacity.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **SGLang** | 10K | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Py | CUDA | Py 3.9+ | ~2026-Q1 | Apache 2.0 |
| **LMCache** | — | [LMCache/LMCache](https://github.com/LMCache/LMCache) | Py | CUDA | Py 3.9+, vLLM compatible | ~2025-Q4 ⚠ | Apache 2.0 |
| **Caveman** | — | [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) | Py | CPU | Py 3.9+ | ~2025-Q3 ⚠ | ⚠ |
| **Semantic Kernel** | — | [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) | C#/Py/Java | CPU | .NET 8+ or Py 3.10+ | ~2026-Q1 | MIT |

---

## 13. DREAMING & CONSOLIDATION

> **Purpose**: Offline memory processing between sessions. Consolidate, compress, prune, and promote knowledge. Named after the biological process it mimics.


| Project | ★ | Source | Accel | Description |
|---------|---:|--------|-------|-------------|
| **OpenClaw Dreaming** | — | OpenClaw built-in | CPU (calls Ollama → Metal) | 6-signal ranking. dreams.md output |
| **autoDream** | — | OpenClaw built-in | CPU (calls Ollama → Metal) | 3-gate inter-session consolidation |
| **AAAK Compression** | — | [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace) | CPU | 30x lossless context compression |
| **Vault Architecture** | — | Pattern | — | Obsidian claim-named notes + MOCs |
| **MEMORY.md** | — | Pattern | — | Pointer index → vault on disk |

---

## 14. ORCHESTRATION & ROUTING

> **Purpose**: Route tasks to the right agent, model, or tool. The executive function layer — knows what to do without containing the knowledge itself.


| Project | ★ | GitHub | Lang | Accel | Env | Updated | License |
|---------|---:|--------|------|-------|-----|---------|---------|
| **Dify** | 70K | [langgenius/dify](https://github.com/langgenius/dify) | Py/TS | CPU | Docker, 4GB+ RAM | ~2026-Q1 | Apache 2.0 |
| **n8n** | 55K | [n8n-io/n8n](https://github.com/n8n-io/n8n) | TS | CPU | Node 18+, Docker | ~2026-Q1 | Sust. Use |
| **AutoGen** | 38K | [microsoft/autogen](https://github.com/microsoft/autogen) | Py | CPU | Py 3.9+ | ~2026-Q1 | CC-BY-4.0 |
| **CrewAI** | 25K | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Py | CPU | Py 3.10+ | ~2026-Q1 | MIT |
| **LangGraph** | 10K | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Py/JS | CPU | Py 3.9+ or Node 18+ | ~2026-Q1 | MIT |

---

## 15. EVALUATION & BENCHMARKING

> **Purpose**: Measure whether your RAG/memory system actually works. Retrieval precision, generation faithfulness, memory recall accuracy.


| Project | ★ | GitHub | Updated |
|---------|---:|--------|---------|
| **RAGAs** | 8K | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | ~2026-Q1 |
| **DeepEval** | 5K | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | ~2026-Q1 |
| **GraphRAG-Benchmark** | — | [GraphRAG-Bench/GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) | ~2026-Q1 |
| **LoCoMo** | — | [Alab-NII/LoCoMo](https://github.com/Alab-NII/LoCoMo) | ~2025-Q3 |
| **VectorDBBench** | — | [zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench) | ~2026-Q1 |

---

## 16. META-RESOURCES

| Resource | Link |
|---------|------|
| **Awesome-Agent-Memory** | [TeleAI-UAGI/Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) |
| **Awesome-Memory-for-Agents** | [TsinghuaC3I/Awesome-Memory-for-Agents](https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents) |
| **Awesome-GraphRAG** | [DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) |
| **GPU Platforms Breakdown** | [gpu-compute-platforms-breakdown.md](./gpu-compute-platforms-breakdown.md) |

---

## COGNITION MAP: Human → AI

| Human Mechanism | AI Equivalent | Projects | ★ |
|----------------|--------------|----------|---:|
| Sensory buffer | Input parsing | Unstructured, Docling, Firecrawl |
| Attention filter | Reranking | zerank-2, RAGatouille, ColBERT |
| Working memory | Context window | LMCache, Caveman, CAG, CLaRa |
| Chunking | Doc chunking | All RAG frameworks |
| Elaborative encoding | KG construction | GraphRAG, LightRAG, Cognee |
| Episodic memory | Session memory | MemPalace, Zep, Honcho, TiMem |
| Semantic memory | Vector retrieval | FAISS, Chroma, Milvus |
| Procedural memory | Agent skills | OpenClaw, LangMem |
| Method of loci | Hierarchical retrieval | MemPalace wing/hall/room |
| Relational reasoning | Graph traversal | GraphRAG, LightRAG, Neo4j |
| Sleep consolidation | Offline processing | OpenClaw dreaming, AAAK |
| Spaced repetition | Frequency signals | OpenClaw 6-signal, Stacks |
| Pattern separation | Contradiction detection | MemPalace, Supermemory, Zep |
| Metamemory | Router models | LangGraph, coordinator routing |
| Reconsolidation | Summarization drift | Memov git tracking mitigates |
| Skill acquisition | Prompt compilation | DSPy optimizers, STORM |
| Tip-of-tongue | Failed high-conf retrieval | No AI equivalent — research gap |

---