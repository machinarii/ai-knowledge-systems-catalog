# ML Compute Platforms: A Technical Reference

Every ML compute platform follows the same layered architecture. The hardware differs, but the template is universal: raw accelerator at the bottom, friendly frameworks at the top, with optimization layers in between.

---

## The Universal Stack Pattern

```
┌─────────────────────────────────────┐
│  6. SERVING & DEPLOYMENT            │  ← HTTP API, batching, model management
├─────────────────────────────────────┤
│  5. INFERENCE OPTIMIZATION          │  ← Graph fusion, quantization, compilation
├─────────────────────────────────────┤
│  4. ML FRAMEWORKS                   │  ← PyTorch, TensorFlow, JAX, MLX
├─────────────────────────────────────┤
│  3. KERNEL LANGUAGE                 │  ← Write custom GPU/accelerator programs
├─────────────────────────────────────┤
│  2. OPTIMIZED KERNEL LIBRARIES      │  ← Pre-built fast math (matmul, conv, FFT)
├─────────────────────────────────────┤
│  1. COMPUTE API                     │  ← Talk to the hardware
├─────────────────────────────────────┤
│  0. HARDWARE                        │  ← The actual silicon
└─────────────────────────────────────┘
```

The higher the layer, the easier to use. The lower the layer, the more performance extractable. Most ML practitioners work at L4 (frameworks). Infrastructure engineers work at L2–L3. Researchers rarely go below L4.

---

## NVIDIA

```
┌─────────────────────────────────────┐
│  Triton Server, vLLM, TGI           │  Serving
├─────────────────────────────────────┤
│  TensorRT / TensorRT-LLM            │  Inference optimization
├─────────────────────────────────────┤
│  PyTorch, TensorFlow, JAX            │  ML frameworks
├─────────────────────────────────────┤
│  Triton (the language), CUDA C++     │  Kernel language
├─────────────────────────────────────┤
│  cuBLAS, cuDNN, cuSPARSE            │  Kernel libraries
├─────────────────────────────────────┤
│  CUDA                                │  Compute API
├─────────────────────────────────────┤
│  CUDA Cores + Tensor Cores           │  Hardware
│  (RTX, Tesla, A100, H100, B200)      │
└─────────────────────────────────────┘
```

**CUDA** — NVIDIA's proprietary GPU programming API. Since 2007. GPU kernels written in CUDA C++ run on NVIDIA hardware only. The foundation that the ML ecosystem was built on.

**cuBLAS / cuDNN / cuSPARSE** — Pre-built, hand-tuned GPU kernels. cuBLAS handles linear algebra (matrix multiply — the core transformer operation). cuDNN handles neural network primitives (convolution, attention, normalization). NVIDIA employs teams whose job is tuning these per GPU architecture (Ampere, Hopper, Blackwell each get architecture-specific optimized paths).

**Triton** (the language, not the server) — OpenAI's GPU kernel language. Write GPU programs in Python-like syntax; Triton compiles to optimized CUDA. How FlashAttention was implemented. ~50 lines of Triton replaces ~500 lines of CUDA C++. PyTorch 2.0's `torch.compile` uses Triton as its backend.

**TensorRT** — NVIDIA's inference compiler. Analyzes computation graphs, fuses operations, selects optimal kernels per GPU, calibrates quantization. Output is a GPU-specific optimized binary. Much faster than vanilla PyTorch, but rigid — compiled for a specific GPU, batch size, and precision.

**NCCL** ("nickel") — Multi-GPU communication library. All-reduce, all-gather, broadcast over NVLink/PCIe/InfiniBand. Required for distributed training and multi-GPU inference.

**Triton Inference Server** (different project, confusing naming) — NVIDIA's model serving platform. Batching, model versioning, GPU scheduling. vLLM and TensorRT-LLM can run inside it.

| Pros | Cons |
|------|------|
| 17 years of ecosystem maturity — universal framework support | Proprietary — CUDA only runs on NVIDIA hardware |
| Most optimized kernel libraries (cuDNN) — consistently extracts more of theoretical FLOPS than any competitor | Vendor lock-in — migrating away requires rewriting CUDA kernels |
| Every ML library supports CUDA first, everything else second | Expensive — premium pricing on hardware, especially datacenter GPUs |
| Largest pool of CUDA-trained engineers and StackOverflow answers | CUDA version / driver / GPU generation compatibility matrix causes "CUDA hell" installation issues |
| Tensor Cores provide dedicated matrix-multiply hardware | Power consumption significantly higher than purpose-built accelerators |
| NVLink provides high-bandwidth multi-GPU interconnect | Consumer GPUs artificially limited (VRAM, features) vs datacenter SKUs |

> **Ecosystem**: Dominant since 2007. Proprietary. Self-reinforcing adoption cycle.

---

## Apple Silicon

```
┌─────────────────────────────────────┐
│  Ollama, llama.cpp server            │  Serving
├─────────────────────────────────────┤
│  Core ML                             │  Inference optimization
├─────────────────────────────────────┤
│  MLX, PyTorch (MPS backend)          │  ML frameworks
├─────────────────────────────────────┤
│  Metal Shading Language              │  Kernel language
├─────────────────────────────────────┤
│  MPS (Metal Performance Shaders)     │  Kernel libraries
├─────────────────────────────────────┤
│  Metal                               │  Compute API
├─────────────────────────────────────┤
│  Apple GPU + Neural Engine            │  Hardware
│  (M1–M4, unified memory)             │
└─────────────────────────────────────┘
```

Three distinct layers that share a namespace:

**Metal** (L1 — Compute API) — Apple's low-level GPU programming interface. Equivalent of CUDA or Vulkan. Since 2014. Compute shaders written in Metal Shading Language dispatch directly to GPU cores. llama.cpp has hand-written `.metal` shader files for transformer operations — maximum performance, no PyTorch in the path.

**MPS** (L2 — Metal Performance Shaders) — Pre-built optimized GPU kernels on top of Metal for common operations (matmul, convolution, FFT). PyTorch's `device="mps"` routes tensor operations to these kernels. Convenient but has overhead: not every op has an MPS kernel (missing ones silently fall back to CPU), and there's synchronization cost from the PyTorch→MPS translation.

**MLX** (L4 — ML Framework) — Apple's ML framework, released December 2023. Purpose-built for unified memory architecture. On Apple Silicon, CPU and GPU share the same physical memory. PyTorch with MPS still conceptually "moves" tensors between CPU/GPU (adding sync overhead even though no data moves). MLX eliminates this — ops freely run on CPU or GPU with zero-copy. Uses lazy evaluation. Has its own model format and quantization, separate from GGUF.

**Core ML** (L5 — Inference Optimization) — Apple's inference compiler. Converts models from PyTorch/TensorFlow/ONNX into optimized format. Routes computation across CPU, GPU, and Neural Engine automatically. Primarily used for on-device iOS/macOS apps.

| Pros | Cons |
|------|------|
| Unified memory — GPU/CPU share same physical RAM, no data copying | Apple-only — nothing built on Metal/MLX runs on Linux or Windows |
| Large unified memory pools (up to 512GB on M4 Ultra) enable running models that won't fit in discrete GPU VRAM | Smaller ecosystem — fewer libraries, fewer engineers, fewer answers |
| Power efficient — significantly better perf/watt than discrete GPUs | No multi-device scaling — can't link multiple Macs for distributed training |
| MLX designed from scratch for unified memory (zero-copy, lazy eval) | MPS backend in PyTorch is incomplete — silent CPU fallbacks degrade performance without warning |
| Metal shaders in llama.cpp are highly optimized for transformer inference | Neural Engine capabilities are mostly undocumented for third-party developers |
| Single-machine simplicity — no driver versioning hell | Training throughput significantly lower than equivalent-cost NVIDIA setup |

> **Ecosystem**: Metal since 2014, MLX since 2023. Proprietary. Growing fast but Apple-only.

---

## AMD

```
┌─────────────────────────────────────┐
│  vLLM (ROCm), llama.cpp (HIP)       │  Serving
├─────────────────────────────────────┤
│  (limited — in progress)             │  Inference optimization
├─────────────────────────────────────┤
│  PyTorch (ROCm backend)              │  ML frameworks
├─────────────────────────────────────┤
│  HIP (CUDA-compatible syntax)        │  Kernel language
├─────────────────────────────────────┤
│  rocBLAS, MIOpen                     │  Kernel libraries
├─────────────────────────────────────┤
│  ROCm                                │  Compute API
├─────────────────────────────────────┤
│  CDNA / RDNA Compute Units           │  Hardware
│  (MI300X, RX 7900 XTX, RX 9070 XT)  │
└─────────────────────────────────────┘
```

**ROCm** (Radeon Open Compute) — AMD's CUDA equivalent. Open-source, unlike CUDA.

**HIP** (Heterogeneous-compute Interface for Portability) — Kernel language syntactically near-identical to CUDA. A tool called `hipify` mechanically translates most CUDA code to HIP. This is how llama.cpp, PyTorch, and other CUDA-first projects get AMD support — transpiled, not rewritten.

**rocBLAS / MIOpen** — AMD's cuBLAS/cuDNN equivalents. Functional, but the kernel optimization gap vs NVIDIA is measurable. Fewer engineers tuning fewer kernels for fewer workloads.

| Pros | Cons |
|------|------|
| Open-source stack (ROCm, HIP) — no proprietary lock-in | Narrow official GPU support — mostly Instinct MI-series datacenter + select RDNA3 consumer |
| HIP is syntactically near-identical to CUDA — low migration friction | Kernel optimization gap vs cuDNN — fewer engineers, less architecture-specific tuning |
| Competitive hardware price/performance, especially consumer GPUs | Driver/kernel/framework version matrix is narrower — more edge cases, fewer StackOverflow answers |
| MI300X has 192GB HBM3 — largest memory of any single accelerator | No TensorRT equivalent — inference optimization tooling lags significantly |
| tinygrad has strong AMD support as an alternative framework | PyTorch ROCm builds sometimes lag behind CUDA builds by weeks |
| Consumer RDNA cards (RX 7900/9070) offer good VRAM/dollar | ROCm on Linux only — no Windows or macOS support |

> **Ecosystem**: Growing since 2016. Open-source. Most credible challenger to CUDA.

---

## Intel

```
┌─────────────────────────────────────┐
│  OpenVINO Model Server               │  Serving
├─────────────────────────────────────┤
│  OpenVINO                            │  Inference optimization
├─────────────────────────────────────┤
│  PyTorch (CPU/XPU), TensorFlow       │  ML frameworks
├─────────────────────────────────────┤
│  SYCL (open standard)                │  Kernel language
├─────────────────────────────────────┤
│  oneDNN (MKL-DNN)                    │  Kernel libraries
├─────────────────────────────────────┤
│  oneAPI                              │  Compute API
├─────────────────────────────────────┤
│  Xeon CPUs, Arc GPUs, Gaudi NPUs     │  Hardware
└─────────────────────────────────────┘
```

**oneAPI** — Intel's cross-architecture compute platform. Covers CPUs, GPUs (Arc), and accelerators (Gaudi). Programming model is SYCL — an open C++ standard, unlike CUDA/Metal.

**oneDNN** (formerly MKL-DNN) — Optimized neural network kernels. Very effective on Intel CPUs — PyTorch uses oneDNN automatically for CPU operations like matmul and convolution.

**OpenVINO** — Intel's TensorRT equivalent. Compresses and optimizes models for Intel hardware. Surprisingly effective for CPU inference — can significantly outperform vanilla PyTorch through graph optimization and INT8 quantization. Widely used in edge/IoT where Intel CPUs are common.

**Gaudi accelerators** (from Habana Labs acquisition) — Purpose-built AI training chips competing with NVIDIA datacenter GPUs. Cloud-only (AWS, Intel Developer Cloud).

| Pros | Cons |
|------|------|
| Open standards (SYCL, oneAPI) — no proprietary lock-in | Arc GPU ML ecosystem is thin — few production deployments |
| oneDNN makes Intel CPUs competitive for inference without any GPU | Gaudi accelerators are cloud-only — not available as purchasable hardware |
| OpenVINO is excellent for CPU-optimized inference — large deployment base in edge/IoT | SYCL adoption is slow — most developers still write CUDA first |
| Intel CPUs are everywhere — no special hardware procurement needed for CPU inference | GPU compute performance significantly behind NVIDIA and AMD at equivalent price points |
| AMX (Advanced Matrix Extensions) in Xeon adds hardware matrix acceleration | llama.cpp SYCL backend is the least-tested GPU path |
| Strong in enterprise environments where Intel infrastructure already exists | Developer community for Intel ML is small compared to CUDA |

> **Ecosystem**: Niche for GPU (2020+). Strong for CPU inference. Open standards.

---

## ARM

```
┌─────────────────────────────────────┐
│  ONNX Runtime, TFLite, llama.cpp     │  Serving
├─────────────────────────────────────┤
│  Arm NN, ACL optimization             │  Inference optimization
├─────────────────────────────────────┤
│  PyTorch, TensorFlow Lite, ONNX RT   │  ML frameworks
├─────────────────────────────────────┤
│  (C/C++ with NEON/SVE intrinsics)    │  Kernel language
├─────────────────────────────────────┤
│  Arm Compute Library (ACL)           │  Kernel libraries
├─────────────────────────────────────┤
│  NEON / SVE / SVE2 SIMD              │  Compute API (CPU SIMD)
│  Mali / Immortalis GPU               │  Compute API (GPU)
│  Ethos NPU                           │  Compute API (NPU)
├─────────────────────────────────────┤
│  Cortex-A (mobile/desktop/server)    │  Hardware
│  Cortex-M (microcontroller)          │
│  Neoverse (server/cloud)             │
│  Custom cores (Apple M-series,       │
│   Qualcomm Oryon, Samsung, etc.)     │
└─────────────────────────────────────┘
```

ARM is an instruction set architecture (ISA), not a single platform — it's licensed to dozens of companies who build their own chips. This makes the ecosystem fragmented but ubiquitous.

**NEON / SVE / SVE2** — ARM's SIMD (Single Instruction, Multiple Data) extensions. NEON is the baseline (128-bit vectors, available on virtually all ARM CPUs since ARMv7). SVE (Scalable Vector Extension) and SVE2 support variable-length vectors up to 2048 bits — critical for server-class ML inference on Neoverse/Graviton chips.

**Arm Compute Library (ACL)** — Optimized math kernels for ARM CPUs and Mali GPUs. The ARM equivalent of cuBLAS/oneDNN. Covers gemm, convolution, pooling, activation functions. Used by Arm NN and frameworks like TensorFlow Lite under the hood.

**Arm NN** — Inference engine that routes operations across CPU (NEON/SVE), GPU (Mali/Immortalis), and NPU (Ethos) based on what's available and optimal. Acts as a hardware abstraction layer.

**Mali / Immortalis GPU** — ARM's GPU designs, licensed to chip makers. Used in most Android phones and many embedded systems. OpenCL and Vulkan compute capable. Immortalis (2022+) added hardware ray tracing and improved compute throughput.

**Ethos NPU** — ARM's dedicated neural processing unit for edge inference. Ethos-U55/U65 target microcontrollers (Cortex-M), Ethos-N78 targets mobile/embedded (Cortex-A). INT8/INT16 optimized. Very power efficient but narrow workload support.

**Key distinction**: Apple Silicon (M1–M4) uses ARM ISA but with fully custom cores and its own GPU — the Metal/MPS/MLX stack described above, not ARM's ACL/Mali stack. Same instruction set, completely different software ecosystem. Similarly, Qualcomm's Snapdragon uses ARM ISA with custom Oryon CPU cores and Adreno GPU — its own QNN stack, not ARM's. AWS Graviton uses ARM Neoverse cores with standard Linux/ACL tooling.

| Pros | Cons |
|------|------|
| Ubiquitous — billions of ARM devices deployed (phones, tablets, SBCs, servers, IoT) | Fragmented ecosystem — every licensee has different GPU/NPU/software stack |
| Excellent power efficiency — best perf/watt for inference at edge | No unified ML software stack across ARM vendors (unlike CUDA for NVIDIA) |
| llama.cpp runs well on ARM CPU via NEON — no GPU needed for small models | Mali GPU compute significantly weaker than Apple GPU or Qualcomm Adreno |
| SVE/SVE2 on server-class ARM (Graviton, Neoverse) enables competitive cloud inference | Training on ARM is not practical — no equivalent of CUDA Tensor Cores |
| Ethos NPU enables ultra-low-power inference on microcontrollers | Ethos NPU supports limited model architectures — not general-purpose |
| AWS Graviton instances offer competitive price/performance for inference | ARM GPU drivers for ML (OpenCL) are less mature than CUDA or Metal |
| Raspberry Pi 5 / Hailo-8L NPU combination enables capable edge inference | Most ML frameworks treat ARM as a secondary target — CPU fallback, not optimized |

> **Ecosystem**: Mature for mobile (2010+), growing for server (2018+ with Graviton), early for ML training. ISA is open (licensed), implementations are proprietary.

---

## Qualcomm

```
┌─────────────────────────────────────┐
│  ONNX Runtime (QNN provider)         │  Serving
├─────────────────────────────────────┤
│  QNN SDK, AI Hub                     │  Inference optimization
├─────────────────────────────────────┤
│  TensorFlow Lite, ONNX Runtime       │  ML frameworks
├─────────────────────────────────────┤
│  Hexagon SDK                         │  Kernel language
├─────────────────────────────────────┤
│  (proprietary kernels)               │  Kernel libraries
├─────────────────────────────────────┤
│  Qualcomm AI Engine                  │  Compute API
├─────────────────────────────────────┤
│  Hexagon DSP + Adreno GPU + NPU      │  Hardware
│  (Snapdragon chips)                   │
└─────────────────────────────────────┘
```

Qualcomm's AI stack runs on Snapdragon chips, found in most flagship Android phones, some Windows laptops (Snapdragon X Elite/Plus), and XR headsets. Uses ARM ISA but with custom Oryon CPU cores, proprietary Adreno GPU, and Hexagon DSP/NPU.

| Pros | Cons |
|------|------|
| Integrated AI across CPU + GPU + DSP + NPU — automatic workload routing | Proprietary SDK — QNN/Hexagon tools are closed-source |
| Snapdragon X Elite/Plus brings NPU to Windows laptops (45+ TOPS) | Model conversion required — ONNX → QNN compilation step |
| Excellent power efficiency for on-device inference | Development tooling is less mature and less documented than CUDA/Metal |
| Adreno GPU is significantly more capable than ARM Mali for compute | PC/laptop Snapdragon ecosystem is still early — limited software support |
| Large installed base in mobile — billions of Snapdragon devices | Server/cloud presence is minimal — phone/laptop/edge only |
| AI Hub provides pre-optimized model library for common architectures | Community is small — few third-party resources, tutorials, or open-source projects |

> **Ecosystem**: Mature for mobile (2018+), early for PC (2024+). Proprietary.

---

## Google

```
┌─────────────────────────────────────┐
│  Vertex AI, Cloud TPU API            │  Serving
├─────────────────────────────────────┤
│  XLA (compiler)                      │  Inference optimization
├─────────────────────────────────────┤
│  JAX, TensorFlow, PyTorch (limited)  │  ML frameworks
├─────────────────────────────────────┤
│  StableHLO / HLO                     │  Kernel language
├─────────────────────────────────────┤
│  (XLA handles this internally)       │  Kernel libraries
├─────────────────────────────────────┤
│  TPU API                             │  Compute API
├─────────────────────────────────────┤
│  TPU v4, v5e, v5p, v6e               │  Hardware
│  (systolic arrays)                    │
└─────────────────────────────────────┘
```

**XLA** (Accelerated Linear Algebra) — Google's ML compiler. Takes a computation graph, optimizes, generates device-specific code. Backend for JAX and TensorFlow.

**TPUs** (Tensor Processing Units) — Google's custom ML accelerators. Cloud-only (Google Cloud). Systolic array architecture optimized for matrix multiplication. JAX is the primary programming model; PyTorch support is second-class.

**Edge TPU / Coral** — Google's edge inference accelerator. USB or module form factor. INT8 models only. Runs TensorFlow Lite models compiled via the Edge TPU Compiler. Low power (~2W), fixed-function.

| Pros | Cons |
|------|------|
| TPUs offer excellent matrix multiply throughput — competitive with NVIDIA for training | Cannot purchase TPUs — cloud-only (Google Cloud) |
| XLA compiler produces highly optimized code automatically | JAX is the first-class citizen — PyTorch on TPU is less polished |
| TPU pods enable very large-scale distributed training | Debugging on TPU is harder — fewer tools, less visibility than CUDA |
| Competitive cloud pricing vs NVIDIA GPU instances for sustained training | Vendor lock-in to Google Cloud infrastructure |
| Edge TPU (Coral) enables cheap, low-power edge inference | Edge TPU is INT8-only — limited model architecture support |
| JAX's functional paradigm produces cleaner, more reproducible code | XLA compilation adds significant startup latency (cold start problem) |

> **Ecosystem**: Mature for cloud training (2015+). Proprietary hardware. JAX/TensorFlow-first.

---

## Emerging / Specialized Accelerators

### Groq (LPU)

Groq's Language Processing Unit uses a deterministic, compiler-driven architecture — no caches, no branch prediction, no dynamic scheduling. Everything is statically scheduled at compile time. This eliminates memory bandwidth bottlenecks and produces predictable latency.

| Pros | Cons |
|------|------|
| Extremely fast inference — record-setting tokens/second for LLMs | Cloud-only API — no purchasable hardware |
| Deterministic latency — no variance between requests | Cannot train models — inference only |
| No memory bandwidth bottleneck by design | Model support limited to what Groq has compiled |
| Simple pricing model (per-token) | Architectural constraints limit flexibility for novel model architectures |

### Cerebras (Wafer-Scale)

Cerebras builds wafer-scale engines — entire silicon wafers as single chips. The CS-3 has 4 trillion transistors, 900,000 cores, and 44GB of on-chip SRAM. Eliminates memory hierarchy by putting everything on-die.

| Pros | Cons |
|------|------|
| Eliminates off-chip memory bottleneck entirely | Extremely expensive hardware — datacenter-only |
| Excellent for sparse models and large-scale training | Cloud-only access (Cerebras Cloud) |
| Can train models that don't fit in GPU memory without parallelism | Narrow software ecosystem — custom SDK required |

### Tenstorrent

Founded by Jim Keller (architect of AMD Zen, Apple A-series). Open-source RISC-V based AI accelerators. Focuses on datacenter inference with the Wormhole and Grayskull chips.

| Pros | Cons |
|------|------|
| Open-source ISA (RISC-V) — no licensing fees | Early stage — limited production deployments |
| Designed by one of the most respected chip architects | Software stack (TT-Metalium, TT-Buda) is immature vs CUDA |
| Competitive price/performance claims | Small developer community |

### Hailo

Edge AI processors (Hailo-8, Hailo-8L, Hailo-15) designed for embedded inference. Dataflow architecture — the model graph is compiled directly onto the chip's structure.

| Pros | Cons |
|------|------|
| Very power efficient — 26 TOPS at ~2.5W (Hailo-8) | Inference-only — no training capability |
| Pairs well with Raspberry Pi 5 via M.2 HAT | INT8/INT16 only — quantization required |
| Good for continuous inference workloads (video, sensors) | Compilation toolchain is complex — model conversion is non-trivial |
| Competitive with Google Coral but more flexible | Limited to models that fit the dataflow architecture |

### Samsung (Exynos NPU)

Samsung's in-house NPU integrated into Exynos SoCs for Galaxy devices. Uses ARM ISA with custom NPU blocks.

| Pros | Cons |
|------|------|
| On-device inference in Samsung phones/tablets | Proprietary SDK, minimal third-party documentation |
| Integrated into widely deployed consumer hardware | Only available in Exynos-based devices (not Snapdragon Galaxy models) |
| ONE (On-device Neural Engine) compiler is open-source | Narrow model architecture support |

### Huawei Ascend (CANN)

Huawei's AI accelerator line — Ascend 910B (training), Ascend 310 (inference). CANN (Compute Architecture for Neural Networks) is the software stack. Significant in China where US export controls restrict NVIDIA GPU access.

| Pros | Cons |
|------|------|
| Primary alternative to NVIDIA in China | Export-controlled — limited availability outside China |
| Ascend 910B competitive with A100 for training | CANN software ecosystem is much smaller than CUDA |
| Strong government/institutional backing | English documentation and community support is sparse |
| MindSpore framework provides end-to-end pipeline | Most Western ML frameworks have limited/no Ascend support |

---

## Comparative Analysis

### Performance (Datacenter / High-End)

Approximate peak specs for flagship accelerators. Real-world performance depends heavily on software optimization — paper specs are necessary but not sufficient. NVIDIA's out-of-box experience consistently matches specs; AMD's requires significant tuning to reach advertised numbers.

| Accelerator | FP16/BF16 TFLOPS | Memory | Bandwidth | LLM Inference (tok/s, Llama 70B) | TDP |
|---|---|---|---|---|---|
| **NVIDIA B200** | ~2,250 (sparse) | 192GB HBM3e | 8 TB/s | Fastest overall | 1000W |
| **NVIDIA H200** | ~990 (sparse) | 141GB HBM3e | 4.8 TB/s | ~10% over H100 | 700W |
| **NVIDIA H100 SXM** | ~990 (sparse) | 80GB HBM3 | 3.35 TB/s | Baseline datacenter | 700W |
| **AMD MI300X** | ~1,300 (dense) | 192GB HBM3 | 5.3 TB/s | ~40% lower latency than H100 for memory-bound models | 750W |
| **AMD MI325X** | ~1,300 (dense) | 256GB HBM3e | 6 TB/s | Competitive with H200 | 750W |
| **Google TPU v5p** | ~459 (bf16) | 95GB HBM2e | 2.76 TB/s | Competitive (JAX-optimized) | ~450W |
| **Intel Gaudi 3** | ~1,835 (bf16) | 128GB HBM2e | 3.7 TB/s | Limited benchmarks | 600W |

### Performance (Consumer / Local Inference)

For local LLM inference, memory capacity and bandwidth matter more than peak FLOPS. The model must fit in memory; after that, tokens/second scales with bandwidth.

| Hardware | Memory | Bandwidth | Typical tok/s (Llama 8B, Q4) | Typical tok/s (70B, Q4) | TDP |
|---|---|---|---|---|---|
| **NVIDIA RTX 4090** | 24GB GDDR6X | 1 TB/s | ~120 | Doesn't fit (24GB) | 450W |
| **NVIDIA RTX 5090** | 32GB GDDR7 | 1.79 TB/s | ~150+ | Marginal fit | 575W |
| **AMD RX 7900 XTX** | 24GB GDDR6 | 960 GB/s | ~90 (ROCm) | Doesn't fit | 355W |
| **AMD RX 9070 XT** | 16GB GDDR6 | 650 GB/s | ~60 (ROCm/Vulkan) | Doesn't fit | 300W |
| **Apple M4 Max** | 128GB unified | 546 GB/s | ~55 | ~15–20 (fits in memory) | ~75W |
| **Apple M4 Ultra** | 512GB unified | 1.09 TB/s | ~100 | ~35–40 | ~150W |
| **Intel Arc A770** | 16GB GDDR6 | 560 GB/s | ~30 (SYCL) | Doesn't fit | 225W |
| **Raspberry Pi 5 + Hailo-8L** | 8GB system + 13 TOPS NPU | — | ~5–10 (INT8, small models) | Not viable | ~15W |

The Apple Silicon trade-off: lower bandwidth per dollar than discrete GPUs, but models that don't fit in 24GB VRAM can run in 128–512GB unified memory. A 70B Q4 model runs at ~15 tok/s on M4 Max — slow, but it runs. On a 24GB RTX 4090, it doesn't run at all without offloading.

### Efficiency (Performance per Watt)

| Platform | Perf/Watt Profile | Best For |
|---|---|---|
| **Apple Silicon** | Excellent — M4 Max runs 32B Q8 at ~75W total system power | Always-on local inference, laptops, quiet operation |
| **NVIDIA datacenter** | Moderate — H100 at 700W, B200 at 1000W. High absolute performance, high absolute power | Maximum throughput where power budget is unconstrained |
| **NVIDIA consumer** | Poor — RTX 4090 at 450W for 24GB usable memory | Burst inference/training, not 24/7 operation |
| **AMD datacenter** | Moderate — MI300X at 750W, similar to NVIDIA per-token | Large model inference where 192GB memory avoids multi-GPU |
| **AMD consumer** | Moderate — better than NVIDIA consumer perf/watt, but ROCm overhead | Budget local inference with large VRAM |
| **Groq LPU** | Excellent — deterministic architecture eliminates wasted compute | High-throughput inference (cloud only) |
| **Hailo-8** | Excellent — 26 TOPS at ~2.5W | Continuous edge inference (video, sensors) |
| **ARM CPU (Graviton)** | Good — competitive inference throughput at fraction of GPU power | Cloud inference where cost/watt matters more than peak speed |
| **Google TPU** | Good — systolic arrays are efficient for sustained matrix-multiply | Large-scale training where utilization is high |

### Cost

| Platform | Hardware Cost | Cloud Hourly | Cost/Million Tokens (approx) | Notes |
|---|---|---|---|---|
| **NVIDIA H100 SXM** | ~$25–30K (if available) | $2.50–3.50/hr | $0.10–0.30 | Most competitive cloud pricing due to abundant supply |
| **NVIDIA H200** | ~$30–40K | $3.50–4.50/hr | $0.08–0.25 | Better perf/$ than H100 for most workloads |
| **NVIDIA B200** | ~$35–50K | $4.50–7.00/hr | $0.06–0.20 | Highest throughput, but high hourly rate |
| **NVIDIA RTX 4090** | ~$1,600 | N/A (consumer) | Amortized ~$0.02–0.05 | Best consumer value for CUDA inference |
| **AMD MI300X** | ~$10–15K | $3.50–5.00/hr | $0.15–0.40 | Cloud pricing elevated due to limited providers |
| **AMD RX 7900 XTX** | ~$900 | N/A | Amortized ~$0.02–0.04 | Best VRAM/dollar consumer GPU (24GB) |
| **Apple M4 Max** | ~$3,500–5,000 (Mac Studio) | N/A | Amortized ~$0.01–0.03 | Includes entire system, not just GPU |
| **Apple M4 Ultra** | ~$7,000–10,000 (Mac Studio) | N/A | Amortized ~$0.01–0.02 | 512GB unified memory is unique at this price |
| **Google TPU v5e** | Cloud only | $1.20–2.00/hr | $0.05–0.15 | Competitive for sustained training workloads |
| **Groq LPU** | Cloud only | Per-token pricing | $0.05–0.10 | No hourly rate — pay per token |
| **AWS Graviton (ARM)** | Cloud only | $0.40–1.00/hr | $0.20–0.50 (CPU inference) | Cheapest cloud option for CPU-only inference |

### RAM / Memory Considerations

| Platform | Max Memory | Memory Type | Expandable? | Key Constraint |
|---|---|---|---|---|
| **NVIDIA H100** | 80GB | HBM3 (dedicated) | No — fixed per GPU. Multi-GPU for more | 80GB limits single-GPU model size to ~40B Q8 |
| **NVIDIA H200** | 141GB | HBM3e (dedicated) | No | 141GB fits most 70B models in single GPU |
| **NVIDIA B200** | 192GB | HBM3e (dedicated) | No | Matches MI300X capacity |
| **AMD MI300X** | 192GB | HBM3 (dedicated) | No | Largest single-GPU memory until B200 |
| **Apple M4 Max** | 128GB | Unified (shared CPU/GPU) | No — fixed at purchase | All 128GB available to GPU, but shared with OS/apps (~10GB overhead) |
| **Apple M4 Ultra** | 512GB | Unified (shared CPU/GPU) | No — fixed at purchase | Largest addressable memory for local inference at any price |
| **NVIDIA RTX 4090** | 24GB | GDDR6X (dedicated) | No | Hard ceiling — model must fit in 24GB |
| **AMD RX 7900 XTX** | 24GB | GDDR6 (dedicated) | No | Same ceiling as 4090, lower bandwidth |
| **System RAM (CPU inference)** | 64–512GB+ | DDR5 (system) | Yes — add DIMMs | Slow (~50–80 GB/s) but enormous. llama.cpp CPU mode |

### Availability / Supply

| Platform | Purchase Availability (April 2026) | Lead Time | Notes |
|---|---|---|---|
| **NVIDIA H100/H200** | Available | Days–weeks (cloud), weeks (purchase) | Supply has normalized significantly since 2024 shortage |
| **NVIDIA B200** | Constrained | Weeks–months | High demand, complex packaging. Allocations favor large buyers |
| **NVIDIA RTX 4090/5090** | Available (consumer retail) | In stock at most retailers | 5090 may have brief launch-window shortages |
| **AMD MI300X/MI325X** | Available | Days–weeks | Fewer cloud providers offer AMD — limited but improving |
| **AMD RX 7900/9070** | Available (consumer retail) | In stock | No supply issues for consumer AMD GPUs |
| **Apple M4 Max/Ultra** | Available | 1–3 weeks (Apple Store) | Mac Studio configs, no supply constraints |
| **Google TPU** | Cloud only | Instant (on-demand) | No purchase option. Quota limits for new accounts |
| **Groq LPU** | Cloud only | Instant (API) | No purchase option. API rate limits apply |
| **Hailo-8/8L** | Available | In stock at distributors | M.2 modules and RPi HATs readily available |
| **Intel Arc** | Available | In stock | Low demand means abundant supply |
| **Tenstorrent** | Limited | Months | Early-stage. Developer kits available |
| **Cerebras** | Cloud only | Instant (Cerebras Cloud) | No purchase option |

### Open-Source Community & Academic Research

| Platform | OSS Community Size | Academic Papers Using Platform | Key OSS Projects | Research Adoption |
|---|---|---|---|---|
| **NVIDIA CUDA** | Dominant — largest by 10x+ | Vast majority of ML papers assume CUDA | PyTorch, TensorFlow, JAX, llama.cpp, vLLM, FlashAttention, Triton | Default platform for all major ML research labs |
| **AMD ROCm** | Growing — ~5% of CUDA's size | Increasing — MI300X appearing in benchmarks | PyTorch (ROCm), llama.cpp (HIP), tinygrad, vLLM (ROCm) | Adopted by some HPC labs (LUMI, Frontier supercomputers) |
| **Apple Metal/MLX** | Small but active | Minimal — few academic papers target Apple Silicon | MLX, llama.cpp (Metal), Ollama, swift-transformers, mlx-community models | Limited academic adoption — seen as consumer/hobbyist platform |
| **Intel oneAPI** | Small | Moderate — Intel funds academic partnerships | OpenVINO, oneDNN, llama.cpp (SYCL), Intel Extension for PyTorch | Strong in EU-funded research, edge/IoT academic work |
| **ARM** | Fragmented (per vendor) | Growing — edge AI research increasingly targets ARM | TFLite, ONNX Runtime, Arm Compute Library, llama.cpp (NEON) | Strong in embedded/edge/IoT research |
| **Google TPU** | Small (Google-centric) | Significant — Google Brain/DeepMind papers use TPU | JAX, TensorFlow, T5X, Flax, Pax | Primarily Google internal + affiliated labs |
| **Groq** | Minimal | Few | Groq API client libraries | Used in speed benchmark papers, not research platform |
| **Hailo** | Small | Growing in edge AI | HailoRT, Hailo Model Zoo | Edge inference research, RPi community projects |

### Python Library Ecosystem for AI

| Platform | PyTorch | TensorFlow | JAX | llama.cpp | Hugging Face | scikit-learn | ONNX Runtime | Notes |
|---|---|---|---|---|---|---|---|---|
| **NVIDIA CUDA** | Full support (default) | Full support | Full support | Full support | Full support | CPU only | Full support | Every library targets CUDA first |
| **AMD ROCm** | Supported (may lag 1–2 releases) | Limited | Experimental | Full support (HIP) | Via PyTorch ROCm | CPU only | ROCm EP available | PyTorch ROCm builds sometimes delayed |
| **Apple MPS** | Supported (gaps — silent CPU fallbacks) | Limited | No | N/A (uses Metal) | Via PyTorch MPS | CPU only | CoreML EP | Not all PyTorch ops have MPS kernels |
| **Apple MLX** | N/A (separate framework) | N/A | N/A | N/A (uses Metal) | mlx-community models | N/A | N/A | Own ecosystem: mlx, mlx-lm, mlx-whisper |
| **Apple Metal** | N/A (native shaders) | N/A | N/A | Full support | N/A | N/A | N/A | llama.cpp and Ollama use Metal directly |
| **Intel oneAPI** | Intel Extension for PyTorch | Full CPU support | Experimental | Full support (SYCL) | Via PyTorch/IPEX | Full CPU support (MKL) | Full support | oneDNN accelerates CPU ops silently |
| **ARM CPU** | Full CPU support (NEON) | TFLite (optimized) | CPU only | Full support (NEON/SVE) | Via PyTorch CPU | Full support | Full support (ARM EP) | ACL optimizes under the hood |
| **Google TPU** | torch_xla (second-class) | Full support (native) | Full support (native) | N/A | Via JAX/TF | N/A | N/A | JAX is the first-class path |
| **Vulkan** | N/A | N/A | N/A | Supported (experimental) | N/A | N/A | N/A | Cross-platform fallback for llama.cpp |

---

## Cross-Platform Comparison

```
              NVIDIA        Apple         AMD           Intel         ARM           Qualcomm      Google
            ──────────    ──────────    ──────────    ──────────    ──────────    ──────────    ──────────
Compute API CUDA          Metal         ROCm/HIP      oneAPI/SYCL   NEON/SVE      QAI Engine    TPU API
Kernel libs cuBLAS        MPS           rocBLAS       oneDNN        ACL           (proprietary) (XLA)
            cuDNN                       MIOpen
Kernel lang CUDA C++      Metal SL      HIP           SYCL          C + NEON      Hexagon SDK   StableHLO
            Triton                                                  intrinsics
Framework   PyTorch       MLX           PyTorch       PyTorch       TFLite        TFLite        JAX
            TF, JAX       PyTorch/MPS   (ROCm)        (CPU/XPU)     ONNX RT       ONNX RT       TF
Inference   TensorRT      Core ML       (in progress) OpenVINO      Arm NN        QNN SDK       XLA
Serving     vLLM, TGI     Ollama        vLLM (ROCm)   OV Server     llama.cpp     ONNX RT       Vertex AI
            Triton Srv    llama.cpp     llama.cpp     llama.cpp     (CPU)
Multi-GPU   NCCL          —             RCCL          oneCCL        —             —             (internal)
Open source No            No            Yes           Yes (SYCL)    ISA licensed  No            Partial
Maturity    2007+         2014/2023+    2016+         2020+         2010+ (mob)   2018+         2015+
            (dominant)    (growing)     (growing)     (niche GPU)   2018+ (srv)   (mobile)      (cloud)
Training    Excellent     Limited       Good          Limited       Impractical   No            Excellent
Inference   Excellent     Very good     Good          Good (CPU)    Good (edge)   Good (mobile) Good
Edge/mobile No            Yes (iOS)     No            Yes (OpenVINO) Yes          Yes           Yes (Coral)
```

---

## What llama.cpp Supports

llama.cpp is the most portable ML inference engine — it has backends for nearly every platform:

```
llama.cpp
  ├── CPU         (default — AVX/AVX2/AVX-512 on x86, NEON/SVE on ARM)
  ├── CUDA        (NVIDIA — most tested GPU backend)
  ├── Metal       (Apple — second most tested)
  ├── ROCm/HIP    (AMD — functional, less tested)
  ├── Vulkan      (cross-platform GPU — any GPU with Vulkan drivers)
  ├── SYCL        (Intel — least tested)
  └── KOMPUTE     (Vulkan compute — experimental)
```

This makes llama.cpp the practical benchmark for "does this hardware support ML inference" — if llama.cpp has a backend for it, the hardware is viable.

---

## The Training vs Inference Split

The platform that's best for training is often not the best for inference, and vice versa:

| | Best for Training | Best for Inference |
|---|---|---|
| **Cloud, large scale** | NVIDIA (H100/B200), Google TPU | Groq LPU, NVIDIA TensorRT, Google TPU |
| **Cloud, cost-sensitive** | AMD MI300X, Google TPU v5e | AWS Graviton (ARM), Intel Xeon + OpenVINO |
| **Local, high-end** | NVIDIA RTX 4090/5090 | Apple Silicon (M4 Max/Ultra), NVIDIA RTX |
| **Local, budget** | AMD RX 7900 XTX (ROCm) | CPU-only (llama.cpp, any platform) |
| **Edge / mobile** | Not practical | Hailo, Qualcomm NPU, Apple Neural Engine, ARM Ethos |
| **Microcontroller** | Not possible | ARM Ethos-U, TFLite Micro |

---

*April 2026. See [AI Knowledge Systems Catalog](./ai-knowledge-systems-catalog.md) for project-level platform compatibility details.*
