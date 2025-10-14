# ROCm/HIP-specific extensions and kernels
# This file is included from the main CMakeLists.txt when VLLM_TARGET_DEVICE is "rocm"

message(STATUS "Configuring ROCm extensions")

#
# _C extension - Core CUDA/HIP kernels (hipified)
#

set(VLLM_EXT_SRC
  "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"
  "csrc/cache_kernels.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/merge_attn_states.cu"
  "csrc/attention/vertical_slash_index.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/layernorm_quant_kernels.cu"
  "csrc/sampler.cu"
  "csrc/cuda_view.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/quantization/w8a8/int8/scaled_quant.cu"
  "csrc/quantization/w8a8/fp8/common.cu"
  "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/quantization/gguf/gguf_kernel.cu"
  "csrc/quantization/activation_kernels.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/custom_all_reduce.cu"
  "csrc/torch_bindings.cpp")

# Add QuickReduce kernels for ROCm
list(APPEND VLLM_EXT_SRC
  "csrc/custom_quickreduce.cu"
)

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

#
# _rocm_C extension
#
set(VLLM_ROCM_EXT_SRC
  "csrc/rocm/torch_bindings.cpp"
  "csrc/rocm/skinny_gemms.cu"
  "csrc/rocm/attention.cu")

define_gpu_extension_target(
  _rocm_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_ROCM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
