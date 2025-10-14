# CUDA-specific extensions and kernels
# This file is included from the main CMakeLists.txt when VLLM_TARGET_DEVICE is "cuda"

message(STATUS "Configuring CUDA extensions")

#
# _C extension - Core CUDA kernels
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

SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

# Set CUTLASS_REVISION. Used for FetchContent. Also fixes some bogus messages when building.
set(CUTLASS_REVISION "v4.2.1" CACHE STRING "CUTLASS revision to use")

# Use the specified CUTLASS source directory for compilation if VLLM_CUTLASS_SRC_DIR is provided
if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
  set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
endif()

if(VLLM_CUTLASS_SRC_DIR)
  if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
    get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
  endif()
  message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
  FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
else()
  FetchContent_Declare(
      cutlass
      GIT_REPOSITORY https://github.com/nvidia/cutlass.git
      # Please keep this in sync with CUTLASS_REVISION line above.
      GIT_TAG ${CUTLASS_REVISION}
      GIT_PROGRESS TRUE

      # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
      # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
      # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
      GIT_SHALLOW TRUE
  )
endif()
FetchContent_MakeAvailable(cutlass)

list(APPEND VLLM_EXT_SRC
  "csrc/quantization/awq/gemm_kernels.cu"
  "csrc/permute_cols.cu"
  "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
  "csrc/quantization/fp4/nvfp4_quant_entry.cu"
  "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
  "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"
  "csrc/cutlass_extensions/common.cpp"
  "csrc/quantization/w8a8/fp8/per_token_group_quant.cu"
  "csrc/quantization/w8a8/int8/per_token_group_quant.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

# Only build Marlin kernels if we are building for at least some compatible archs.
# Keep building Marlin for 9.0 as there are some group sizes and shapes that
# are not supported by Machete yet.
# 9.0 for latest bf16 atomicAdd PTX
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0;8.7;9.0+PTX" "${CUDA_ARCHS}")
if (MARLIN_ARCHS)

  #
  # For the Marlin kernels we automatically generate sources for various
  # preselected input type pairs and schedules.
  # Generate sources:
  set(MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/gptq_marlin/generate_kernels.py)
  file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)

  message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH}")
  message(STATUS "Last run Marlin generate script hash: $CACHE{MARLIN_GEN_SCRIPT_HASH}")

  if (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH}
      OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=$PYTHONPATH
        ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT}
      RESULT_VARIABLE marlin_generation_result
      OUTPUT_VARIABLE marlin_generation_result
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
    )

    if (NOT marlin_generation_result EQUAL 0)
      message(FATAL_ERROR "Marlin generation failed."
                          " Result: \"${marlin_generation_result}\""
                          "\nCheck the log for details: "
                          "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
    else()
      set(MARLIN_GEN_SCRIPT_HASH ${MARLIN_GEN_SCRIPT_HASH}
          CACHE STRING "Last run Marlin generate script hash" FORCE)
      message(STATUS "Marlin generation completed successfully.")
    endif()
  else()
    message(STATUS "Marlin generation script has not changed, skipping generation.")
  endif()

  file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/gptq_marlin/kernel_*.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
    CUDA_ARCHS "${MARLIN_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()

  list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

  set(MARLIN_SRCS
     "csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu"
     "csrc/quantization/gptq_marlin/gptq_marlin.cu"
     "csrc/quantization/gptq_marlin/gptq_marlin_repack.cu"
     "csrc/quantization/gptq_marlin/awq_marlin_repack.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_SRCS}"
    CUDA_ARCHS "${MARLIN_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties("csrc/quantization/gptq_marlin/gptq_marlin.cu"
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
  list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")

  message(STATUS "Building Marlin kernels for archs: ${MARLIN_ARCHS}")
else()
  message(STATUS "Not building Marlin kernels as no compatible archs found"
                 " in CUDA target architectures")
endif()

# Only build AllSpark kernels if we are building for at least some compatible archs.
cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
if (ALLSPARK_ARCHS)
  set(ALLSPARK_SRCS
     "csrc/quantization/gptq_allspark/allspark_repack.cu"
     "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
  set_gencode_flags_for_srcs(
    SRCS "${ALLSPARK_SRCS}"
    CUDA_ARCHS "${ALLSPARK_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${ALLSPARK_SRCS}")
  message(STATUS "Building AllSpark kernels for archs: ${ALLSPARK_ARCHS}")
else()
  message(STATUS "Not building AllSpark kernels as no compatible archs found"
                 " in CUDA target architectures")
endif()


set(SCALED_MM_3X_ARCHS)
# The cutlass_scaled_mm kernels for Hopper (c3x, i.e. CUTLASS 3.x) require
# CUDA 12.0 or later
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
  set(SRCS
     "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm90 as CUDA Compiler version is "
                   "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                   "later if you intend on running FP8 quantized models on "
                   "Hopper.")
  else()
    message(STATUS "Not building scaled_mm_c3x_sm90 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()


# The cutlass_scaled_mm kernels for Geforce Blackwell SM120 (c3x, i.e. CUTLASS 3.x) require
# CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS
    "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
  )
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm120 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm120 as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                   "later if you intend on running FP8 quantized models on "
                   "Blackwell.")
  else()
    message(STATUS "Not building scaled_mm_c3x_120 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()


# The cutlass_scaled_mm kernels for Blackwell SM100 (c3x, i.e. CUTLASS 3.x)
# require CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS
    "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu"
  )
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
  # Let scaled_mm_c2x know it doesn't need to build these arches
  list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
  message(STATUS "Building scaled_mm_c3x_sm100 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building scaled_mm_c3x_sm100 as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                   "later if you intend on running FP8 quantized models on "
                   "Blackwell.")
  else()
    message(STATUS "Not building scaled_mm_c3x_100 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()

#
# For the cutlass_scaled_mm kernels we want to build the c2x (CUTLASS 2.x)
# kernels for the remaining archs that are not already built for 3x.
# (Build 8.9 for FP8)
cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
  "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
# subtract out the archs that are already built for 3x
list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
if (SCALED_MM_2X_ARCHS)
  set(SRCS "csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
  message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
else()
  if (SCALED_MM_3X_ARCHS)
    message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                   " for and covered by scaled_mm_c3x")
  else()
    message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                  "in CUDA target architectures")
  endif()
endif()

#
# 2:4 Sparse Kernels

# The 2:4 sparse kernels cutlass_scaled_sparse_mm and cutlass_compressor
# require CUDA 12.2 or later (and only work on Hopper).
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/sparse/cutlass/sparse_scaled_mm_c3x.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1")
  message(STATUS "Building sparse_scaled_mm_c3x for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
    message(STATUS "Not building sparse_scaled_mm_c3x kernels as CUDA Compiler version is "
                   "not >= 12.2, we recommend upgrading to CUDA 12.2 or later "
                   "if you intend on running FP8 sparse quantized models on Hopper.")
  else()
    message(STATUS "Not building sparse_scaled_mm_c3x as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()

# The nvfp4_scaled_mm_sm120 kernels for Geforce Blackwell SM120 require
# CUDA 12.8 or later
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(FP4_ARCHS "12.0a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
  set(SRCS
    "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${FP4_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
  message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
else()
  message(STATUS "Not building NVFP4 as no compatible archs were found.")
  # clear FP4_ARCHS
  set(FP4_ARCHS)
endif()

# FP4 Archs and flags
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(FP4_ARCHS "10.0a;10.1a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
  set(SRCS
    "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
    "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
    "csrc/quantization/fp4/nvfp4_experts_quant.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
    "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${FP4_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
else()
  message(STATUS "Not building NVFP4 as no compatible archs were found.")
  # clear FP4_ARCHS
  set(FP4_ARCHS)
endif()

# CUTLASS MLA Archs and flags
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(MLA_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND MLA_ARCHS)
  set(SRCS
    "csrc/attention/mla/sm100_cutlass_mla_kernel.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${MLA_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
  # Add MLA-specific include directories only to MLA source files
  set_source_files_properties(${SRCS}
    PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
  message(STATUS "Building CUTLASS MLA for archs: ${MLA_ARCHS}")
else()
  message(STATUS "Not building CUTLASS MLA as no compatible archs were found.")
  # clear MLA_ARCHS
  set(MLA_ARCHS)
endif()

# CUTLASS MoE kernels

# The MoE kernel cutlass_moe_mm requires CUDA 12.3 or later (and ONLY works
# on Hopper). get_cutlass_(pplx_)moe_mm_data should only be compiled
# if it's possible to compile MoE kernels that use its output.
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
  message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                   "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                   "if you intend on running FP8 quantized MoE models on Hopper.")
  else()
    message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                   "if you intend on running FP8 quantized MoE models on Blackwell.")
  else()
    message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

# moe_data.cu is used by all CUTLASS MoE kernels.
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
  set(SRCS "csrc/quantization/w8a8/cutlass/moe/moe_data.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    message(STATUS "Not building moe_data as CUDA Compiler version is "
                   "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                   "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
  else()
    message(STATUS "Not building moe_data as no compatible archs found "
                   "in CUDA target architectures.")
  endif()
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
  set(SRCS "csrc/quantization/w8a8/cutlass/moe/blockwise_scaled_group_mm_sm100.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
  message(STATUS "Building blockwise_scaled_group_mm_sm100 for archs: ${SCALED_MM_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    message(STATUS "Not building blockwise_scaled_group_mm_sm100 kernels as CUDA Compiler version is "
                   "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                   "if you intend on running FP8 quantized MoE models on Blackwell.")
  else()
    message(STATUS "Not building blockwise_scaled_group_mm_sm100 as no compatible archs found "
                   "in CUDA target architectures")
  endif()
endif()

#
# Machete kernels

# The machete kernels only work on hopper and require CUDA 12.0 or later.
# Only build Machete kernels if we are building for something compatible with sm90a
cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND MACHETE_ARCHS)
  #
  # For the Machete kernels we automatically generate sources for various
  # preselected input type pairs and schedules.
  # Generate sources:
  set(MACHETE_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py)
  file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

  message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
  message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

  if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
      OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$PYTHONPATH
        ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
      RESULT_VARIABLE machete_generation_result
      OUTPUT_VARIABLE machete_generation_output
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
    )

    if (NOT machete_generation_result EQUAL 0)
      message(FATAL_ERROR "Machete generation failed."
                          " Result: \"${machete_generation_result}\""
                          "\nCheck the log for details: "
                          "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
    else()
      set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
          CACHE STRING "Last run machete generate script hash" FORCE)
      message(STATUS "Machete generation completed successfully.")
    endif()
  else()
    message(STATUS "Machete generation script has not changed, skipping generation.")
  endif()

  # Add machete generated sources
  file(GLOB MACHETE_GEN_SOURCES "csrc/quantization/machete/generated/*.cu")
  list(APPEND VLLM_EXT_SRC ${MACHETE_GEN_SOURCES})

  # forward compatible
  set_gencode_flags_for_srcs(
    SRCS "${MACHETE_GEN_SOURCES}"
    CUDA_ARCHS "${MACHETE_ARCHS}")

  list(APPEND VLLM_EXT_SRC
    csrc/quantization/machete/machete_pytorch.cu)

  message(STATUS "Building Machete kernels for archs: ${MACHETE_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
      AND MACHETE_ARCHS)
    message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                   "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                   "later if you intend on running w4a16 quantized models on "
                   "Hopper.")
  else()
    message(STATUS "Not building Machete kernels as no compatible archs "
                   "found in CUDA target architectures")
  endif()
endif()

# Only build W4A8 kernels if we are building for something compatible with sm90a
cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
  set(SRCS
     "csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu")

  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${W4A8_ARCHS}")

  list(APPEND VLLM_EXT_SRC "${SRCS}")

  message(STATUS "Building W4A8 kernels for archs: ${W4A8_ARCHS}")
else()
  if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
      AND W4A8_ARCHS)
    message(STATUS "Not building W4A8 kernels as CUDA Compiler version is "
                   "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                   "later if you intend on running w4a16 quantized models on "
                   "Hopper.")
  else()
    message(STATUS "Not building W4A8 kernels as no compatible archs "
                   "found in CUDA target architectures")
  endif()
endif()

# Hadacore kernels
cuda_archs_loose_intersection(HADACORE_ARCHS "8.0;8.9;9.0" "${CUDA_ARCHS}")
if(HADACORE_ARCHS)
  set(SRCS "csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${HADACORE_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  message(STATUS "Building hadacore")
endif()

message(STATUS "Enabling C extension.")
define_gpu_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

#
# _moe_C extension
#

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu"
  "csrc/moe/moe_wna16.cu"
  "csrc/moe/grouped_topk_kernels.cu")

set(MOE_PERMUTE_SRC
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
    "csrc/moe/moe_permute_unpermute_op.cu")

list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set(VLLM_MOE_WNA16_SRC
  "csrc/moe/moe_wna16.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_WNA16_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
# 9.0 for latest bf16 atomicAdd PTX
cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0;8.7;9.0+PTX" "${CUDA_ARCHS}")
if (MARLIN_MOE_ARCHS)

  #
  # For the Marlin MOE kernels we automatically generate sources for various
  # preselected input type pairs and schedules.
  # Generate sources:
  set(MOE_MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
  file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)

  message(STATUS "Marlin MOE generation script hash: ${MOE_MARLIN_GEN_SCRIPT_HASH}")
  message(STATUS "Last run Marlin MOE generate script hash: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH}")

  if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH}
      OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=$PYTHONPATH
        ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT}
      RESULT_VARIABLE moe_marlin_generation_result
      OUTPUT_VARIABLE moe_marlin_generation_output
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
    )

    if (NOT moe_marlin_generation_result EQUAL 0)
      message(FATAL_ERROR "Marlin MOE generation failed."
                          " Result: \"${moe_marlin_generation_result}\""
                          "\nCheck the log for details: "
                          "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
    else()
      set(MOE_MARLIN_GEN_SCRIPT_HASH ${MOE_MARLIN_GEN_SCRIPT_HASH}
          CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
      message(STATUS "Marlin MOE generation completed successfully.")
    endif()
  else()
    message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
  endif()

  file(GLOB MOE_WNAA16_MARLIN_SRC "csrc/moe/marlin_moe_wna16/*.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MOE_WNAA16_MARLIN_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MOE_WNAA16_MARLIN_SRC}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()

  list(APPEND VLLM_MOE_EXT_SRC ${MOE_WNAA16_MARLIN_SRC})

  message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_ARCHS}")
else()
  message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                 " in CUDA target architectures")
endif()

message(STATUS "Enabling moe extension.")
define_gpu_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# For CUDA we also build and ship some external projects.
include(cmake/external_projects/flashmla.cmake)
include(cmake/external_projects/qutlass.cmake)

# vllm-flash-attn should be last as it overwrites some CMake functions
include(cmake/external_projects/vllm_flash_attn.cmake)
