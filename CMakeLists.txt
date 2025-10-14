cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example:
# mkdir build && cd build
# cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=.. ..
# cmake --build . --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build . --target _C
# cmake --install . --component _C
project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from docker/Dockerfile.rocm
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.8.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.8.0")

#
# Try to find python package with an executable that exactly matches
# `VLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# Forward device-specific extensions to external CMake scripts.
# This replaces the early return pattern for better modularity.
#
if (VLLM_TARGET_DEVICE STREQUAL "cpu")
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
  return()
elseif (VLLM_TARGET_DEVICE STREQUAL "cuda")
  # Continue to CUDA/ROCm specific setup below
elseif (VLLM_TARGET_DEVICE STREQUAL "rocm")
  # Continue to CUDA/ROCm specific setup below
else()
  message(FATAL_ERROR "Unsupported VLLM_TARGET_DEVICE: ${VLLM_TARGET_DEVICE}. "
                      "Supported values are: cuda, rocm, cpu")
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  set(VLLM_GPU_LANG "HIP")

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # ROCm 5.X and 6.X
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()


if(VLLM_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  #
  # For other GPU targets override the GPU architectures detected by cmake/torch
  # and filter them by the supported versions for the current language.
  # The final set of arches is stored in `VLLM_GPU_ARCHES`.
  #
  override_gpu_arches(VLLM_GPU_ARCHES
    ${VLLM_GPU_LANG}
    "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Set compression mode for CUDA >=13.x.
#
if(VLLM_GPU_LANG STREQUAL "CUDA" AND
   DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

#
# Set CUDA include flags for CXX compiler.
#
if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
  if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
  endif()
endif()

#
# Use FetchContent for C++ dependencies that are compiled as part of vLLM's build process.
# setup.py will override FETCHCONTENT_BASE_DIR to play nicely with sccache.
# Each dependency that produces build artifacts should override its BINARY_DIR to avoid
# conflicts between build types. It should instead be set to ${CMAKE_BINARY_DIR}/<dependency>.
#
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # Overriding the default -O set up by cmake, adding ggdb3 for the most verbose devug info
  #
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG "${CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG} -O0 -ggdb3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb3")

  #
  # Certain HIP functions are marked as [[nodiscard]], yet vllm ignores the result which generates
  # a lot of warnings that always mask real issues. Suppressing until this is properly addressed.
  #
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()

#
# Define other extension targets
#

#
# cumem_allocator extension (CUDA only)
#

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(VLLM_CUMEM_EXT_SRC
    "csrc/cumem_allocator.cpp")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_CUMEM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  message(STATUS "Enabling cumem allocator extension.")
  # link against cuda driver library
  list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  define_gpu_extension_target(
    cumem_allocator
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)
endif()

#
# Backend-specific extensions
# Include the appropriate backend file based on VLLM_GPU_LANG
#

if(VLLM_GPU_LANG STREQUAL "CUDA")
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/cuda_extension.cmake)
elseif(VLLM_GPU_LANG STREQUAL "HIP")
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/rocm_extension.cmake)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
  include(cmake/external_projects/flashmla.cmake)
  include(cmake/external_projects/qutlass.cmake)

  # vllm-flash-attn should be last as it overwrites some CMake functions
  include(cmake/external_projects/vllm_flash_attn.cmake)
endif()
