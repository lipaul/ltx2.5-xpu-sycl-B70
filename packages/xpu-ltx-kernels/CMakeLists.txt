cmake_minimum_required(VERSION 3.22)
project(xpu_ltx_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resolve the active venv python (matches the uv environment's torch). Pass
# -DPython3_EXECUTABLE=<venv>/bin/python if find_package picks the wrong one.
if(DEFINED ENV{VIRTUAL_ENV})
  set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()
if(NOT Python3_EXECUTABLE AND DEFINED ENV{UV_PROJECT_ENVIRONMENT})
  set(Python3_EXECUTABLE "$ENV{UV_PROJECT_ENVIRONMENT}/bin/python")
endif()

# Locate the torch that the active uv venv is using. Resolve through Python so
# the build always matches the runtime torch (XPU pin, CXX11 ABI, etc).
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import os,torch;print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import torch;print('1' if torch.compiled_with_cxx11_abi() else '0')"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# The venv root = the resolved interpreter's sys.prefix (venv may not be active).
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys;print(sys.prefix)"
  OUTPUT_VARIABLE PYTHON_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(TORCH_INCLUDE_DIR "${TORCH_ROOT}/include")
set(TORCH_LIB_DIR "${TORCH_ROOT}/lib")

# SYCL/ESIMD toolchain MUST match torch-xpu's SYCL runtime. torch 2.12.0+xpu
# links libsycl.so.8 (oneAPI 2025.3); the newer standalone compilers (2026.x)
# ship libsycl.so.9 whose device-image format is incompatible (crashes in
# ProgramManager::addImage at dlopen). Use the 2025.3 icpx + its headers + its
# libsycl.so.8, all self-consistent. Installable via:
#   sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2025.3
if(EXISTS "/opt/intel/oneapi/compiler/2025.3/bin/icpx")
  set(_SYCL_COMPILER "/opt/intel/oneapi/compiler/2025.3/bin/icpx")
  set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/2025.3/include")
  set(SYCL_LIB_DIR "/opt/intel/oneapi/compiler/2025.3/lib")
else()
  set(_SYCL_COMPILER "icpx")
  set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/latest/include")
  set(SYCL_LIB_DIR "/opt/intel/oneapi/compiler/latest/lib")
endif()
message(STATUS "SYCL compiler: ${_SYCL_COMPILER}")
message(STATUS "SYCL include:  ${SYCL_INCLUDE_DIR}")
message(STATUS "SYCL lib:      ${SYCL_LIB_DIR}")
set(CMAKE_CXX_COMPILER "${_SYCL_COMPILER}")
set(SYCL_LINK_FLAGS "-fsycl")
set(SYCL_COMPILE_FLAGS
  "-fsycl"
  "-fsycl-targets=spir64"
  "-fsycl-esimd-force-stateless-mem"
  "-fno-sycl-device-code-split-esimd"
  "-O3")

# icpx compiles SYCL/ESIMD in .cpp files via -fsycl (CMake has no .sycl
# extension support and would silently drop those sources). All device code
# lives in .cpp sources compiled with SYCL_COMPILE_FLAGS.
file(GLOB KERNEL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp")
message(STATUS "KERNEL_SOURCES: ${KERNEL_SOURCES}")

add_library(xpu_ltx_kernels SHARED ${KERNEL_SOURCES})
target_include_directories(xpu_ltx_kernels PRIVATE
  "${TORCH_INCLUDE_DIR}"
  "${TORCH_INCLUDE_DIR}/torch/csrc/api/include"
  "${SYCL_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_compile_options(xpu_ltx_kernels PRIVATE ${SYCL_COMPILE_FLAGS} -fPIC)
target_link_options(xpu_ltx_kernels PRIVATE
  ${SYCL_LINK_FLAGS}
  "-L${SYCL_LIB_DIR}"
  "-L${TORCH_LIB_DIR}"
  "-Wl,-rpath,${SYCL_LIB_DIR}"
  "-Wl,-rpath,${TORCH_LIB_DIR}")
target_link_libraries(xpu_ltx_kernels PRIVATE
  sycl
  torch
  torch_cpu
  torch_python
  c10
  torch_xpu
  c10_xpu
  dl)
set_target_properties(xpu_ltx_kernels PROPERTIES
  LINKER_LANGUAGE CXX
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/xpu_ltx_kernels")

# ---- helper target: build via uv -------------------------------------------
add_custom_target(build_pkg ALL
  COMMAND ${CMAKE_COMMAND} -E echo
    "xpu_ltx_kernels built. Install into the venv with:"
    "uv pip install -e packages/xpu-ltx-kernels --no-build-isolation"
  DEPENDS xpu_ltx_kernels)