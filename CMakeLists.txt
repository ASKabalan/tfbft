cmake_minimum_required(VERSION 3.19...3.25)

project(butterfly_fft LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# if (NOT SKBUILD)
#     message(FATAL_ERROR "SKBUILD is not set")
# endif()

find_package(CUDAToolkit REQUIRED VERSION 12)
set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

find_package(MPI REQUIRED)
find_package(NVHPC REQUIRED COMPONENTS MATH NCCL)

string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
# Set source files

# Add the executable
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
# Detect the installed nanobind package and import it into CMake
#execute_process(
#  COMMAND "${Python_EXECUTABLE}"
#          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
#  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)

set(XLA_DIR ${Python_SITELIB}/jaxlib/include)
message(STATUS "XLA_DIR: ${XLA_DIR}")


message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)
# Include directories for the executable
set(SOURCES src/csrc/extensions.cc src/csrc/butterfly_fft.cu src/csrc/bit_reverse.cu src/csrc/collectives.cc)
nanobind_add_module(butterfly_fft_lib SHARED ${SOURCES})
set_target_properties(butterfly_fft_lib PROPERTIES CUDA_ARCHITECTURES "70;80;89")
target_link_libraries(butterfly_fft_lib  PRIVATE NVHPC::CUDA MPI::MPI_CXX NVHPC::NCCL)
target_include_directories(butterfly_fft_lib PUBLIC 
                           src/csrc/include
                           ${XLA_DIR}
                           ${NVHPC_CUDA_INCLUDE_DIR}
                           ${NVHPC_MATH_INCLUDE_DIR}
                           ${MPI_CXX_INCLUDE_DIRS}
                           ${NVHPC_NCCL_INCLUDE_DIR}
                          )

install(TARGETS butterfly_fft_lib LIBRARY DESTINATION butterfly_fft)



