cmake_minimum_required(VERSION 3.10)
project(minimal_cmake_example)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
# Set the path to your torchure_smplx library
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build)
set(USE_OPEN3D OFF BOOL "Build with Open3D support")
find_package(torchure_smplx REQUIRED)
find_package(Torch REQUIRED)
find_package(Open3D QUIET)

# Create your executable
add_executable(minimal_cmake_example main.cpp)

# Link against the smplx library (which import Torch) and optionally to the Open3D library
if(OPEN3D_FOUND)
    target_compile_definitions(minimal_cmake_example PRIVATE USE_OPEN3D=ON)
    target_link_libraries(minimal_cmake_example PRIVATE Open3D::Open3D)
endif()
target_link_libraries(minimal_cmake_example PRIVATE torchure_smplx::smplx)
