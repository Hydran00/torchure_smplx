cmake_minimum_required(VERSION 3.10)
project(torchure_smplx VERSION 1.0.0)

set(BUILD_TESTS ON CACHE BOOL "Compile tests")
set(USE_OPEN3D ON BOOL "Build with Open3D support")
message(STATUS "Open3D support: ${USE_OPEN3D}")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(Torch REQUIRED)
message(STATUS "Found Torch in path: ${Torch_DIR}")
find_package(ZLIB REQUIRED)

# ---- SMPLX Library ----
set(SMPLX_SOURCES
    src/smplx/smplx.cpp
    src/smplx/joint_names.cpp
    src/smplx/lbs.cpp
    src/smplx/vertex_ids.cpp
    src/smplx/vertex_joint_selector.cpp
    thirdparty/cnpy/cnpy.cpp
)

add_library(smplx STATIC ${SMPLX_SOURCES})
add_library(torchure_smplx::smplx ALIAS smplx)

# Add to export set
set_target_properties(smplx PROPERTIES EXPORT_NAME smplx)

# ---- Chamferdist Library ----
add_subdirectory(thirdparty/chamferdist)
# This command registers the target for export (needed even if no install)
export(TARGETS smplx chamferdist
    FILE "${CMAKE_CURRENT_BINARY_DIR}/torchure_smplxTargets.cmake"
    NAMESPACE torchure_smplx::)

target_include_directories(smplx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/smplx
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnpy
)

target_link_libraries(smplx PUBLIC ${TORCH_LIBRARIES} ${ZLIB_LIBRARIES} chamferdist)


# ---- Examples ----
# Benchmark
add_executable(benchmark samples/benchmark.cpp)
target_link_libraries(benchmark PRIVATE smplx)
# Fitting
add_executable(fitting samples/fitting.cpp)
target_link_libraries(fitting PRIVATE smplx chamferdist)
if(USE_OPEN3D)
    find_package(Open3D QUIET)
    if(NOT Open3D_FOUND)
        message(WARNING "Open3D not found. Please install Open3D or disable USE_OPEN3D.")
        set(USE_OPEN3D OFF)
    else()
        message(STATUS "Found Open3D: ${Open3D_VERSION}")
        message(STATUS "Open3D dir: ${Open3D_DIR}")
        target_compile_definitions(fitting PRIVATE USE_OPEN3D)
        target_link_libraries(fitting PRIVATE Open3D::Open3D)
    endif()
endif()
# Consistency check
add_executable(consistency_check samples/consistency_check/consistency_check.cpp)
target_link_libraries(consistency_check PRIVATE smplx)

# ---- Tests ----
if(BUILD_TESTS)
    add_executable(test_cnpy_smplx tests/cnpy/test_cnpy_smplx.cpp)
    target_link_libraries(test_cnpy_smplx PRIVATE smplx)
    add_executable(test_chamferdist tests/chamferdist/test_chamfer.cpp)
    target_link_libraries(test_chamferdist PRIVATE chamferdist)
endif()


include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/torchure_smplxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/torchure_smplxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/torchure_smplxConfig.cmake"
    INSTALL_DESTINATION unused # No install needed
)
