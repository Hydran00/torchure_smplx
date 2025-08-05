cmake_minimum_required(VERSION 3.10)
project(torchure_smplx)

set(COMPILE_TESTS ON CACHE BOOL "Compile tests")
set(USE_OPEN3D OFF CACHE BOOL "Build with Open3D support")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(Open3D QUIET)

# Check if target exists
if(TARGET Open3D::Open3D)
    message(STATUS "Open3D::Open3D target exists")
    get_target_property(Open3D_TARGET_LOCATION Open3D::Open3D LOCATION)
    message(STATUS "Open3D target location: ${Open3D_TARGET_LOCATION}")
else()
    message(STATUS "Open3D::Open3D target does NOT exist")
endif()
message(STATUS "========================")

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# ---- SMPLX Library (Shared or Static) ----
set(SMPLX_SOURCES
    src/smplx/smplx.cpp
    src/smplx/joint_names.cpp
    src/smplx/lbs.cpp
    src/smplx/vertex_ids.cpp
    src/smplx/vertex_joint_selector.cpp
    thirdparty/cnpy/cnpy.cpp
)

add_library(smplx STATIC ${SMPLX_SOURCES})
target_include_directories(smplx PUBLIC include/smplx thirdparty/cnpy)
target_link_libraries(smplx PUBLIC ${TORCH_LIBRARIES} ${ZLIB_LIBRARIES})
# target_compile_features(smplx PUBLIC cxx_std_17)

# ---- Benchmark Executable ----
add_executable(benchmark samples/benchmark.cpp)
target_include_directories(benchmark PUBLIC include/smplx)
target_link_libraries(benchmark PRIVATE smplx)

# ---- Fitting Executable ----
add_subdirectory(thirdparty/chamferdist)
add_executable(fitting samples/fitting.cpp)
target_include_directories(fitting PUBLIC include/smplx thirdparty/chamferdist)

# Link everything EXCEPT Open3D first
target_link_libraries(fitting PRIVATE smplx chamferdist)

if(USE_OPEN3D)
    target_compile_definitions(fitting PRIVATE USE_OPEN3D)
    target_include_directories(fitting PRIVATE /home/hydran00/Open3D/include)
    target_link_libraries(fitting PRIVATE Open3D::Open3D)
endif()


# ---- Test Executable ----
if(${COMPILE_TESTS})
    add_executable(test_cnpy_smplx tests/cnpy/test_cnpy_smplx.cpp)
    target_include_directories(test_cnpy_smplx PUBLIC include/smplx)
    target_link_libraries(test_cnpy_smplx PRIVATE smplx)

    add_executable(test_chamferdist tests/chamferdist/test_chamfer.cpp)
    target_include_directories(test_chamferdist PUBLIC thirdparty/chamferdist)
    target_link_libraries(test_chamferdist PRIVATE chamferdist)
endif()
