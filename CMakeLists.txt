cmake_minimum_required(VERSION 3.10)
project(torchure_smplx)

set(COMPILE_TESTS ON CACHE BOOL "Compile tests")
# Torch
set(Torch_DIR "/home/hydran00/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
message(STATUS "Torch_DIR is set to: ${Torch_DIR}")

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
# add_subdirectory(thirdparty/cnpy)

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
target_compile_features(smplx PUBLIC cxx_std_17)




# ---- Benchmark Executable ----
add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PUBLIC include/smplx)
target_link_libraries(benchmark PRIVATE smplx)

# ---- Test Executable ----
if(${COMPILE_TESTS})
    add_executable(test_cnpy_smplx tests/cnpy/test_cnpy_smplx.cpp)
    target_include_directories(test_cnpy_smplx PUBLIC include/smplx)
    target_link_libraries(test_cnpy_smplx PRIVATE smplx)
endif()
