cmake_minimum_required(VERSION 3.10)
project(torchure_smplx)

# Torch
set(Torch_DIR "/home/hydran00/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Zlib (used by cnpy)
find_package(ZLIB REQUIRED)

# ---- CNPY Library ----
add_library(cnpy STATIC src/cnpy/cnpy.cpp)
target_include_directories(cnpy PUBLIC include/cnpy)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# ---- SMPLX Library (Shared or Static) ----
set(SMPLX_SOURCES
    src/model/smplx.cpp
    src/model/joint_names.cpp
    src/model/lbs.cpp
    src/model/vertex_ids.cpp
    src/model/vertex_joint_selector.cpp
)

add_library(smplx SHARED ${SMPLX_SOURCES})
add_library(smplx_static STATIC ${SMPLX_SOURCES}) # Optional: Keep if you need both

target_include_directories(smplx PUBLIC include/model)
target_link_libraries(smplx PUBLIC Torch::Torch ZLIB::ZLIB cnpy)
target_compile_features(smplx PUBLIC cxx_std_17)

target_include_directories(smplx_static PUBLIC include/model)
target_link_libraries(smplx_static PUBLIC Torch::Torch ZLIB::ZLIB cnpy)
target_compile_features(smplx_static PUBLIC cxx_std_17)

# ---- Benchmark Executable ----
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE smplx)

# ---- Test Executable ----
add_executable(test_cnpy_smpl tests/cnpy/test_cnpy_smpl.cpp)
target_link_libraries(test_cnpy_smpl PRIVATE cnpy Torch::Torch)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET example-app
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:example-app>)
#endif (MSVC)


# Test section
# add_executable(test_cnpy_smpl tests/cnpy/test_cnpy_smpl.cpp)
# target_include_directories(test_cnpy_smpl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ./include/cnpy)
# target_link_libraries(test_cnpy_smpl "${TORCH_LIBRARIES}" cnpy)



# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET example-app
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:example-app>)
#endif (MSVC)