cmake_minimum_required(VERSION 3.10)
project(torchure_smplx)

# Torch
set(Torch_DIR "/home/hydran00/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
message(STATUS "Torch_DIR is set to: ${Torch_DIR}")

find_package(Torch REQUIRED)

add_subdirectory(thirdparty/cnpy)

# ---- SMPLX Library (Shared or Static) ----
set(SMPLX_SOURCES
    src/model/smplx.cpp
    src/model/joint_names.cpp
    src/model/lbs.cpp
    src/model/vertex_ids.cpp
    src/model/vertex_joint_selector.cpp
)

add_library(smplx STATIC ${SMPLX_SOURCES})
target_include_directories(smplx PUBLIC include/model)
target_link_libraries(smplx PUBLIC cnpy ${TORCH_LIBRARIES})
target_compile_features(smplx PUBLIC cxx_std_17)

# ---- Benchmark Executable ----
# add_executable(benchmark benchmark.cpp)
# target_link_libraries(benchmark PRIVATE smplx cnpy)

# ---- Test Executable ----
add_executable(test_cnpy_smpl tests/cnpy/test_cnpy_smpl.cpp)
target_include_directories(test_cnpy_smpl PUBLIC include/model)
target_link_directories(test_cnpy_smpl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnpy/include)
target_link_libraries(test_cnpy_smpl PRIVATE cnpy ${TORCH_LIBRARIES})


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