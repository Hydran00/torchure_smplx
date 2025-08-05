cmake_minimum_required(VERSION 3.18)
project(chamferdist LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# Detect Torch
find_package(Torch REQUIRED)
find_package(CUDA QUIET)

if(CUDA_FOUND)
    message(STATUS "CUDA found — enabling CUDA compilation for Chamferdist")
    enable_language(CUDA)
    set(USE_CUDA TRUE)
else()
    message(STATUS "CUDA not found — compiling CPU-only version for Chamferdist")
    set(USE_CUDA FALSE)
endif()

# Collect source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/knn_cpu.cpp
)

if(USE_CUDA)
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/knn.cu)
endif()

# Create the extension
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Set C++ standard
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Compile definitions
if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_CUDA)
endif()

# Compile options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 --use_fast_math>
)

# Link against Torch
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Position-independent code for shared library
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
