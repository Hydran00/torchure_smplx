cmake_minimum_required(VERSION 3.10)
project(cnpy)

cmake_minimum_required(VERSION 3.10)
project(cnpy_static_lib)

# Optionally allow user to toggle position independent code
option(CNPY_BUILD_POSITION_INDEPENDENT_CODE "Build position independent code" ON)

# Create the static library target
add_library(cnpy STATIC
    src/cnpy.cpp
)
target_include_directories(cnpy PUBLIC include)
# Add zlib dependency (required by cnpy)
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# Set position-independent code if enabled
if(CNPY_BUILD_POSITION_INDEPENDENT_CODE)
    set_target_properties(cnpy PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Include directories for consumers of this library
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
