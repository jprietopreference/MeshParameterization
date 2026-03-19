cmake_minimum_required(VERSION 3.18)
project(CGALParameterization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(CGALPARAM_BUILD_WASM "Build for WebAssembly (Emscripten)" OFF)
option(CGALPARAM_BUILD_CLI "Build command-line tool" ON)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
include(FetchContent)

# Eigen (header-only)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# Boost (headers only — CGAL just needs header includes)
FetchContent_Declare(
    boost
    URL https://archives.boost.io/release/1.78.0/source/boost_1_78_0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Only populate, don't process CMakeLists (avoids building 400+ compiled Boost targets)
FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
endif()

# CGAL (header-only since v5.0)
FetchContent_Declare(
    cgal
    GIT_REPOSITORY https://github.com/CGAL/cgal.git
    GIT_TAG        v5.5.5
    GIT_SHALLOW    TRUE
)
# Disable CGAL's optional features we don't need
set(CGAL_DISABLE_GMP ON CACHE BOOL "" FORCE)
set(WITH_GMP OFF CACHE BOOL "" FORCE)
set(WITH_MPFR OFF CACHE BOOL "" FORCE)
set(WITH_CGAL_Core OFF CACHE BOOL "" FORCE)
set(WITH_CGAL_ImageIO OFF CACHE BOOL "" FORCE)
set(WITH_CGAL_Qt6 OFF CACHE BOOL "" FORCE)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE ON CACHE BOOL "" FORCE)
# Only fetch CGAL headers, don't process its full CMakeLists
FetchContent_GetProperties(cgal)
if(NOT cgal_POPULATED)
    FetchContent_Populate(cgal)
endif()

# Create a minimal CGAL interface target (header-only)
# Glob all CGAL package include directories
add_library(CGAL_headers INTERFACE)
file(GLOB cgal_pkg_dirs "${cgal_SOURCE_DIR}/*/include")
target_include_directories(CGAL_headers INTERFACE ${cgal_pkg_dirs})
target_compile_definitions(CGAL_headers INTERFACE CGAL_DISABLE_GMP=1)

# Add Boost include path (traditional layout for Boost 1.78)
target_include_directories(CGAL_headers INTERFACE ${boost_SOURCE_DIR})

add_library(CGAL::CGAL ALIAS CGAL_headers)

# tinygltf (header-only glTF loader)
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tinygltf)
if(NOT tinygltf_POPULATED)
    FetchContent_Populate(tinygltf)
endif()

# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
add_library(cgalparam
    src/gltf_io.cpp
    src/cgal_parameterize.cpp
    src/seam_cut.cpp
    src/distortion.cpp
)

target_include_directories(cgalparam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${tinygltf_SOURCE_DIR}
)

target_link_libraries(cgalparam PUBLIC
    Eigen3::Eigen
    CGAL::CGAL
)

if(MSVC)
    target_compile_definitions(cgalparam PUBLIC _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(cgalparam PUBLIC /EHsc /bigobj)
endif()

# --------------------------------------------------------------------------
# CLI tool
# --------------------------------------------------------------------------
if(CGALPARAM_BUILD_CLI AND NOT CGALPARAM_BUILD_WASM)
    add_executable(cgalparam_cli src/main.cpp)
    target_link_libraries(cgalparam_cli PRIVATE cgalparam)
endif()

# --------------------------------------------------------------------------
# WASM build
# --------------------------------------------------------------------------
if(CGALPARAM_BUILD_WASM)
    target_compile_definitions(cgalparam PUBLIC CGALPARAM_WASM=1)

    add_executable(cgalparam_wasm wasm/bindings.cpp)
    target_link_libraries(cgalparam_wasm PRIVATE cgalparam)

    set_target_properties(cgalparam_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -s WASM=1 \
            -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS'] \
            -s ALLOW_MEMORY_GROWTH=1 \
            -s MODULARIZE=1 \
            -s EXPORT_NAME='CGALParam' \
            -s ENVIRONMENT='web,worker' \
            --bind"
    )
endif()
