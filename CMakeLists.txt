cmake_minimum_required(VERSION 3.18)
project(MeshParameterization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(MESHPARAM_BUILD_WASM "Build for WebAssembly (Emscripten)" OFF)
option(MESHPARAM_BUILD_TESTS "Build unit tests" ON)
option(MESHPARAM_BUILD_CLI "Build command-line tool" ON)

# --------------------------------------------------------------------------
# Dependencies via FetchContent
# --------------------------------------------------------------------------
include(FetchContent)

# Eigen (header-only linear algebra)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# libigl (header-only geometry processing)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(libigl)

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

# Spectra (header-only eigenvalue solver, for MDS)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        v1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
    FetchContent_Populate(spectra)
endif()

# --------------------------------------------------------------------------
# Library target
# --------------------------------------------------------------------------
add_library(meshparam
    src/mesh.cpp
    src/gltf_io.cpp
    src/laplacian.cpp
    src/heat_geodesics.cpp
    src/mds.cpp
    src/poisson_fill.cpp
    src/parameterizer.cpp
    src/distortion.cpp
)

target_include_directories(meshparam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${tinygltf_SOURCE_DIR}
    ${spectra_SOURCE_DIR}/include
)

target_link_libraries(meshparam PUBLIC
    Eigen3::Eigen
    igl::core
)

# MSVC: enable M_PI and friends, disable warnings about non-standard extensions
if(MSVC)
    target_compile_definitions(meshparam PUBLIC _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(meshparam PUBLIC /EHsc)
endif()

# --------------------------------------------------------------------------
# CLI tool
# --------------------------------------------------------------------------
if(MESHPARAM_BUILD_CLI AND NOT MESHPARAM_BUILD_WASM)
    add_executable(meshparam_cli src/main.cpp)
    target_link_libraries(meshparam_cli PRIVATE meshparam)
endif()

# --------------------------------------------------------------------------
# WASM build
# --------------------------------------------------------------------------
if(MESHPARAM_BUILD_WASM)
    set(MESHPARAM_BUILD_TESTS OFF)
    set(MESHPARAM_BUILD_CLI OFF)

    add_executable(meshparam_wasm wasm/bindings.cpp)
    target_link_libraries(meshparam_wasm PRIVATE meshparam)

    set_target_properties(meshparam_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -s WASM=1 \
            -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS'] \
            -s ALLOW_MEMORY_GROWTH=1 \
            -s MODULARIZE=1 \
            -s EXPORT_NAME='MeshParam' \
            -s ENVIRONMENT='web,worker' \
            --bind"
    )
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(MESHPARAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
