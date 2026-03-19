cmake_minimum_required(VERSION 3.18)
project(CGALParamNative VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use vcpkg CGAL (with GMP/MPFR/EPICK)
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# tinygltf (header-only)
include(FetchContent)
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

# Library — reuse source from cgal_param but with native EPICK kernel
add_library(cgalparam_native
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src/gltf_io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src/cgal_parameterize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src/seam_cut.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src/distortion.cpp
)

target_include_directories(cgalparam_native PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/include
    ${tinygltf_SOURCE_DIR}
)

# CGALPARAM_NATIVE flag → use EPICK kernel
target_compile_definitions(cgalparam_native PUBLIC CGALPARAM_NATIVE=1)

target_link_libraries(cgalparam_native PUBLIC
    Eigen3::Eigen
    CGAL::CGAL
)

if(MSVC)
    target_compile_definitions(cgalparam_native PUBLIC _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(cgalparam_native PUBLIC /EHsc /bigobj)
endif()

# CLI
add_executable(cgalparam_native_cli
    ${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src/main.cpp
)
target_link_libraries(cgalparam_native_cli PRIVATE cgalparam_native)

# CGAL reference test
add_executable(test_cgal_reference test_cgal_reference.cpp)
target_link_libraries(test_cgal_reference PRIVATE Eigen3::Eigen CGAL::CGAL)
if(MSVC)
    target_compile_definitions(test_cgal_reference PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(test_cgal_reference PRIVATE /EHsc /bigobj)
endif()
