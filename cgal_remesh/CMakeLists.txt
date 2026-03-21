cmake_minimum_required(VERSION 3.18)
project(CGALRemesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)

include(FetchContent)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tinygltf)

add_executable(cgal_remesh_cli src/main.cpp)

target_include_directories(cgal_remesh_cli PRIVATE ${tinygltf_SOURCE_DIR})
target_link_libraries(cgal_remesh_cli PRIVATE CGAL::CGAL)

if(MSVC)
    target_compile_definitions(cgal_remesh_cli PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(cgal_remesh_cli PRIVATE /EHsc)
endif()
