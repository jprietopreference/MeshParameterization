cmake_minimum_required(VERSION 3.18)
project(OccTessellate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCascade via vcpkg
find_package(OpenCASCADE REQUIRED)

# tinygltf (header-only)
include(FetchContent)
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tinygltf)

add_executable(occ_tessellate_cli
    src/main.cpp
)

target_include_directories(occ_tessellate_cli PRIVATE
    ${tinygltf_SOURCE_DIR}
    ${OpenCASCADE_INCLUDE_DIR}
)

# Link all needed OCC libraries via their full paths
target_link_directories(occ_tessellate_cli PRIVATE ${OpenCASCADE_LIBRARY_DIR})

target_link_libraries(occ_tessellate_cli PRIVATE
    TKernel TKMath TKBRep TKTopAlgo TKMesh TKG3d TKG2d TKGeomBase TKGeomAlgo
    TKDESTEP TKDEIGES TKDE TKXSBase TKLCAF TKCAF TKCDF
    TKBO TKShHealing TKPrim TKXCAF TKVCAF
)

if(MSVC)
    target_compile_definitions(occ_tessellate_cli PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()
