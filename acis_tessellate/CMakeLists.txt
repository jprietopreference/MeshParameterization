cmake_minimum_required(VERSION 3.18)
project(AcisTessellate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ACIS and Interop paths
set(ACIS_ROOT "D:/dev/Spatial/ACIS_2026.1.0.0" CACHE PATH "ACIS installation")
set(INTEROP_ROOT "D:/dev/Spatial/Interop_2026.1.0.0" CACHE PATH "Interop installation")

# tinygltf (header-only)
include(FetchContent)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tinygltf)

add_executable(acis_tessellate_cli
    src/main.cpp
    src/unlock_license.cpp
)

target_include_directories(acis_tessellate_cli PRIVATE
    ${tinygltf_SOURCE_DIR}
    ${ACIS_ROOT}/include
    ${INTEROP_ROOT}/include
)

target_link_directories(acis_tessellate_cli PRIVATE
    ${ACIS_ROOT}/win_b64/code/lib
    ${INTEROP_ROOT}/win_b64/code/lib
)

# Core ACIS libs
target_link_libraries(acis_tessellate_cli PRIVATE
    # Interop
    SPAIop SPAIInterop SPAIopAcis SPAIopUtilities
    # ACIS core
    kern_scm ga_scm ds_scm
    # Faceting
    fct_scm
    # Compound (needed by facet_api)
    covr_scm
    # Geometry
    ga_husk_scm
    # I/O
    acisio_scm
)

if(MSVC)
    target_compile_definitions(acis_tessellate_cli PRIVATE
        _USE_MATH_DEFINES NOMINMAX _WINDOWS_SOURCE NT
    )
    target_compile_options(acis_tessellate_cli PRIVATE /EHsc /bigobj)
endif()

# Copy ACIS DLLs to output directory
file(GLOB ACIS_DLLS "${ACIS_ROOT}/win_b64/code/bin/*.dll")
file(GLOB INTEROP_DLLS "${INTEROP_ROOT}/win_b64/code/bin/*.dll")
foreach(dll ${ACIS_DLLS} ${INTEROP_DLLS})
    add_custom_command(TARGET acis_tessellate_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:acis_tessellate_cli>"
    )
endforeach()
