cmake_minimum_required(VERSION 3.18)
project(MeshParamServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for intra-request parallelism
find_package(OpenMP)

# Draco for mesh compression (optional, from vcpkg)
find_package(draco CONFIG QUIET)

# Dependencies via FetchContent (reuse from parent)
include(FetchContent)

FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(libigl)

FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tinygltf)
if(NOT tinygltf_POPULATED)
    FetchContent_Populate(tinygltf)
endif()

FetchContent_Declare(spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        v1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
    FetchContent_Populate(spectra)
endif()

# Boost (headers for CGAL — populate only)
FetchContent_Declare(boost
    URL https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    FetchContent_Populate(boost)
endif()

# CGAL (header-only — populate only, don't process CMakeLists to avoid target conflicts)
FetchContent_Declare(cgal
    GIT_REPOSITORY https://github.com/CGAL/cgal.git
    GIT_TAG        v5.5.5
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(cgal)
if(NOT cgal_POPULATED)
    FetchContent_Populate(cgal)
endif()

# Build unified CGAL include
set(CGAL_UNIFIED "${cgal_SOURCE_DIR}/unified_include")
if(NOT EXISTS "${CGAL_UNIFIED}/CGAL/basic.h")
    message(STATUS "Creating unified CGAL include directory...")
    file(GLOB cgal_pkg_dirs "${cgal_SOURCE_DIR}/*/include")
    foreach(pkg_inc ${cgal_pkg_dirs})
        file(GLOB_RECURSE pkg_headers RELATIVE "${pkg_inc}" "${pkg_inc}/*")
        foreach(hdr ${pkg_headers})
            if(NOT EXISTS "${CGAL_UNIFIED}/${hdr}")
                configure_file("${pkg_inc}/${hdr}" "${CGAL_UNIFIED}/${hdr}" COPYONLY)
            endif()
        endforeach()
    endforeach()
endif()

# --------------------------------------------------------------------------
# meshparam library
# --------------------------------------------------------------------------
set(MESHPARAM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(MESHPARAM_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_library(meshparam_lib
    ${MESHPARAM_SRC_DIR}/mesh.cpp
    ${MESHPARAM_SRC_DIR}/gltf_io.cpp
    ${MESHPARAM_SRC_DIR}/laplacian.cpp
    ${MESHPARAM_SRC_DIR}/heat_geodesics.cpp
    ${MESHPARAM_SRC_DIR}/mds.cpp
    ${MESHPARAM_SRC_DIR}/poisson_fill.cpp
    ${MESHPARAM_SRC_DIR}/parameterizer.cpp
    ${MESHPARAM_SRC_DIR}/distortion.cpp
)

target_include_directories(meshparam_lib PUBLIC
    ${MESHPARAM_INC_DIR}
    ${tinygltf_SOURCE_DIR}
    ${spectra_SOURCE_DIR}/include
)

target_link_libraries(meshparam_lib PUBLIC Eigen3::Eigen igl::core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(meshparam_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(meshparam_lib PUBLIC MESHPARAM_OPENMP=1)
endif()

if(MSVC)
    target_compile_definitions(meshparam_lib PUBLIC _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(meshparam_lib PUBLIC /EHsc)
endif()

# --------------------------------------------------------------------------
# cgalparam library
# --------------------------------------------------------------------------
set(CGAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/src")
set(CGAL_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cgal_param/include")

add_library(cgalparam_lib
    ${CGAL_SRC_DIR}/gltf_io.cpp
    ${CGAL_SRC_DIR}/cgal_parameterize.cpp
    ${CGAL_SRC_DIR}/seam_cut.cpp
    ${CGAL_SRC_DIR}/distortion.cpp
)

target_include_directories(cgalparam_lib PUBLIC
    ${CGAL_INC_DIR}
    ${tinygltf_SOURCE_DIR}
    ${CGAL_UNIFIED}
    ${boost_SOURCE_DIR}
)

target_link_libraries(cgalparam_lib PUBLIC Eigen3::Eigen)
target_compile_definitions(cgalparam_lib PUBLIC CGAL_DISABLE_GMP=1 CGALPARAM_NO_TINYGLTF_IMPL=1)

if(MSVC)
    target_compile_definitions(cgalparam_lib PUBLIC _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(cgalparam_lib PUBLIC /EHsc /bigobj)
endif()

# --------------------------------------------------------------------------
# Server executable
# --------------------------------------------------------------------------
add_executable(meshparam_server main.cpp)

target_include_directories(meshparam_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # httplib.h
)

target_link_libraries(meshparam_server PRIVATE meshparam_lib cgalparam_lib)
if(WIN32)
    target_link_libraries(meshparam_server PRIVATE ws2_32)  # httplib needs Winsock on MinGW
endif()

# Draco disabled — BabylonJS doesn't expose custom GENERIC attributes from Draco
# if(draco_FOUND)
#     target_link_libraries(meshparam_server PRIVATE draco::draco)
#     target_compile_definitions(meshparam_server PRIVATE HAS_DRACO=1)
# endif()

if(MSVC)
    target_compile_definitions(meshparam_server PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

# --------------------------------------------------------------------------
# Benchmark CLI (process-isolated per-method parameterization)
# --------------------------------------------------------------------------
add_executable(meshparam_bench bench_cli.cpp)

target_include_directories(meshparam_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # methods.inc
)

target_link_libraries(meshparam_bench PRIVATE meshparam_lib cgalparam_lib)

if(MSVC)
    target_compile_definitions(meshparam_bench PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

# --------------------------------------------------------------------------
# Composite Majorization library (Shtengel et al. 2017)
# --------------------------------------------------------------------------
set(CM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/CompMajor/src")

# Find MKL (required for Pardiso solver in Composite Majorization)
# Searched in: CMAKE_PREFIX_PATH, vcpkg installed dir, system paths
find_package(MKL CONFIG QUIET)
if(NOT MKL_FOUND)
    # Build hint list from vcpkg and environment
    set(_MKL_HINTS "")
    if(DEFINED VCPKG_INSTALLED_DIR)
        list(APPEND _MKL_HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
    endif()
    if(DEFINED ENV{VCPKG_ROOT})
        list(APPEND _MKL_HINTS "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
    endif()
    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        get_filename_component(_vcpkg_root "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
        get_filename_component(_vcpkg_root "${_vcpkg_root}/../.." ABSOLUTE)
        list(APPEND _MKL_HINTS "${_vcpkg_root}/installed/${VCPKG_TARGET_TRIPLET}")
    endif()

    find_path(MKL_INCLUDE_DIR mkl.h HINTS ${_MKL_HINTS} PATH_SUFFIXES include)
    find_library(MKL_CORE_LIB mkl_core HINTS ${_MKL_HINTS} PATH_SUFFIXES lib)
    find_library(MKL_INTEL_LIB mkl_intel_lp64 HINTS ${_MKL_HINTS} PATH_SUFFIXES lib)
    find_library(MKL_THREAD_LIB mkl_sequential HINTS ${_MKL_HINTS} PATH_SUFFIXES lib)
    if(MKL_INCLUDE_DIR AND MKL_CORE_LIB)
        set(MKL_FOUND TRUE)
        set(MKL_LIBRARIES ${MKL_INTEL_LIB} ${MKL_CORE_LIB} ${MKL_THREAD_LIB})
        set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
        message(STATUS "MKL found: ${MKL_INCLUDE_DIR}")
    else()
        message(STATUS "MKL not found — Composite Majorization (CM) will be disabled")
    endif()
endif()

if(MKL_FOUND)
    add_library(compmajor_lib
        ${CM_SRC_DIR}/Energy.cpp
        ${CM_SRC_DIR}/EnergySymDir.cpp
        ${CM_SRC_DIR}/Newton.cpp
        ${CM_SRC_DIR}/Solver.cpp
        ${CM_SRC_DIR}/PardisoSolver.cpp
        ${CM_SRC_DIR}/Utils.cpp
    )

    target_include_directories(compmajor_lib PUBLIC
        ${CM_SRC_DIR}
        ${MKL_INCLUDE_DIRS}
    )

    if(TARGET MKL::MKL)
        target_link_libraries(compmajor_lib PUBLIC Eigen3::Eigen igl::core MKL::MKL)
    else()
        target_link_libraries(compmajor_lib PUBLIC Eigen3::Eigen igl::core ${MKL_LIBRARIES})
    endif()

    if(MSVC)
        target_compile_definitions(compmajor_lib PUBLIC _USE_MATH_DEFINES NOMINMAX)
        target_compile_options(compmajor_lib PUBLIC /EHsc)
    endif()

    # Link CM to bench CLI
    target_link_libraries(meshparam_bench PRIVATE compmajor_lib)
    target_compile_definitions(meshparam_bench PRIVATE HAS_COMPMAJOR=1)

    # Link CM to server
    target_link_libraries(meshparam_server PRIVATE compmajor_lib)
    target_compile_definitions(meshparam_server PRIVATE HAS_COMPMAJOR=1)

    message(STATUS "CompMajor (Composite Majorization) enabled with MKL Pardiso")

    # Auto-copy MKL DLLs to build dir after build
    if(WIN32 AND MKL_CORE_LIB)
        get_filename_component(_mkl_lib_dir "${MKL_CORE_LIB}" DIRECTORY)
        get_filename_component(_mkl_bin_dir "${_mkl_lib_dir}/../bin" ABSOLUTE)
        if(EXISTS "${_mkl_bin_dir}")
            add_custom_command(TARGET meshparam_bench POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_mkl_bin_dir}/mkl_core.2.dll"
                    "${_mkl_bin_dir}/mkl_intel_thread.2.dll"
                    "${_mkl_bin_dir}/mkl_def.2.dll"
                    "${_mkl_bin_dir}/mkl_avx2.2.dll"
                    "${_mkl_bin_dir}/mkl_sequential.2.dll"
                    $<TARGET_FILE_DIR:meshparam_bench>
                COMMENT "Copying MKL DLLs to build directory"
                COMMAND_ERROR_IS_FATAL ANY
            )
            message(STATUS "MKL DLLs will be auto-copied from ${_mkl_bin_dir}")
        endif()
    endif()
else()
    message(STATUS "MKL not found — CompMajor disabled")
endif()
