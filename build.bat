@echo off
REM Build script for MeshParameterization
REM Requires: Visual Studio 2022 (any edition), Ninja, CMake
REM Optional: vcpkg (for MKL/CGAL native)

REM Auto-detect VS installation via vswhere
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set VS_PATH=%%i

if not defined VS_PATH (
    echo ERROR: Visual Studio 2022 not found. Install VS2022 with C++ workload.
    exit /b 1
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

echo === CMAKE CONFIGURE ===
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DMESHPARAM_BUILD_TESTS=ON -DMESHPARAM_BUILD_CLI=ON 2>&1
echo === CMAKE BUILD ===
cmake --build build 2>&1
echo === DONE ===
