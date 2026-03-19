@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
echo === CMAKE CONFIGURE ===
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DMESHPARAM_BUILD_TESTS=ON -DMESHPARAM_BUILD_CLI=ON 2>&1
echo === CMAKE BUILD ===
cmake --build build 2>&1
echo === DONE ===
