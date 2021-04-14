mkdir build 
cd build
cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
    -DCMAKE_PREFIX_PATH=%CONDA_PREFIX%/Library ^
    -DCMAKE_TOOLCHAIN_FILE=%CONDA_PREFIX%/h2o.cmake ^
    ../

