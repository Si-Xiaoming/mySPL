cmake_minimum_required(VERSION 3.10)

project(MySPL LANGUAGES C CXX)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_BUGFIX "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BUGFIX})

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

###  CMAKE_CURRENT_BINARY_DIR  当前正在处理的二进制目录路径   对于此处是在build文件夹下创建
###  CMAKE_CURRENT_SOURCE_DIR  当前处理的CMakeLists.txt所在的目录  MySPL


if(MSVC)
    add_compile_definitions("/utf-8")    
endif(MSVC)


find_package(Eigen3 REQUIRED)

find_package(Laszip REQUIRED)



## eigen库比较特殊，在include中直接提供源码，所以在找到eigen后，直接include其路径即可
## 编译的ranger静态库，需要通过
## CMAKE_CURRENT_LIST_DIR  当前正在处理的列表文件的完整目录
include_directories(${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/depends)

#把当前的目录添加到搜索目录
#include_directories(${CMAKE_CURRENT_LIST_DIR})   



add_subdirectory(depends)
add_subdirectory(src)
