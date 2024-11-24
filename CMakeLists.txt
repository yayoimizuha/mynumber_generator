cmake_minimum_required(VERSION 3.31)
project(mynumber_generator)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl -Qstd=c++20 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl -v")
add_executable(mynumber_generator main.cpp)
target_link_directories(mynumber_generator PUBLIC "C:\\Program Files (x86)\\Common Files\\intel\\Shared Libraries\\lib")
