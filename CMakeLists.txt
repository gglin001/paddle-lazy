cmake_minimum_required(VERSION 3.10)

project(paddle_lazy)

# paddle use c++14
set(CMAKE_CXX_STANDARD 14)

# find packages
find_package(Python3 COMPONENTS Interpreter Development)
# https://pybind11.readthedocs.io/en/stable/cmake/index.html
find_package(pybind11 CONFIG)
message("Found Python3: ${Python3_EXECUTABLE}")
message("Python3_SITEARCH: ${Python3_SITEARCH}")

set(PADDLE_INC_DIR "${Python3_SITEARCH}/paddle/include/")
set(PADDLE_LIB_DIR "${Python3_SITEARCH}/paddle/fluid/")
set(GLOG_INC_DIR "${CMAKE_SOURCE_DIR}/3rd")

# do not need link with glog/gflags, just need header files(with same version)
include_directories(${GLOG_INC_DIR})
include_directories(${PADDLE_INC_DIR})
include_directories(${CMAKE_SOURCE_DIR})

link_directories(${PADDLE_LIB_DIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
add_definitions(-DPADDLE_WITH_IPU)
# add_definitions(-DPADDLE_WITH_MKLDNN)

add_subdirectory(paddle_lazy)
