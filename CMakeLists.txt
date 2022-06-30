cmake_minimum_required(VERSION 3.10)

project(paddle_lazy)

# paddle use c++14
set(CMAKE_CXX_STANDARD 14)

# find packages
# find_package(glog REQUIRED)
# find_package(gflags REQUIRED)
# find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
message("Found Python3: ${Python3_EXECUTABLE}")
include_directories(${PYTHON_INCLUDE_DIRS})

# TODO(alleng) remove hardcode
set(PADDLE_INC_DIR "/opt/conda/lib/python3.7/site-packages/paddle/include/")
set(PADDLE_LIB_DIR "/opt/conda/lib/python3.7/site-packages/paddle/fluid/")
include_directories(${PADDLE_INC_DIR})
# do not need link with glog/gflags, just need header files(with same version)
# may conflict with other libs
include_directories("/opt/conda/include")
link_directories(${PADDLE_LIB_DIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
# add_definitions(-DPADDLE_WITH_MKLDNN)

add_subdirectory(paddle_lazy)
