# execute_process(
#   COMMAND ${Python3_EXECUTABLE} codegen/autogen_eager_ops.py
#   COMMAND echo "finish autogen_ops"
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# execute_process(
#   COMMAND ${Python3_EXECUTABLE} codegen/autogen_ops_map.py
#   COMMAND echo "finish autogen_ops_map"
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_library(eager_backend SHARED op_runner.cc eager_ops.cc eager_ops_autogen.cc)
