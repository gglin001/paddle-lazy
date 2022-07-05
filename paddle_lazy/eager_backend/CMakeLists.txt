execute_process(
  COMMAND ${Python3_EXECUTABLE} codegen/codegen.py
  COMMAND echo "finish codegen"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_library(eager_backend SHARED eager_ops.cc op_runner.cc autogen_ops.cc)
