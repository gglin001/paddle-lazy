# TODO(alleng) remove flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")

add_subdirectory(eager_backend)

execute_process(
  COMMAND ${Python3_EXECUTABLE} codegen/autogen_lazy_nodes.py
  COMMAND echo "finish autogen_lazy_node"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

execute_process(
  COMMAND ${Python3_EXECUTABLE} codegen/autogen_kernels.py
  COMMAND echo "finish autogen_kernel"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

file(GLOB KERNELS kernels/*.cc)
message("KERNELS: ${KERNELS}")
set(LAZY_SRCS lazy_allocator.cc lazy_tensor.cc lazy_backend.cc
              lazy_nodes.cc)

add_library(lazy_backend SHARED ${LAZY_SRCS} ${KERNELS})
target_link_libraries(lazy_backend PRIVATE eager_backend)
target_link_libraries(lazy_backend PRIVATE paddle_import)
target_link_libraries(lazy_backend PRIVATE :core_avx.so) # special name

pybind11_add_module(lazy_lib lib.cc)
target_link_libraries(lazy_lib PRIVATE lazy_backend)

add_executable(test_phi_api test_phi_api.cc)
target_link_libraries(test_phi_api PRIVATE :core_avx.so) # special name
target_link_libraries(test_phi_api PRIVATE ${Python3_LIBRARIES})
target_link_libraries(test_phi_api PRIVATE lazy_backend)
