file(GLOB KERNELS kernels/*.cc)
message("KERNELS: ${KERNELS}")
set(LAZY_SRCS lib.cc lazy_allocator.cc)

pybind11_add_module(lazy_lib ${LAZY_SRCS} ${KERNELS})
target_link_libraries(lazy_lib PRIVATE :core_avx.so) # special name

add_executable(test_phi_api test_phi_api.cc)
target_link_libraries(test_phi_api PRIVATE :core_avx.so) # special name
target_link_libraries(test_phi_api PRIVATE ${Python3_LIBRARIES})
