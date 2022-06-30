add_library(lib SHARED lib.cc)
target_link_libraries(lib PRIVATE :core_avx.so) # special name

add_executable(test_phi_api test_phi_api.cc)
target_link_libraries(test_phi_api PRIVATE :core_avx.so) # special name
target_link_libraries(test_phi_api PRIVATE ${Python3_LIBRARIES})
