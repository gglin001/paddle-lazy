add_executable(sin sin.cpp)
# TODO(allen) only link needed libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIRPaddle)
target_link_libraries(sin ${LIBS})
