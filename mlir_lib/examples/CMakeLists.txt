# TODO(allen) only link needed libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIRPaddleDialect)

add_executable(sin sin.cpp)
target_link_libraries(sin ${LIBS})

add_executable(relu relu.cpp)
target_link_libraries(relu ${LIBS})

add_executable(conv2d conv2d.cpp)
target_link_libraries(conv2d ${LIBS})
