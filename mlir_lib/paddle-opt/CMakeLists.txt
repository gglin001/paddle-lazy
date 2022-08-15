get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIRPaddle)
add_llvm_executable(paddle-opt paddle-opt.cpp)

llvm_update_compile_flags(paddle-opt)
target_link_libraries(paddle-opt PRIVATE ${LIBS})
