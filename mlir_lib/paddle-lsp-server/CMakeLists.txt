get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIRLspServerLib MLIRPaddleDialect)
add_llvm_executable(paddle-lsp-server paddle-lsp-server.cpp)

llvm_update_compile_flags(paddle-lsp-server)
target_link_libraries(paddle-lsp-server PRIVATE ${LIBS})
