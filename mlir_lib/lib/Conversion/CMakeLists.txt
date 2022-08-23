add_subdirectory(PaddleToTosa)

add_mlir_library(PaddleMLIRConversionPasses
  Passes.cpp

  DEPENDS
  PaddleMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  PaddleMLIRPaddleToTosa
)
