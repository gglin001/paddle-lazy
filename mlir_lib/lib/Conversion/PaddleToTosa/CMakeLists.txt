add_mlir_conversion_library(PaddleMLIRPaddleToTosa
  PaddleToTosa.cpp

  DEPENDS
  PaddleMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRPaddle
)
