# Metal backend CMakeLists.txt for Triton
# LLVM-based lowering targeting AIR (Apple Intermediate Representation)

message(STATUS "Configuring Metal backend for Triton")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the MSL translation library (legacy path)
add_triton_library(TritonMetalMSL
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/TranslateToMSL.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRFuncDialect
  TritonIR
  TritonGPUIR
)

target_include_directories(TritonMetalMSL PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build the LLVM lowering library (Option C path)
add_triton_library(TritonMetalToLLVM
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/TritonMetalToLLVM/TritonGPUToLLVM.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRTransforms
  TritonIR
  TritonGPUIR
  TritonGPUToLLVM
)

target_include_directories(TritonMetalToLLVM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build Python module plugin
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMetal ${CMAKE_CURRENT_SOURCE_DIR}/triton_metal.cc
    LINK_LIBS TritonMetalMSL TritonMetalToLLVM)
  target_link_libraries(TritonMetal PRIVATE Python3::Module pybind11::headers)
  target_include_directories(TritonMetal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
