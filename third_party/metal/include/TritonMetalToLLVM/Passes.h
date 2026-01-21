/**
 * Metal backend LLVM lowering passes.
 */

#ifndef TRITON_METAL_PASSES_H
#define TRITON_METAL_PASSES_H

#include "TritonMetalToLLVM/TargetInfo.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::Metal {

// Create the main TritonGPU to LLVM conversion pass for Metal
std::unique_ptr<Pass>
createConvertTritonMetalToLLVMPass(StringRef arch = "apple-m1");

// Populate patterns for converting Triton operations to LLVM with AIR intrinsics
void populateTritonMetalToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const TargetInfo &targetInfo);

// Populate patterns for DotOp conversion using simdgroup_matrix
void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const TargetInfo &targetInfo);

// Populate patterns for memory operations
void populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const TargetInfo &targetInfo);

// Populate patterns for control flow operations
void populateControlFlowToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const TargetInfo &targetInfo);

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_PASSES_H
