/**
 * Metal-specific target information for LLVM lowering.
 *
 * Defines target characteristics like address spaces, intrinsics,
 * and simdgroup properties for Apple GPUs.
 */

#ifndef TRITON_METAL_TARGETINFO_H
#define TRITON_METAL_TARGETINFO_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::Metal {

// AIR (Apple Intermediate Representation) address spaces
constexpr unsigned kDeviceMemorySpace = 1;    // device (global) memory
constexpr unsigned kConstantMemorySpace = 2;  // constant memory
constexpr unsigned kThreadgroupMemorySpace = 3; // threadgroup (shared) memory

// Simdgroup matrix sizes
constexpr unsigned kSimdgroupMatrixSize = 8;  // 8x8 matrix tiles
constexpr unsigned kSimdgroupSize = 32;       // 32 threads per simdgroup

class TargetInfo {
public:
  explicit TargetInfo(StringRef arch) : arch_(arch.str()) {}

  // Returns the target triple for AIR
  std::string getTargetTriple() const {
    return "air64-apple-macosx15.0.0";
  }

  // Returns the data layout for AIR
  std::string getDataLayout() const {
    return "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
           "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-"
           "v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
           "v512:512:512-v1024:1024:1024-n8:16:32";
  }

  // Get the architecture string
  StringRef getArch() const { return arch_; }

  // Threads per warp (simdgroup in Metal)
  unsigned getThreadsPerWarp() const { return kSimdgroupSize; }

  // Maximum threads per threadgroup
  unsigned getMaxThreadsPerBlock() const { return 1024; }

  // Maximum threadgroup memory (32KB typical for Apple GPUs)
  unsigned getMaxSharedMemory() const { return 32768; }

  // Address space for shared/threadgroup memory
  int getSharedAddressSpace() const {
    return kThreadgroupMemorySpace;
  }

  // Get LLVM pointer type for device memory
  LLVM::LLVMPointerType getDevicePointerType(MLIRContext *ctx) const {
    return LLVM::LLVMPointerType::get(ctx, kDeviceMemorySpace);
  }

  // Get LLVM pointer type for threadgroup memory
  LLVM::LLVMPointerType getThreadgroupPointerType(MLIRContext *ctx) const {
    return LLVM::LLVMPointerType::get(ctx, kThreadgroupMemorySpace);
  }

  // AIR intrinsic names for simdgroup matrix operations (f32)
  static StringRef getSimdgroupMatrixInitF32() {
    return "air.simdgroup_matrix_8x8_init_diag.v64f32.f32";
  }

  static StringRef getSimdgroupMatrixLoadF32Threadgroup() {
    return "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
  }

  static StringRef getSimdgroupMatrixLoadF32Device() {
    return "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
  }

  static StringRef getSimdgroupMatrixStoreF32Device() {
    return "air.simdgroup_matrix_8x8_store.v64f32.p1f32";
  }

  static StringRef getSimdgroupMatrixStoreF32Threadgroup() {
    return "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
  }

  static StringRef getSimdgroupMatrixMultiplyAccumulateF32() {
    return "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32";
  }

  static StringRef getThreadgroupBarrier() {
    return "air.wg.barrier";
  }

  // Get the LLVM vector type for simdgroup_matrix<float, 8, 8>
  static VectorType getSimdgroupMatrixF32Type(MLIRContext *ctx) {
    return VectorType::get({64}, Float32Type::get(ctx));
  }

  // Get the LLVM vector type for simdgroup_matrix<half, 8, 8>
  static VectorType getSimdgroupMatrixF16Type(MLIRContext *ctx) {
    return VectorType::get({64}, Float16Type::get(ctx));
  }

private:
  std::string arch_;
};

} // namespace mlir::triton::Metal

#endif // TRITON_METAL_TARGETINFO_H
