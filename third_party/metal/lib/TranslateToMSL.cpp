/**
 * Direct TTGIR to MSL (Metal Shading Language) translation.
 *
 * This translates Triton GPU IR directly to MSL source code,
 * handling BlockedEncodingAttr for thread-to-element mapping.
 * Uses simdgroup_matrix for hardware-accelerated matrix multiply.
 *
 * Key insight from handoff: The scalar emitter approach cannot correctly
 * handle Triton's tensor semantics. For matmul, we use simdgroup_matrix
 * which provides 8x8 hardware-accelerated matrix operations.
 */

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iomanip>

namespace mlir {
namespace triton {
namespace metal {

namespace {

// Tensor shape information for 2D tensor operations
struct TensorInfo {
  std::vector<int64_t> shape;
  bool is2D() const { return shape.size() == 2; }
  int64_t rows() const { return shape.size() >= 1 ? shape[0] : 1; }
  int64_t cols() const { return shape.size() >= 2 ? shape[1] : 1; }
};

// Information about a matmul operation detected in the function
struct MatmulInfo {
  bool hasMatmul = false;
  int64_t BLOCK_M = 0;
  int64_t BLOCK_N = 0;
  int64_t BLOCK_K = 0;
  Type elemType;
  // Track argument positions
  int numPtrArgs = 0;    // Number of pointer arguments
  int numScalarArgs = 0; // Number of scalar arguments
};

// Maps MLIR values to MSL variable names
class ValueMapper {
public:
  std::string getOrCreate(Value v) {
    auto it = valueMap_.find(v.getAsOpaquePointer());
    if (it != valueMap_.end()) {
      return it->second;
    }
    std::string name = "v" + std::to_string(counter_++);
    valueMap_[v.getAsOpaquePointer()] = name;
    return name;
  }

  std::string get(Value v) {
    auto it = valueMap_.find(v.getAsOpaquePointer());
    if (it != valueMap_.end()) {
      return it->second;
    }
    return "v_unknown";
  }

  void set(Value v, const std::string& name) {
    valueMap_[v.getAsOpaquePointer()] = name;
  }

  bool has(Value v) {
    return valueMap_.find(v.getAsOpaquePointer()) != valueMap_.end();
  }

private:
  std::unordered_map<const void*, std::string> valueMap_;
  int counter_ = 0;
};

// Convert MLIR type to MSL type string
std::string typeToMSL(Type type) {
  if (type.isF32()) return "float";
  if (type.isF16()) return "half";
  if (type.isBF16()) return "bfloat";
  if (type.isF64()) return "float";  // Metal doesn't have double
  if (type.isInteger(1)) return "bool";
  if (type.isInteger(8)) return "char";
  if (type.isInteger(16)) return "short";
  if (type.isInteger(32)) return "int";
  if (type.isInteger(64)) return "long";
  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return "device " + typeToMSL(ptrType.getPointeeType()) + "*";
  }
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return typeToMSL(tensorType.getElementType());
  }
  return "float";
}

// Get element type from a potentially tensor type
Type getElementType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getElementType();
  }
  return type;
}

// Get tensor shape
TensorInfo getTensorInfo(Type type) {
  TensorInfo info;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    for (auto dim : tensorType.getShape()) {
      info.shape.push_back(dim);
    }
  }
  return info;
}

// Check if type is a tensor type
bool isTensorType(Type type) {
  return isa<RankedTensorType>(type);
}

// Check if a for loop contains a tt.dot operation (matmul K-loop)
bool isMatmulKLoop(scf::ForOp forOp) {
  bool hasDot = false;
  forOp.walk([&](triton::DotOp dotOp) {
    hasDot = true;
  });
  return hasDot;
}

// MSL code emitter
class MSLEmitter {
public:
  MSLEmitter() {}

  std::string emit(ModuleOp module) {
    // Emit header with simdgroup matrix support
    os_ << "#include <metal_stdlib>\n";
    os_ << "#include <metal_simdgroup_matrix>\n";
    os_ << "using namespace metal;\n\n";

    // Process each function (prefer triton::FuncOp over func::FuncOp)
    module.walk([&](triton::FuncOp funcOp) {
      std::string name = funcOp.getName().str();
      if (emittedFunctions_.find(name) == emittedFunctions_.end()) {
        emittedFunctions_.insert(name);
        emitFunction(funcOp);
      }
    });

    // Also handle regular MLIR functions if not already emitted
    module.walk([&](func::FuncOp funcOp) {
      std::string name = funcOp.getName().str();
      if (emittedFunctions_.find(name) == emittedFunctions_.end()) {
        emittedFunctions_.insert(name);
        emitMLIRFunction(funcOp);
      }
    });

    return os_.str();
  }

private:
  // Analyze function to find matmul operations and their tile sizes
  MatmulInfo analyzeForMatmul(Operation* funcOp) {
    MatmulInfo info;
    funcOp->walk([&](triton::DotOp dotOp) {
      info.hasMatmul = true;
      auto resultType = cast<RankedTensorType>(dotOp.getResult().getType());
      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      info.BLOCK_M = resultType.getShape()[0];
      info.BLOCK_N = resultType.getShape()[1];
      info.BLOCK_K = aType.getShape()[1];
      info.elemType = resultType.getElementType();
    });
    return info;
  }

  // Count pointer and scalar arguments
  void countArguments(triton::FuncOp funcOp, MatmulInfo& info) {
    auto argTypes = funcOp.getArgumentTypes();
    for (size_t i = 0; i < argTypes.size(); ++i) {
      if (isa<triton::PointerType>(argTypes[i])) {
        info.numPtrArgs++;
      } else {
        info.numScalarArgs++;
      }
    }
  }

  void emitFunction(triton::FuncOp funcOp) {
    funcName_ = funcOp.getName().str();

    // Reset per-function state
    matmulInfo_ = analyzeForMatmul(funcOp);
    countArguments(funcOp, matmulInfo_);
    loadCounter_ = 0;
    inMatmulKLoop_ = false;

    // Check if we should use specialized simdgroup_matrix path
    // Requirements:
    // - Has matmul operation
    // - Block sizes are multiples of 8 (simdgroup_matrix tile size)
    // - Element type is f32 (or f16 for half-precision path)
    bool useSimdgroupMatmul = matmulInfo_.hasMatmul &&
                               (matmulInfo_.BLOCK_M % 8 == 0) &&
                               (matmulInfo_.BLOCK_N % 8 == 0) &&
                               (matmulInfo_.BLOCK_K % 8 == 0) &&
                               (matmulInfo_.elemType.isF32() || matmulInfo_.elemType.isF16());

    if (useSimdgroupMatmul) {
      emitSimdgroupMatmulKernel(funcOp);
      return;
    }

    // Fallback: emit standard kernel for non-matmul operations
    emitStandardKernel(funcOp);
  }

  // Emit standard kernel (for non-matmul ops like vector add)
  void emitStandardKernel(triton::FuncOp funcOp) {
    // Emit kernel signature
    os_ << "[[kernel]] void " << funcName_ << "(\n";

    // Emit arguments
    auto argTypes = funcOp.getArgumentTypes();
    int bufferIdx = 0;
    for (size_t i = 0; i < argTypes.size(); ++i) {
      Type argType = argTypes[i];
      std::string argName = "arg" + std::to_string(i);
      mapper_.set(funcOp.getArgument(i), argName);

      if (i > 0) os_ << ",\n";

      if (isa<triton::PointerType>(argType)) {
        auto ptrType = cast<triton::PointerType>(argType);
        os_ << "    device " << typeToMSL(ptrType.getPointeeType())
            << "* " << argName << " [[buffer(" << bufferIdx++ << ")]]";
      } else {
        os_ << "    constant " << typeToMSL(argType) << "& " << argName
            << " [[buffer(" << bufferIdx++ << ")]]";
      }
    }

    // Add thread position arguments
    os_ << ",\n    uint3 tid [[thread_position_in_threadgroup]]";
    os_ << ",\n    uint3 tgid [[threadgroup_position_in_grid]]";
    os_ << ",\n    uint3 tgsize [[threads_per_threadgroup]]";
    os_ << ",\n    uint simd_lid [[thread_index_in_simdgroup]]";
    os_ << ",\n    uint simd_gid [[simdgroup_index_in_threadgroup]]";
    os_ << "\n) {\n";

    // Compute linear thread ID and program ID
    os_ << "    uint linear_tid = tid.x + tid.y * tgsize.x + tid.z * tgsize.x * tgsize.y;\n";
    os_ << "    uint pid_x = tgid.x;\n";
    os_ << "    uint pid_y = tgid.y;\n";
    os_ << "    uint pid_z = tgid.z;\n\n";

    // Process operations in order
    for (Block &block : funcOp.getBody()) {
      for (Operation &op : block.getOperations()) {
        emitOperation(&op);
      }
    }

    os_ << "}\n\n";
  }

  /**
   * Emit a specialized matmul kernel using simdgroup_matrix hardware acceleration.
   *
   * This is the key implementation for Option B from the handoff document.
   * It uses Apple's simdgroup_matrix for 8x8 hardware-accelerated matrix tiles.
   *
   * Key design decisions:
   * 1. Each simdgroup (32 threads) computes one 8x8 output tile
   * 2. Multiple simdgroups work together to compute larger tiles (BLOCK_M x BLOCK_N)
   * 3. Cooperative loading fills shared memory with A and B tiles
   * 4. K-loop iterates through the shared dimension
   *
   * For a 16x16 tile with 4 warps (128 threads = 4 simdgroups):
   * - simd_gid 0: [0:8, 0:8]
   * - simd_gid 1: [0:8, 8:16]
   * - simd_gid 2: [8:16, 0:8]
   * - simd_gid 3: [8:16, 8:16]
   */
  void emitSimdgroupMatmulKernel(triton::FuncOp funcOp) {
    std::string typeStr = matmulInfo_.elemType.isF16() ? "half" : "float";
    int64_t BLOCK_M = matmulInfo_.BLOCK_M;
    int64_t BLOCK_N = matmulInfo_.BLOCK_N;
    int64_t BLOCK_K = matmulInfo_.BLOCK_K;
    const int64_t SIMD_SIZE = 8;  // simdgroup_matrix tile size

    // Calculate number of simdgroups needed
    int64_t simdgroups_m = BLOCK_M / SIMD_SIZE;
    int64_t simdgroups_n = BLOCK_N / SIMD_SIZE;
    int64_t total_simdgroups = simdgroups_m * simdgroups_n;

    // Emit kernel signature matching Triton's actual arguments
    // Typical matmul signature: (a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)
    os_ << "[[kernel]] void " << funcName_ << "(\n";

    auto argTypes = funcOp.getArgumentTypes();
    int bufferIdx = 0;
    std::vector<std::string> argNames;

    for (size_t i = 0; i < argTypes.size(); ++i) {
      Type argType = argTypes[i];
      std::string argName = "arg" + std::to_string(i);
      argNames.push_back(argName);
      mapper_.set(funcOp.getArgument(i), argName);

      if (i > 0) os_ << ",\n";

      if (isa<triton::PointerType>(argType)) {
        auto ptrType = cast<triton::PointerType>(argType);
        os_ << "    device " << typeToMSL(ptrType.getPointeeType())
            << "* " << argName << " [[buffer(" << bufferIdx++ << ")]]";
      } else {
        os_ << "    constant " << typeToMSL(argType) << "& " << argName
            << " [[buffer(" << bufferIdx++ << ")]]";
      }
    }

    // Add Metal-specific thread position arguments
    os_ << ",\n    uint3 tgid [[threadgroup_position_in_grid]]";
    os_ << ",\n    uint simd_lid [[thread_index_in_simdgroup]]";
    os_ << ",\n    uint simd_gid [[simdgroup_index_in_threadgroup]]";
    os_ << ",\n    uint3 tgsize [[threads_per_threadgroup]]";
    os_ << "\n) {\n";

    // Map argument names to semantic roles based on typical matmul kernel signature
    // Expected order: a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
    std::string A_ptr = argNames.size() > 0 ? argNames[0] : "arg0";
    std::string B_ptr = argNames.size() > 1 ? argNames[1] : "arg1";
    std::string C_ptr = argNames.size() > 2 ? argNames[2] : "arg2";
    std::string M_arg = argNames.size() > 3 ? argNames[3] : "arg3";
    std::string N_arg = argNames.size() > 4 ? argNames[4] : "arg4";
    std::string K_arg = argNames.size() > 5 ? argNames[5] : "arg5";
    std::string stride_am = argNames.size() > 6 ? argNames[6] : "arg6";
    std::string stride_ak = argNames.size() > 7 ? argNames[7] : "arg7";
    std::string stride_bk = argNames.size() > 8 ? argNames[8] : "arg8";
    std::string stride_bn = argNames.size() > 9 ? argNames[9] : "arg9";
    std::string stride_cm = argNames.size() > 10 ? argNames[10] : "arg10";
    std::string stride_cn = argNames.size() > 11 ? argNames[11] : "arg11";

    // Constants for tile sizes
    os_ << "    // Tile sizes\n";
    os_ << "    constexpr uint BLOCK_M = " << BLOCK_M << ";\n";
    os_ << "    constexpr uint BLOCK_N = " << BLOCK_N << ";\n";
    os_ << "    constexpr uint BLOCK_K = " << BLOCK_K << ";\n";
    os_ << "    constexpr uint SIMD_SIZE = " << SIMD_SIZE << ";  // simdgroup_matrix tile size\n\n";

    // Shared memory for tiles
    os_ << "    // Shared memory for A and B tiles\n";
    os_ << "    threadgroup " << typeStr << " A_shared[BLOCK_M][BLOCK_K];\n";
    os_ << "    threadgroup " << typeStr << " B_shared[BLOCK_K][BLOCK_N];\n\n";

    // Compute tile position based on program ID (threadgroup position)
    os_ << "    // Compute this threadgroup's tile position\n";
    os_ << "    uint tile_m = tgid.x * BLOCK_M;\n";
    os_ << "    uint tile_n = tgid.y * BLOCK_N;\n\n";

    // Each simdgroup handles one 8x8 output sub-tile
    os_ << "    // Each simdgroup computes one 8x8 output sub-tile\n";
    os_ << "    // Layout: simd_gid maps to (row_block, col_block) in row-major order\n";
    os_ << "    uint simd_row = simd_gid / " << simdgroups_n << ";  // Which 8-row block\n";
    os_ << "    uint simd_col = simd_gid % " << simdgroups_n << ";  // Which 8-col block\n";
    os_ << "    uint sub_m = simd_row * SIMD_SIZE;  // Starting row within tile\n";
    os_ << "    uint sub_n = simd_col * SIMD_SIZE;  // Starting col within tile\n\n";

    // Declare simdgroup_matrix accumulators
    os_ << "    // Accumulator for 8x8 output tile using simdgroup_matrix\n";
    os_ << "    simdgroup_matrix<" << typeStr << ", 8, 8> acc;\n";
    os_ << "    acc = simdgroup_matrix<" << typeStr << ", 8, 8>(0.0";
    os_ << (matmulInfo_.elemType.isF16() ? "h" : "f") << ");  // Initialize to zero\n\n";

    // Linear thread ID within threadgroup for cooperative loading
    os_ << "    // Linear thread ID for cooperative loading\n";
    os_ << "    uint linear_tid = simd_gid * 32 + simd_lid;\n";
    os_ << "    uint num_threads = " << total_simdgroups * 32 << ";  // Total threads per threadgroup\n\n";

    // K-loop
    os_ << "    // Main K-loop\n";
    os_ << "    for (int k_base = 0; k_base < " << K_arg << "; k_base += BLOCK_K) {\n";

    // Cooperative load of A tile into shared memory
    // Each thread loads (BLOCK_M * BLOCK_K) / num_threads elements
    int64_t total_A_elements = BLOCK_M * BLOCK_K;
    int64_t elements_per_thread_A = (total_A_elements + total_simdgroups * 32 - 1) / (total_simdgroups * 32);

    os_ << "        // Cooperative load of A[tile_m:tile_m+BLOCK_M, k_base:k_base+BLOCK_K]\n";
    os_ << "        for (uint i = 0; i < " << elements_per_thread_A << "; i++) {\n";
    os_ << "            uint idx = linear_tid + i * num_threads;\n";
    os_ << "            if (idx < BLOCK_M * BLOCK_K) {\n";
    os_ << "                uint row = idx / BLOCK_K;\n";
    os_ << "                uint col = idx % BLOCK_K;\n";
    os_ << "                uint global_m = tile_m + row;\n";
    os_ << "                uint global_k = k_base + col;\n";
    os_ << "                " << typeStr << " val = 0.0" << (matmulInfo_.elemType.isF16() ? "h" : "f") << ";\n";
    os_ << "                if (global_m < (uint)" << M_arg << " && global_k < (uint)" << K_arg << ") {\n";
    os_ << "                    val = " << A_ptr << "[global_m * " << stride_am << " + global_k * " << stride_ak << "];\n";
    os_ << "                }\n";
    os_ << "                A_shared[row][col] = val;\n";
    os_ << "            }\n";
    os_ << "        }\n\n";

    // Cooperative load of B tile
    int64_t total_B_elements = BLOCK_K * BLOCK_N;
    int64_t elements_per_thread_B = (total_B_elements + total_simdgroups * 32 - 1) / (total_simdgroups * 32);

    os_ << "        // Cooperative load of B[k_base:k_base+BLOCK_K, tile_n:tile_n+BLOCK_N]\n";
    os_ << "        for (uint i = 0; i < " << elements_per_thread_B << "; i++) {\n";
    os_ << "            uint idx = linear_tid + i * num_threads;\n";
    os_ << "            if (idx < BLOCK_K * BLOCK_N) {\n";
    os_ << "                uint row = idx / BLOCK_N;\n";
    os_ << "                uint col = idx % BLOCK_N;\n";
    os_ << "                uint global_k = k_base + row;\n";
    os_ << "                uint global_n = tile_n + col;\n";
    os_ << "                " << typeStr << " val = 0.0" << (matmulInfo_.elemType.isF16() ? "h" : "f") << ";\n";
    os_ << "                if (global_k < (uint)" << K_arg << " && global_n < (uint)" << N_arg << ") {\n";
    os_ << "                    val = " << B_ptr << "[global_k * " << stride_bk << " + global_n * " << stride_bn << "];\n";
    os_ << "                }\n";
    os_ << "                B_shared[row][col] = val;\n";
    os_ << "            }\n";
    os_ << "        }\n\n";

    // Barrier to ensure all loads complete
    os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n";

    // simdgroup_matrix multiply-accumulate
    // Process K dimension in 8-element chunks
    os_ << "        // simdgroup_matrix multiply-accumulate\n";
    os_ << "        // Process K dimension in 8-element chunks\n";
    os_ << "        for (uint k_sub = 0; k_sub < BLOCK_K; k_sub += SIMD_SIZE) {\n";
    os_ << "            simdgroup_matrix<" << typeStr << ", 8, 8> a_mat, b_mat;\n";
    os_ << "            \n";
    os_ << "            // Load A sub-tile: A_shared[sub_m:sub_m+8, k_sub:k_sub+8]\n";
    os_ << "            simdgroup_load(a_mat, &A_shared[sub_m][k_sub], BLOCK_K);\n";
    os_ << "            \n";
    os_ << "            // Load B sub-tile: B_shared[k_sub:k_sub+8, sub_n:sub_n+8]\n";
    os_ << "            simdgroup_load(b_mat, &B_shared[k_sub][sub_n], BLOCK_N);\n";
    os_ << "            \n";
    os_ << "            // Multiply-accumulate: acc += a_mat * b_mat\n";
    os_ << "            simdgroup_multiply_accumulate(acc, a_mat, b_mat, acc);\n";
    os_ << "        }\n\n";

    // Barrier before next K iteration
    os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n\n";

    // Store result to global memory
    os_ << "    // Store 8x8 result tile to global memory\n";
    os_ << "    {\n";
    os_ << "        uint global_m = tile_m + sub_m;\n";
    os_ << "        uint global_n = tile_n + sub_n;\n";
    os_ << "        \n";
    os_ << "        // Use simdgroup_store with bounds checking\n";
    os_ << "        // simdgroup_store handles the 8x8 tile atomically\n";
    os_ << "        if (global_m + 8 <= (uint)" << M_arg << " && global_n + 8 <= (uint)" << N_arg << ") {\n";
    os_ << "            // Full tile fits - use fast path\n";
    os_ << "            simdgroup_store(acc, " << C_ptr << " + global_m * " << stride_cm << " + global_n * " << stride_cn << ", " << stride_cm << ");\n";
    os_ << "        } else {\n";
    os_ << "            // Partial tile - store to threadgroup memory first, then copy with bounds check\n";
    os_ << "            threadgroup " << typeStr << " temp_out[8][8];\n";
    os_ << "            simdgroup_store(acc, &temp_out[0][0], 8);\n";
    os_ << "            threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "            \n";
    os_ << "            // Each thread in the simdgroup copies one element\n";
    os_ << "            uint lane_row = simd_lid / 8;\n";
    os_ << "            uint lane_col = simd_lid % 8;\n";
    os_ << "            for (uint r = lane_row; r < 8; r += 4) {\n";
    os_ << "                for (uint c = lane_col; c < 8; c += 8) {\n";
    os_ << "                    uint out_m = global_m + r;\n";
    os_ << "                    uint out_n = global_n + c;\n";
    os_ << "                    if (out_m < (uint)" << M_arg << " && out_n < (uint)" << N_arg << ") {\n";
    os_ << "                        " << C_ptr << "[out_m * " << stride_cm << " + out_n * " << stride_cn << "] = temp_out[r][c];\n";
    os_ << "                    }\n";
    os_ << "                }\n";
    os_ << "            }\n";
    os_ << "        }\n";
    os_ << "    }\n";

    os_ << "}\n\n";
  }

  void emitMLIRFunction(func::FuncOp funcOp) {
    // Similar to triton::FuncOp handling
    funcName_ = funcOp.getName().str();

    matmulInfo_ = analyzeForMatmul(funcOp);
    loadCounter_ = 0;
    inMatmulKLoop_ = false;

    os_ << "[[kernel]] void " << funcName_ << "(\n";

    auto argTypes = funcOp.getArgumentTypes();
    int bufferIdx = 0;
    for (size_t i = 0; i < argTypes.size(); ++i) {
      Type argType = argTypes[i];
      std::string argName = "arg" + std::to_string(i);
      mapper_.set(funcOp.getArgument(i), argName);

      if (i > 0) os_ << ",\n";

      if (isa<triton::PointerType>(argType)) {
        auto ptrType = cast<triton::PointerType>(argType);
        os_ << "    device " << typeToMSL(ptrType.getPointeeType())
            << "* " << argName << " [[buffer(" << bufferIdx++ << ")]]";
      } else {
        os_ << "    constant " << typeToMSL(argType) << "& " << argName
            << " [[buffer(" << bufferIdx++ << ")]]";
      }
    }

    os_ << ",\n    uint3 tid [[thread_position_in_threadgroup]]";
    os_ << ",\n    uint3 tgid [[threadgroup_position_in_grid]]";
    os_ << ",\n    uint3 tgsize [[threads_per_threadgroup]]";
    os_ << ",\n    uint simd_lid [[thread_index_in_simdgroup]]";
    os_ << ",\n    uint simd_gid [[simdgroup_index_in_threadgroup]]";
    os_ << "\n) {\n";

    os_ << "    uint linear_tid = tid.x + tid.y * tgsize.x + tid.z * tgsize.x * tgsize.y;\n";
    os_ << "    uint pid_x = tgid.x;\n";
    os_ << "    uint pid_y = tgid.y;\n";
    os_ << "    uint pid_z = tgid.z;\n\n";

    for (Block &block : funcOp.getBody()) {
      for (Operation &op : block.getOperations()) {
        emitOperation(&op);
      }
    }

    os_ << "}\n\n";
  }

  void emitOperation(Operation* op) {
    // Skip function ops (already handled)
    if (isa<triton::FuncOp>(op) || isa<func::FuncOp>(op) || isa<ModuleOp>(op)) {
      return;
    }

    // Triton ops
    if (auto getProgId = dyn_cast<triton::GetProgramIdOp>(op)) {
      emitGetProgramId(getProgId);
    } else if (auto makeRange = dyn_cast<triton::MakeRangeOp>(op)) {
      emitMakeRange(makeRange);
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      emitSplat(splat);
    } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(op)) {
      emitAddPtr(addPtr);
    } else if (auto load = dyn_cast<triton::LoadOp>(op)) {
      emitLoad(load);
    } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
      emitStore(store);
    } else if (auto broadcast = dyn_cast<triton::BroadcastOp>(op)) {
      emitBroadcast(broadcast);
    } else if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(op)) {
      emitExpandDims(expandDims);
    } else if (auto dot = dyn_cast<triton::DotOp>(op)) {
      emitDot(dot);
    }
    // TritonGPU ops
    else if (auto convertLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      emitConvertLayout(convertLayout);
    }
    // Arith ops
    else if (auto addI = dyn_cast<arith::AddIOp>(op)) {
      emitBinaryOp(addI, "+");
    } else if (auto addF = dyn_cast<arith::AddFOp>(op)) {
      emitBinaryOp(addF, "+");
    } else if (auto subI = dyn_cast<arith::SubIOp>(op)) {
      emitBinaryOp(subI, "-");
    } else if (auto subF = dyn_cast<arith::SubFOp>(op)) {
      emitBinaryOp(subF, "-");
    } else if (auto mulI = dyn_cast<arith::MulIOp>(op)) {
      emitBinaryOp(mulI, "*");
    } else if (auto mulF = dyn_cast<arith::MulFOp>(op)) {
      emitBinaryOp(mulF, "*");
    } else if (auto divF = dyn_cast<arith::DivFOp>(op)) {
      emitBinaryOp(divF, "/");
    } else if (auto divSI = dyn_cast<arith::DivSIOp>(op)) {
      emitBinaryOp(divSI, "/");
    } else if (auto remSI = dyn_cast<arith::RemSIOp>(op)) {
      emitBinaryOp(remSI, "%");
    } else if (auto cmpI = dyn_cast<arith::CmpIOp>(op)) {
      emitCmpI(cmpI);
    } else if (auto cmpF = dyn_cast<arith::CmpFOp>(op)) {
      emitCmpF(cmpF);
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      emitConstant(constant);
    } else if (auto extSI = dyn_cast<arith::ExtSIOp>(op)) {
      emitCast(extSI, "int");
    } else if (auto extUI = dyn_cast<arith::ExtUIOp>(op)) {
      emitCast(extUI, "uint");
    } else if (auto truncI = dyn_cast<arith::TruncIOp>(op)) {
      emitCast(truncI, typeToMSL(op->getResult(0).getType()));
    } else if (auto sitofp = dyn_cast<arith::SIToFPOp>(op)) {
      emitCast(sitofp, "float");
    } else if (auto fptosi = dyn_cast<arith::FPToSIOp>(op)) {
      emitCast(fptosi, "int");
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      emitSelect(select);
    } else if (auto andI = dyn_cast<arith::AndIOp>(op)) {
      emitBinaryOp(andI, "&");
    } else if (auto orI = dyn_cast<arith::OrIOp>(op)) {
      emitBinaryOp(orI, "|");
    } else if (auto xorI = dyn_cast<arith::XOrIOp>(op)) {
      emitBinaryOp(xorI, "^");
    } else if (auto negF = dyn_cast<arith::NegFOp>(op)) {
      emitUnaryOp(negF, "-");
    }
    // SCF ops
    else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      emitForLoop(forOp);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      emitIf(ifOp);
    } else if (isa<scf::YieldOp>(op)) {
      // Handled within for/if
    }
    // Return ops
    else if (isa<triton::ReturnOp>(op) || isa<func::ReturnOp>(op)) {
      os_ << "    return;\n";
    }
    // Unknown op - emit as comment
    else {
      os_ << "    // Unhandled op: " << op->getName().getStringRef().str() << "\n";
    }
  }

  void emitGetProgramId(triton::GetProgramIdOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    uint32_t axis = static_cast<uint32_t>(op.getAxis());
    std::string axisVar = (axis == 0) ? "pid_x" : (axis == 1) ? "pid_y" : "pid_z";
    os_ << "    int " << result << " = " << axisVar << ";\n";
  }

  void emitMakeRange(triton::MakeRangeOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    int32_t start = op.getStart();
    int32_t end = op.getEnd();
    int32_t size = end - start;

    // For tensor operations in non-matmul kernels, compute element index
    // based on thread position
    os_ << "    int " << result << " = " << start << " + (linear_tid % " << size << ");\n";
  }

  void emitSplat(triton::SplatOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getSrc());
    Type elemType = getElementType(op.getResult().getType());
    os_ << "    " << typeToMSL(elemType) << " " << result << " = " << src << ";\n";
  }

  void emitAddPtr(triton::AddPtrOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string ptr = mapper_.get(op.getPtr());
    std::string offset = mapper_.get(op.getOffset());
    os_ << "    auto " << result << " = " << ptr << " + " << offset << ";\n";
  }

  void emitLoad(triton::LoadOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string ptr = mapper_.get(op.getPtr());
    Type elemType = getElementType(op.getResult().getType());
    std::string typeStr = typeToMSL(elemType);

    if (op.getMask()) {
      std::string mask = mapper_.get(op.getMask());
      std::string other = op.getOther() ? mapper_.get(op.getOther()) : "0";
      os_ << "    " << typeStr << " " << result
          << " = " << mask << " ? *" << ptr << " : (" << typeStr << ")" << other << ";\n";
    } else {
      os_ << "    " << typeStr << " " << result << " = *" << ptr << ";\n";
    }
  }

  void emitStore(triton::StoreOp op) {
    std::string ptr = mapper_.get(op.getPtr());
    std::string value = mapper_.get(op.getValue());

    if (op.getMask()) {
      std::string mask = mapper_.get(op.getMask());
      os_ << "    if (" << mask << ") *" << ptr << " = " << value << ";\n";
    } else {
      os_ << "    *" << ptr << " = " << value << ";\n";
    }
  }

  void emitBroadcast(triton::BroadcastOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getSrc());
    os_ << "    auto " << result << " = " << src << ";\n";
  }

  void emitExpandDims(triton::ExpandDimsOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getSrc());
    os_ << "    auto " << result << " = " << src << ";\n";
  }

  void emitDot(triton::DotOp op) {
    // For standard kernels, emit a simple FMA
    // (simdgroup_matrix matmul is handled separately in emitSimdgroupMatmulKernel)
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string a = mapper_.get(op.getA());
    std::string b = mapper_.get(op.getB());
    std::string c = mapper_.get(op.getC());

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elemType = resultType.getElementType();
    std::string typeStr = typeToMSL(elemType);

    os_ << "    " << typeStr << " " << result << " = fma(" << a << ", " << b << ", " << c << ");\n";
  }

  void emitConvertLayout(triton::gpu::ConvertLayoutOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getSrc());
    Type elemType = getElementType(op.getResult().getType());
    os_ << "    " << typeToMSL(elemType) << " " << result << " = " << src << ";\n";
  }

  template<typename OpTy>
  void emitBinaryOp(OpTy op, const std::string& opStr) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string lhs = mapper_.get(op.getLhs());
    std::string rhs = mapper_.get(op.getRhs());
    Type resultType = getElementType(op.getResult().getType());
    os_ << "    " << typeToMSL(resultType) << " " << result
        << " = " << lhs << " " << opStr << " " << rhs << ";\n";
  }

  template<typename OpTy>
  void emitUnaryOp(OpTy op, const std::string& opStr) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getOperand());
    Type resultType = getElementType(op.getResult().getType());
    os_ << "    " << typeToMSL(resultType) << " " << result
        << " = " << opStr << src << ";\n";
  }

  void emitCmpI(arith::CmpIOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string lhs = mapper_.get(op.getLhs());
    std::string rhs = mapper_.get(op.getRhs());

    std::string cmpOp;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq: cmpOp = "=="; break;
      case arith::CmpIPredicate::ne: cmpOp = "!="; break;
      case arith::CmpIPredicate::slt: cmpOp = "<"; break;
      case arith::CmpIPredicate::sle: cmpOp = "<="; break;
      case arith::CmpIPredicate::sgt: cmpOp = ">"; break;
      case arith::CmpIPredicate::sge: cmpOp = ">="; break;
      case arith::CmpIPredicate::ult: cmpOp = "<"; break;
      case arith::CmpIPredicate::ule: cmpOp = "<="; break;
      case arith::CmpIPredicate::ugt: cmpOp = ">"; break;
      case arith::CmpIPredicate::uge: cmpOp = ">="; break;
    }
    os_ << "    bool " << result << " = " << lhs << " " << cmpOp << " " << rhs << ";\n";
  }

  void emitCmpF(arith::CmpFOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string lhs = mapper_.get(op.getLhs());
    std::string rhs = mapper_.get(op.getRhs());

    std::string cmpOp;
    switch (op.getPredicate()) {
      case arith::CmpFPredicate::OEQ: cmpOp = "=="; break;
      case arith::CmpFPredicate::ONE: cmpOp = "!="; break;
      case arith::CmpFPredicate::OLT: cmpOp = "<"; break;
      case arith::CmpFPredicate::OLE: cmpOp = "<="; break;
      case arith::CmpFPredicate::OGT: cmpOp = ">"; break;
      case arith::CmpFPredicate::OGE: cmpOp = ">="; break;
      default: cmpOp = "=="; break;
    }
    os_ << "    bool " << result << " = " << lhs << " " << cmpOp << " " << rhs << ";\n";
  }

  void emitConstant(arith::ConstantOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    Type type = op.getResult().getType();
    Type elemType = getElementType(type);

    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      os_ << "    " << typeToMSL(elemType) << " " << result
          << " = " << intAttr.getInt() << ";\n";
    } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      double val = floatAttr.getValueAsDouble();
      os_ << "    " << typeToMSL(elemType) << " " << result << " = ";
      if (val == 0.0) {
        os_ << "0.0f";
      } else {
        os_ << std::fixed << std::setprecision(6) << val << "f";
      }
      os_ << ";\n";
    } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (denseAttr.isSplat()) {
        if (elemType.isF32() || elemType.isF16()) {
          float val = denseAttr.getSplatValue<float>();
          os_ << "    " << typeToMSL(elemType) << " " << result << " = ";
          if (val == 0.0f) {
            os_ << "0.0f";
          } else {
            os_ << std::fixed << std::setprecision(6) << val << "f";
          }
          os_ << ";\n";
        } else if (elemType.isInteger(32)) {
          os_ << "    int " << result << " = "
              << denseAttr.getSplatValue<int32_t>() << ";\n";
        } else {
          os_ << "    " << typeToMSL(elemType) << " " << result << " = 0;\n";
        }
      } else {
        os_ << "    " << typeToMSL(elemType) << " " << result << " = 0; // non-splat dense\n";
      }
    } else {
      os_ << "    " << typeToMSL(elemType) << " " << result << " = 0; // unknown const\n";
    }
  }

  template<typename OpTy>
  void emitCast(OpTy op, const std::string& targetType) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string src = mapper_.get(op.getIn());
    os_ << "    " << targetType << " " << result
        << " = static_cast<" << targetType << ">(" << src << ");\n";
  }

  void emitSelect(arith::SelectOp op) {
    std::string result = mapper_.getOrCreate(op.getResult());
    std::string cond = mapper_.get(op.getCondition());
    std::string trueVal = mapper_.get(op.getTrueValue());
    std::string falseVal = mapper_.get(op.getFalseValue());
    Type resultType = getElementType(op.getResult().getType());
    os_ << "    " << typeToMSL(resultType) << " " << result
        << " = " << cond << " ? " << trueVal << " : " << falseVal << ";\n";
  }

  void emitForLoop(scf::ForOp op) {
    std::string iv = mapper_.getOrCreate(op.getInductionVar());
    std::string lb = mapper_.get(op.getLowerBound());
    std::string ub = mapper_.get(op.getUpperBound());
    std::string step = mapper_.get(op.getStep());

    // Handle iter_args - these are loop-carried values
    auto initArgs = op.getInitArgs();
    auto iterArgs = op.getRegionIterArgs();
    auto results = op.getResults();

    // Declare loop-carried variables before the loop
    for (size_t i = 0; i < initArgs.size(); ++i) {
      std::string initVar = mapper_.get(initArgs[i]);
      std::string iterVar = mapper_.getOrCreate(iterArgs[i]);
      std::string resultVar = mapper_.getOrCreate(results[i]);
      Type elemType = getElementType(iterArgs[i].getType());

      os_ << "    " << typeToMSL(elemType) << " " << iterVar << " = " << initVar << ";\n";
      mapper_.set(results[i], iterVar);
    }

    os_ << "    for (int " << iv << " = " << lb << "; "
        << iv << " < " << ub << "; " << iv << " += " << step << ") {\n";

    // Emit loop body (excluding the yield)
    for (Operation& bodyOp : op.getBody()->without_terminator()) {
      emitOperation(&bodyOp);
    }

    // Handle yield - update loop-carried variables
    if (auto yield = dyn_cast<scf::YieldOp>(op.getBody()->getTerminator())) {
      auto yieldedValues = yield.getOperands();
      for (size_t i = 0; i < yieldedValues.size(); ++i) {
        std::string yieldedVar = mapper_.get(yieldedValues[i]);
        std::string iterVar = mapper_.get(iterArgs[i]);
        if (yieldedVar != iterVar) {
          os_ << "        " << iterVar << " = " << yieldedVar << ";\n";
        }
      }
    }

    os_ << "    }\n";
  }

  void emitIf(scf::IfOp op) {
    std::string cond = mapper_.get(op.getCondition());

    os_ << "    if (" << cond << ") {\n";

    for (Operation& thenOp : op.getThenRegion().front().without_terminator()) {
      emitOperation(&thenOp);
    }

    if (!op.getElseRegion().empty()) {
      os_ << "    } else {\n";
      for (Operation& elseOp : op.getElseRegion().front().without_terminator()) {
        emitOperation(&elseOp);
      }
    }

    os_ << "    }\n";
  }

  std::ostringstream os_;
  ValueMapper mapper_;
  std::string funcName_;
  std::unordered_set<std::string> emittedFunctions_;

  // Matmul-specific state
  MatmulInfo matmulInfo_;
  int loadCounter_ = 0;
  bool inMatmulKLoop_ = false;
};

} // anonymous namespace

// Main entry point for translating TTGIR module to MSL
std::string translateToMSL(ModuleOp module) {
  MSLEmitter emitter;
  return emitter.emit(module);
}

} // namespace metal
} // namespace triton
} // namespace mlir
