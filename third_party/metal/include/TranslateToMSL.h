/**
 * Header for TTGIR to MSL translation.
 */

#ifndef TRITON_METAL_TRANSLATE_TO_MSL_H
#define TRITON_METAL_TRANSLATE_TO_MSL_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir {
namespace triton {
namespace metal {

/// Translate a Triton GPU IR module to Metal Shading Language source code.
std::string translateToMSL(ModuleOp module);

} // namespace metal
} // namespace triton
} // namespace mlir

#endif // TRITON_METAL_TRANSLATE_TO_MSL_H
