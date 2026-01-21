/**
 * Metal backend plugin for Triton.
 * Exposes MSL translation and LLVM lowering to Python via pybind11.
 */

#include "TranslateToMSL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

// Forward declaration for LLVM lowering pass
namespace mlir::triton::Metal {
std::unique_ptr<Pass>
createConvertTritonMetalToLLVMPass(llvm::StringRef arch);
}

void init_triton_metal(py::module &&m) {
  m.doc() = "Metal backend for Triton - supports both MSL and LLVM/AIR lowering";

  // Translate TTGIR module to MSL source code (legacy path)
  m.def("translate_to_msl",
        [](mlir::ModuleOp mod) -> std::string {
          return mlir::triton::metal::translateToMSL(mod);
        },
        "Translate Triton GPU IR module to Metal Shading Language source code");

  // Translate TTGIR module to LLVM IR targeting AIR (Option C path)
  m.def("translate_to_llvmir",
        [](mlir::ModuleOp mod, const std::string &arch) -> std::string {
          // Clone the module to avoid modifying the original
          mlir::OwningOpRef<mlir::ModuleOp> clonedMod = mod.clone();

          // Create and run the LLVM lowering pass
          mlir::PassManager pm(clonedMod->getContext());
          pm.addPass(mlir::triton::Metal::createConvertTritonMetalToLLVMPass(arch));

          if (mlir::failed(pm.run(*clonedMod))) {
            return "";
          }

          // Serialize to LLVM IR string
          std::string llvmIR;
          llvm::raw_string_ostream os(llvmIR);
          clonedMod->print(os);
          return llvmIR;
        },
        py::arg("mod"), py::arg("arch") = "apple-m1",
        "Convert Triton GPU IR module to LLVM IR targeting AIR");

  // Metal-specific utilities
  m.def("get_threads_per_warp", []() { return 32; },
        "Get the number of threads per warp (SIMD width) for Metal");

  m.def("get_max_shared_memory", []() { return 32768; },
        "Get the maximum threadgroup (shared) memory in bytes");

  m.def("get_compute_capability", []() { return "apple-m1"; },
        "Get the Metal compute capability string");

  // AIR target information
  m.def("get_air_target_triple", []() { return "air64-apple-macosx15.0.0"; },
        "Get the AIR target triple for LLVM");

  m.def("get_air_data_layout", []() {
    return "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
           "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-"
           "v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
           "v512:512:512-v1024:1024:1024-n8:16:32";
  }, "Get the AIR data layout for LLVM");
}

PYBIND11_MODULE(metal, m) {
  init_triton_metal(std::move(m));
}
