/**
 * Main TritonGPU to LLVM conversion pass for Metal backend.
 *
 * This pass converts Triton GPU IR to LLVM IR targeting Apple's AIR format.
 * It uses AIR intrinsics for simdgroup_matrix operations.
 */

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::Metal {

// AIR address spaces
constexpr unsigned kDeviceMemorySpace = 1;
constexpr unsigned kConstantMemorySpace = 2;
constexpr unsigned kThreadgroupMemorySpace = 3;

//===----------------------------------------------------------------------===//
// Target Info
//===----------------------------------------------------------------------===//

class TargetInfo {
public:
  explicit TargetInfo(StringRef arch) : arch_(arch.str()) {}

  std::string getTargetTriple() const { return "air64-apple-macosx15.0.0"; }

  std::string getDataLayout() const {
    return "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
           "f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-"
           "v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
           "v512:512:512-v1024:1024:1024-n8:16:32";
  }

  StringRef getArch() const { return arch_; }

  static StringRef getSimdgroupMatrixInitF32() {
    return "air.simdgroup_matrix_8x8_init_diag.v64f32.f32";
  }

  static StringRef getSimdgroupMatrixLoadF32() {
    return "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
  }

  static StringRef getSimdgroupMatrixStoreF32() {
    return "air.simdgroup_matrix_8x8_store.v64f32.p1f32";
  }

  static StringRef getSimdgroupMatrixMulAccF32() {
    return "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32";
  }

  static StringRef getThreadgroupBarrier() {
    return "air.wg.barrier";
  }

  static VectorType getSimdgroupMatrixF32Type(MLIRContext *ctx) {
    return VectorType::get({64}, Float32Type::get(ctx));
  }

private:
  std::string arch_;
};

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class MetalTypeConverter : public LLVMTypeConverter {
public:
  MetalTypeConverter(MLIRContext *ctx, const LowerToLLVMOptions &options)
      : LLVMTypeConverter(ctx, options) {

    // Convert Triton pointer types to LLVM pointers with device address space
    addConversion([&](triton::PointerType type) -> std::optional<Type> {
      return LLVM::LLVMPointerType::get(ctx, kDeviceMemorySpace);
    });

    // Convert tensor types to element type (scalar lowering)
    addConversion([&](RankedTensorType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      // Convert element type if needed
      if (auto converted = convertType(elemType)) {
        return converted;
      }
      return elemType;
    });
  }
};

//===----------------------------------------------------------------------===//
// Operation Converters
//===----------------------------------------------------------------------===//

// Convert tt.get_program_id
struct GetProgramIdOpConversion : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // For now, emit a placeholder constant
    // In real implementation, this would read from kernel arguments or intrinsics
    auto result = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

// Convert tt.make_range
struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int32_t start = op.getStart();
    auto result = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(start));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

// Convert tt.splat - pass through scalar value
struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Convert tt.addptr to pointer arithmetic
struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    auto ptrType = cast<LLVM::LLVMPointerType>(ptr.getType());
    auto gep = LLVM::GEPOp::create(
        rewriter, loc, ptrType, rewriter.getF32Type(), ptr, offset);
    rewriter.replaceOp(op, gep.getResult());
    return success();
  }
};

// Convert tt.load
struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto load = LLVM::LoadOp::create(rewriter, loc, resultType, ptr);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

// Convert tt.store
struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tt.broadcast - pass through
struct BroadcastOpConversion : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Convert tt.expand_dims - pass through
struct ExpandDimsOpConversion : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Convert ttg.convert_layout - pass through
struct ConvertLayoutOpConversion : public OpConversionPattern<triton::gpu::ConvertLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

// Convert tt.dot using AIR simdgroup_matrix intrinsics
struct DotOpConversion : public OpConversionPattern<triton::DotOp> {
  DotOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                  const TargetInfo &targetInfo)
      : OpConversionPattern(typeConverter, context), targetInfo_(targetInfo) {}

  LogicalResult matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto aType = cast<RankedTensorType>(op.getA().getType());

    int64_t M = resultType.getShape()[0];
    int64_t N = resultType.getShape()[1];
    int64_t K = aType.getShape()[1];

    // Check if we can use simdgroup_matrix (8x8 tiles)
    bool canUseSimdgroup = (M % 8 == 0) && (N % 8 == 0) && (K % 8 == 0);

    Type elemType = getTypeConverter()->convertType(resultType.getElementType());
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();

    if (canUseSimdgroup && elemType.isF32()) {
      // For simdgroup-compatible sizes with f32, emit AIR intrinsics
      auto simdMatrixType = TargetInfo::getSimdgroupMatrixF32Type(ctx);

      // Initialize accumulator with zero
      auto zero = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0f));

      // Call air.simdgroup_matrix_8x8_init_diag to initialize accumulator
      auto acc = LLVM::CallIntrinsicOp::create(
          rewriter, loc, TypeRange{simdMatrixType}, ValueRange{zero});
      acc.getProperties().setIntrin(rewriter.getStringAttr(
          TargetInfo::getSimdgroupMatrixInitF32()));
      acc.getProperties().setOpBundleSizes(rewriter.getDenseI32ArrayAttr({}));
      acc.getProperties().setOperandSegmentSizes({1, 0});

      // TODO: Full simdgroup multiply-accumulate implementation
      // For now, extract element 0 as the result
      auto zeroIdx = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      auto extracted = LLVM::ExtractElementOp::create(
          rewriter, loc, elemType, acc.getResult(0), zeroIdx);

      rewriter.replaceOp(op, extracted.getResult());
    } else {
      // Fallback to FMA
      auto fma = LLVM::FMAOp::create(rewriter, loc, elemType, a, b, c);
      rewriter.replaceOp(op, fma.getResult());
    }

    return success();
  }

private:
  const TargetInfo &targetInfo_;
};

// Convert tt.return
struct ReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

class ConvertTritonMetalToLLVMPass
    : public PassWrapper<ConvertTritonMetalToLLVMPass, OperationPass<ModuleOp>> {
public:
  ConvertTritonMetalToLLVMPass() = default;
  ConvertTritonMetalToLLVMPass(StringRef arch) : arch_(arch.str()) {}

  StringRef getArgument() const override { return "convert-triton-metal-to-llvm"; }
  StringRef getDescription() const override {
    return "Convert Triton GPU IR to LLVM IR for Metal/AIR target";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    TargetInfo targetInfo(arch_);

    LowerToLLVMOptions options(ctx);
    options.overrideIndexBitwidth(32);
    MetalTypeConverter typeConverter(ctx, options);

    // Set up conversion target
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<triton::TritonDialect>();
    target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // Collect patterns
    RewritePatternSet patterns(ctx);

    // Add Triton-specific patterns
    patterns.add<GetProgramIdOpConversion>(typeConverter, ctx);
    patterns.add<MakeRangeOpConversion>(typeConverter, ctx);
    patterns.add<SplatOpConversion>(typeConverter, ctx);
    patterns.add<AddPtrOpConversion>(typeConverter, ctx);
    patterns.add<LoadOpConversion>(typeConverter, ctx);
    patterns.add<StoreOpConversion>(typeConverter, ctx);
    patterns.add<BroadcastOpConversion>(typeConverter, ctx);
    patterns.add<ExpandDimsOpConversion>(typeConverter, ctx);
    patterns.add<ConvertLayoutOpConversion>(typeConverter, ctx);
    patterns.add<DotOpConversion>(typeConverter, ctx, targetInfo);
    patterns.add<ReturnOpConversion>(typeConverter, ctx);

    // Add standard dialect conversions
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    // Apply conversion
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Set module attributes for AIR target
    mod->setAttr("llvm.target_triple",
                 StringAttr::get(ctx, targetInfo.getTargetTriple()));
    mod->setAttr("llvm.data_layout",
                 StringAttr::get(ctx, targetInfo.getDataLayout()));
  }

private:
  std::string arch_ = "apple-m1";
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
createConvertTritonMetalToLLVMPass(StringRef arch) {
  return std::make_unique<ConvertTritonMetalToLLVMPass>(arch);
}

// Function to convert module to LLVM IR string for compilation
std::string translateModuleToLLVMIR(ModuleOp module, StringRef arch) {
  // Create and run the pass
  PassManager pm(module.getContext());
  pm.addPass(createConvertTritonMetalToLLVMPass(arch));

  if (failed(pm.run(module))) {
    return "";
  }

  // Serialize to LLVM IR string
  std::string llvmIR;
  llvm::raw_string_ostream os(llvmIR);
  module.print(os);
  return llvmIR;
}

} // namespace mlir::triton::Metal
