"""
Metal backend compiler for Triton.
Handles TTIR -> TTGIR -> MSL -> metallib compilation pipeline.

Option C (LLVM-based) approach:
- Detects matmul patterns and generates specialized simdgroup_matrix code
- Uses Apple's Metal toolchain which is LLVM-based internally
- Targets AIR (Apple Intermediate Representation) through MSL
"""

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, metal

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import subprocess


@dataclass(frozen=True)
class MetalOptions:
    """Compilation options for Metal backend."""
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    cluster_dims: tuple = (1, 1, 1)
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    backend_name: str = 'metal'
    arch: str = None
    instrumentation_mode: str = ""
    # Metal-specific options
    threads_per_warp: int = 32  # Metal SIMD width
    max_threadgroup_memory: int = 32768  # 32KB typical for Apple GPUs
    # Option C: Use LLVM-based lowering
    use_llvm_lowering: bool = False

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):
    """
    Metal backend for Apple GPUs.

    Compilation pipeline:
    1. TTIR: Triton IR with standard optimizations
    2. TTGIR: TritonGPU IR with BlockedEncodingAttr
    3. MSL: Metal Shading Language source (with simdgroup_matrix for matmul)
    4. metallib: Compiled Metal library binary (via AIR)
    """

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'metal'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in MetalOptions.__dataclass_fields__.keys()
                    if k in opts and opts[k] is not None})
        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        """Pack metadata for kernel launch."""
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            1, 1, 1,  # cluster_dims (not used on Metal)
        )

    def get_codegen_implementation(self, options):
        """Return codegen functions for Metal-specific type conversions."""
        codegen_fns = {
            "min_dot_size": self._min_dot_size()
        }
        return codegen_fns

    def _min_dot_size(self):
        """Return minimum dot product size for Metal."""
        def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:
            # simdgroup_matrix uses 8x8 tiles
            return (8, 8, 8)  # (min_m, min_n, min_k)
        return check_dot_compatibility

    def get_module_map(self) -> Dict[str, ModuleType]:
        """Return module map for Metal-specific libraries."""
        return {}

    def load_dialects(self, ctx):
        """Load Metal MLIR dialect into context."""
        pass

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """
        Convert to optimized Triton IR.

        This stage applies standard TTIR optimizations that are
        target-independent.
        """
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        """
        Convert TTIR to TritonGPU IR with BlockedEncodingAttr.

        Key transformation: Assigns thread-to-element mapping using
        linear (blocked) layout. Each thread owns multiple elements
        distributed in a strided pattern.
        """
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # Convert to TTGPUIR with blocked layout
        passes.ttir.add_convert_to_ttgpuir(pm, f"metal:{opt.arch}",
                                           opt.num_warps, 32, opt.num_ctas)

        # Optimize memory access patterns
        passes.ttgpuir.add_coalesce(pm)

        # Remove unnecessary layout conversions
        passes.ttgpuir.add_remove_layout_conversions(pm)

        # Standard optimizations
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)

        pm.run(mod, 'make_ttgir')

        # Extract shared memory size from module
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0

        return mod

    def make_msl(self, src, metadata, options):
        """
        Convert TTGIR to Metal Shading Language.

        Uses direct C++ translation to convert TTGIR to MSL source code.
        The MSL code uses simdgroup_matrix for hardware-accelerated matmul.
        """
        # Dump TTGIR for debugging
        if os.environ.get('TRITON_METAL_DEBUG'):
            import sys
            print("=== TTGIR ===", file=sys.stderr)
            print(str(src), file=sys.stderr)
            print("=============", file=sys.stderr)

        # Translate TTGIR to MSL using C++ backend
        msl_code = metal.translate_to_msl(src)

        # Dump MSL for debugging
        if os.environ.get('TRITON_METAL_DEBUG'):
            import sys
            print("=== MSL ===", file=sys.stderr)
            print(msl_code, file=sys.stderr)
            print("===========", file=sys.stderr)

        if not msl_code.strip():
            raise RuntimeError("MSL translation produced empty output")

        metadata["name"] = "triton_kernel"
        return msl_code

    def make_llvmir(self, src, metadata, options):
        """
        Convert TTGIR to LLVM IR targeting AIR.

        This is the Option C path - proper LLVM-based lowering.
        """
        if os.environ.get('TRITON_METAL_DEBUG'):
            import sys
            print("=== TTGIR (for LLVM) ===", file=sys.stderr)
            print(str(src), file=sys.stderr)
            print("========================", file=sys.stderr)

        # Use the LLVM lowering pass
        try:
            llvm_ir = metal.translate_to_llvmir(src, options.arch or "apple-m1")
        except Exception as e:
            import sys
            print(f"LLVM lowering failed: {e}, falling back to MSL", file=sys.stderr)
            return None

        if os.environ.get('TRITON_METAL_DEBUG'):
            import sys
            print("=== LLVM IR ===", file=sys.stderr)
            print(llvm_ir, file=sys.stderr)
            print("===============", file=sys.stderr)

        metadata["name"] = "triton_kernel"
        return llvm_ir

    def make_metallib(self, src, metadata, options):
        """
        Compile MSL source to Metal library binary.

        Uses Apple's Metal toolchain:
        1. MSL -> AIR (Apple Intermediate Representation)
        2. AIR -> metallib (Metal Library)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            msl_path = os.path.join(tmpdir, "kernel.metal")
            air_path = os.path.join(tmpdir, "kernel.air")
            lib_path = os.path.join(tmpdir, "kernel.metallib")

            # Write MSL source
            with open(msl_path, 'w') as f:
                f.write(src)

            try:
                # Compile MSL to AIR
                result = subprocess.run([
                    'xcrun', '-sdk', 'macosx', 'metal',
                    '-c', msl_path,
                    '-o', air_path
                ], check=True, capture_output=True)

                # Link AIR to metallib
                subprocess.run([
                    'xcrun', '-sdk', 'macosx', 'metallib',
                    air_path,
                    '-o', lib_path
                ], check=True, capture_output=True)

                # Read the metallib binary
                with open(lib_path, 'rb') as f:
                    return f.read()

            except subprocess.CalledProcessError as e:
                import sys
                print(f"Metal compilation failed: {e}", file=sys.stderr)
                if e.stderr:
                    print(f"stderr: {e.stderr.decode()}", file=sys.stderr)
                # Also print the MSL source for debugging
                print(f"MSL source:\n{src}", file=sys.stderr)
                raise RuntimeError(f"Metal compilation failed: {e}")
            except FileNotFoundError:
                raise RuntimeError("Metal toolchain not available (xcrun not found)")

    def make_metallib_from_llvmir(self, src, metadata, options):
        """
        Compile LLVM IR to Metal library binary.

        Uses Apple's Metal toolchain with LLVM IR input:
        1. LLVM IR -> AIR (via metal -x ir)
        2. AIR -> metallib
        """
        if src is None:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            llvm_path = os.path.join(tmpdir, "kernel.ll")
            air_path = os.path.join(tmpdir, "kernel.air")
            lib_path = os.path.join(tmpdir, "kernel.metallib")

            # Write LLVM IR source
            with open(llvm_path, 'w') as f:
                f.write(src)

            try:
                # Compile LLVM IR to AIR
                subprocess.run([
                    'xcrun', '-sdk', 'macosx', 'metal',
                    '-x', 'ir',
                    '-c', llvm_path,
                    '-o', air_path
                ], check=True, capture_output=True)

                # Link AIR to metallib
                subprocess.run([
                    'xcrun', '-sdk', 'macosx', 'metallib',
                    air_path,
                    '-o', lib_path
                ], check=True, capture_output=True)

                # Read the metallib binary
                with open(lib_path, 'rb') as f:
                    return f.read()

            except subprocess.CalledProcessError as e:
                import sys
                print(f"LLVM IR compilation failed: {e}", file=sys.stderr)
                if e.stderr:
                    print(f"stderr: {e.stderr.decode()}", file=sys.stderr)
                return None
            except FileNotFoundError:
                return None

    def add_stages(self, stages, options, language=None):
        """
        Register compilation stages.

        Pipeline: ttir -> ttgir -> msl -> metallib
        """
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    def hash(self):
        """Return a unique hash for this backend configuration."""
        try:
            result = subprocess.run(
                ['xcrun', '--show-sdk-version'],
                capture_output=True, text=True
            )
            sdk_version = result.stdout.strip() if result.returncode == 0 else "unknown"
        except FileNotFoundError:
            sdk_version = "unknown"

        return f"metal-{self.target.arch}-sdk{sdk_version}"
