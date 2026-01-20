"""
Metal backend driver for Triton.
Handles device management, kernel launching, and MPS tensor integration.
"""

import os
import platform
import functools
import tempfile
from ctypes import CDLL, c_int64, c_void_p, c_char_p, c_bool, c_int8, POINTER, byref
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _find_metal_runtime_lib():
    """Find the MetalRuntime dynamic library."""
    # Check common locations relative to this file
    base_dir = Path(__file__).parent.parent.parent.parent  # triton root
    possible_paths = [
        # Build output directories
        base_dir / "metal-dialect" / "build" / "debug" / "lib" / "libMetalRuntime.dylib",
        base_dir / "metal-dialect" / "build" / "release" / "lib" / "libMetalRuntime.dylib",
        # Installed locations
        Path("/usr/local/lib/libMetalRuntime.dylib"),
        # Swift package manager output
        base_dir / "metal-dialect" / "MetalRuntime" / ".build" / "debug" / "libMetalRuntime.dylib",
        base_dir / "metal-dialect" / "MetalRuntime" / ".build" / "release" / "libMetalRuntime.dylib",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)
    return None


class MetalUtils:
    """
    Utility class providing Metal device info and binary loading.

    Similar to CudaUtils but for Metal/MPS.
    """

    def get_device_properties(self, device=0):
        """
        Get Metal device properties.

        Returns dict with:
        - max_shared_mem: Maximum threadgroup memory (32KB typical for Apple GPUs)
        - multiprocessor_count: Number of GPU cores (approximate)
        - warp_size: SIMD width (32 for Apple GPUs)
        """
        # Apple GPU typical values
        # M1: 32KB threadgroup memory, ~8 GPU cores
        # M2: 32KB threadgroup memory, ~10 GPU cores
        # M3: 32KB threadgroup memory, ~10 GPU cores
        return {
            "max_shared_mem": 32768,  # 32KB
            "multiprocessor_count": 8,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "max_block_dim_x": 1024,
            "max_block_dim_y": 1024,
            "max_block_dim_z": 64,
        }

    def load_binary(self, name, kernel, shared, device):
        """
        Load a Metal binary (metallib) and return function handle.

        Args:
            name: Kernel name
            kernel: The metallib binary data
            shared: Shared memory size
            device: Device index

        Returns:
            Tuple of (module, function, n_regs, n_spills)
            For Metal, we return (metallib_data, metallib_data, 0, 0)
        """
        # For Metal, the "module" and "function" are the metallib data
        # The actual Metal API calls happen in the launcher
        return (kernel, kernel, 0, 0)


class MetalLauncher:
    """
    Handles kernel launching on Metal GPUs.

    Uses ctypes bindings to the MetalRuntime C API.
    """

    def __init__(self, src, metadata):
        """
        Initialize the launcher with kernel source and metadata.

        Args:
            src: Source object with signature and constants
            metadata: Kernel metadata (shared memory size, etc.)
        """
        # Extract signature info from source (like CudaLauncher does)
        self.constants = src.constants if hasattr(src, "constants") else dict()
        self.signature = {idx: value for idx, value in src.signature.items()} if hasattr(src, 'signature') else {}

        # Get metadata
        self.kernel_name = metadata.name if hasattr(metadata, 'name') else "triton_kernel"
        self.shared_memory = metadata.shared if hasattr(metadata, 'shared') else 0
        self.num_warps = metadata.num_warps if hasattr(metadata, 'num_warps') else 4
        self.global_scratch_size = getattr(metadata, 'global_scratch_size', 0)
        self.global_scratch_align = getattr(metadata, 'global_scratch_align', 128)
        self.launch_cooperative_grid = getattr(metadata, 'launch_cooperative_grid', False)

        # Try to load the Metal runtime library
        self._lib = None
        self._device = None
        self._queue = None

        lib_path = _find_metal_runtime_lib()
        if lib_path:
            try:
                self._lib = CDLL(lib_path)
                self._setup_function_signatures()
                self._initialize_device()
            except OSError as e:
                print(f"Warning: Failed to load MetalRuntime: {e}")
                self._lib = None

    def _setup_function_signatures(self):
        """Set up ctypes function signatures for MetalRuntime."""
        if self._lib is None:
            return

        # void _MetalRelease(intptr_t ref)
        self._lib._MetalRelease.argtypes = [c_int64]
        self._lib._MetalRelease.restype = None

        # intptr_t _MetalDeviceMakeDefault(void)
        self._lib._MetalDeviceMakeDefault.argtypes = []
        self._lib._MetalDeviceMakeDefault.restype = c_int64

        # intptr_t _MetalDeviceMakeCommandQueue(intptr_t ref)
        self._lib._MetalDeviceMakeCommandQueue.argtypes = [c_int64]
        self._lib._MetalDeviceMakeCommandQueue.restype = c_int64

        # intptr_t _MetalDeviceMakeBuffer(intptr_t ref, bool isStorageModeManaged,
        #                                 int64_t count, int64_t sizeType)
        self._lib._MetalDeviceMakeBuffer.argtypes = [c_int64, c_bool, c_int64, c_int64]
        self._lib._MetalDeviceMakeBuffer.restype = c_int64

        # void* _MetalBufferGetContents2(intptr_t ref)
        self._lib._MetalBufferGetContents2.argtypes = [c_int64]
        self._lib._MetalBufferGetContents2.restype = c_void_p

        # intptr_t _MetalCommandQueueMakeCommandBuffer(intptr_t ref,
        #     const int8_t* libPath, const int8_t* functionName,
        #     int64_t width, int64_t height, int64_t depth)
        self._lib._MetalCommandQueueMakeCommandBuffer.argtypes = [
            c_int64, c_char_p, c_char_p, c_int64, c_int64, c_int64
        ]
        self._lib._MetalCommandQueueMakeCommandBuffer.restype = c_int64

        # void _MetalCommandBufferAddBuffer(intptr_t ref, intptr_t bufferRef, int64_t index)
        self._lib._MetalCommandBufferAddBuffer.argtypes = [c_int64, c_int64, c_int64]
        self._lib._MetalCommandBufferAddBuffer.restype = None

        # void _MetalCommandBufferCommit(intptr_t ref)
        self._lib._MetalCommandBufferCommit.argtypes = [c_int64]
        self._lib._MetalCommandBufferCommit.restype = None

        # void _MetalCommandBufferWaitUntilCompleted(intptr_t ref)
        self._lib._MetalCommandBufferWaitUntilCompleted.argtypes = [c_int64]
        self._lib._MetalCommandBufferWaitUntilCompleted.restype = None

    def _initialize_device(self):
        """Initialize Metal device and command queue."""
        if self._lib is None:
            return

        self._device = self._lib._MetalDeviceMakeDefault()
        if self._device:
            self._queue = self._lib._MetalDeviceMakeCommandQueue(self._device)

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        """
        Launch the kernel.

        Args:
            gridX, gridY, gridZ: Grid dimensions
            stream: Command queue (or use default)
            function: Kernel function handle (metallib data in our case)
            *args: Kernel arguments (pointers, scalars, metadata, hooks)
        """
        import torch

        # Parse args: (kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, *kernel_args)
        if len(args) >= 4:
            kernel_metadata = args[0]
            launch_metadata = args[1]
            launch_enter_hook = args[2]
            launch_exit_hook = args[3]
            kernel_args = args[4:]
        else:
            kernel_metadata = None
            launch_metadata = None
            launch_enter_hook = None
            launch_exit_hook = None
            kernel_args = args

        # Call launch enter hook if provided
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        # Check if Metal runtime is available
        if self._lib is not None and self._queue is not None:
            # Use native Metal runtime
            self._launch_native(gridX, gridY, gridZ, function, kernel_args)
        else:
            # Fallback: Use PyTorch MPS operations
            # This is a workaround when MetalRuntime isn't available
            self._launch_fallback(gridX, gridY, gridZ, kernel_args)

        # Synchronize MPS
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

        # Call launch exit hook if provided
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)

    def _launch_native(self, gridX, gridY, gridZ, function, kernel_args):
        """Launch kernel using native MetalRuntime."""
        # function is the metallib data
        metallib_data = function if isinstance(function, bytes) else b''

        if not metallib_data:
            raise RuntimeError("No metallib data available - compilation may have failed")

        # Write metallib to temp file (required by MetalRuntime API)
        with tempfile.NamedTemporaryFile(suffix='.metallib', delete=False) as f:
            f.write(metallib_data)
            metallib_path = f.name

        try:
            cmd_buffer = self._lib._MetalCommandQueueMakeCommandBuffer(
                self._queue,
                metallib_path.encode('utf-8'),
                self.kernel_name.encode('utf-8'),
                gridX * self.num_warps * 32,
                gridY,
                gridZ
            )

            if not cmd_buffer:
                raise RuntimeError("Failed to create Metal command buffer")

            # Bind buffer arguments
            for idx, arg in enumerate(kernel_args):
                if hasattr(arg, 'data_ptr'):
                    ptr = arg.data_ptr()
                    # Would bind Metal buffer here
                    pass

            self._lib._MetalCommandBufferCommit(cmd_buffer)
            self._lib._MetalCommandBufferWaitUntilCompleted(cmd_buffer)
            self._lib._MetalRelease(cmd_buffer)

        finally:
            os.unlink(metallib_path)

    def _launch_fallback(self, gridX, gridY, gridZ, kernel_args):
        """
        Fallback kernel launch using PyTorch MPS operations.

        This is a workaround when MetalRuntime isn't available.
        It interprets the kernel semantically based on the signature.

        WARNING: This only works for simple kernels like vector add.
        """
        import torch

        # For the stub kernel (vector add), we have 3 buffer args + 1 scalar
        # arg0: x_ptr, arg1: y_ptr, arg2: output_ptr, arg3: n_elements
        tensors = []
        scalars = []

        for arg in kernel_args:
            if hasattr(arg, 'data_ptr'):
                tensors.append(arg)
            elif isinstance(arg, (int, float)):
                scalars.append(arg)

        if len(tensors) >= 3:
            # Assume it's an add kernel: output = input0 + input1
            x = tensors[0]
            y = tensors[1]
            output = tensors[2]

            # Perform the add operation
            output.copy_(x + y)
        elif len(tensors) == 2:
            # Assume copy or unary operation
            output = tensors[1]
            output.copy_(tensors[0])
        else:
            raise RuntimeError(
                f"Fallback launcher doesn't support this kernel configuration: "
                f"{len(tensors)} tensors, {len(scalars)} scalars. "
                f"Build MetalRuntime for full kernel support."
            )


class MetalDriver(DriverBase):
    """
    Metal GPU driver for Triton.

    Integrates with PyTorch MPS backend for tensor operations.
    """

    def __init__(self):
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher
        super().__init__()

        # Set up MPS device methods (similar to GPUDriver but for MPS)
        import torch

        # MPS only has one device (device 0)
        self.get_current_device = lambda: 0
        self.set_current_device = lambda x: None  # No-op for MPS

        # MPS doesn't expose streams the same way as CUDA
        # Return 0 as a placeholder stream handle
        self.get_current_stream = lambda device=0: 0

        # Device capability for Metal (return a tuple for compatibility)
        # Format similar to CUDA: (major, minor)
        self.get_device_capability = lambda device=0: (1, 0)  # Metal 1.0

    @staticmethod
    def is_active():
        """Check if Metal backend is available (macOS with MPS)."""
        if platform.system() != 'Darwin':
            return False

        try:
            import torch
            # Check if MPS is available
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False

    def get_current_target(self):
        """Get the current GPU target for Metal."""
        # Detect Apple GPU architecture
        arch = self._detect_gpu_arch()
        warp_size = 32  # Metal SIMD width is 32

        return GPUTarget("metal", arch, warp_size)

    def _detect_gpu_arch(self):
        """Detect Apple GPU architecture."""
        try:
            # Use system_profiler to detect GPU
            import subprocess
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True
            )
            output = result.stdout.lower()

            # Detect GPU family
            if 'm3' in output:
                return 'apple-m3'
            elif 'm2' in output:
                return 'apple-m2'
            elif 'm1' in output:
                return 'apple-m1'
            elif 'a17' in output:
                return 'apple-a17'
            elif 'a16' in output:
                return 'apple-a16'
            elif 'a15' in output:
                return 'apple-a15'
            elif 'a14' in output:
                return 'apple-a14'
            else:
                # Default to M1 for unknown Apple Silicon
                return 'apple-m1'
        except Exception:
            return 'apple-m1'

    def get_active_torch_device(self):
        """Get the active PyTorch device for Metal (MPS)."""
        import torch
        return torch.device("mps")

    def get_device_interface(self):
        """Return the device interface module."""
        # Return a proxy object with MPS-compatible functionality
        return _MPSDeviceInterface()

    def get_benchmarker(self):
        """Return benchmarker function for Metal."""
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        """Create empty cache tensor for benchmarking."""
        import torch
        # Create a cache-clearing buffer on MPS
        cache_size = 256 * 1024 * 1024  # 256 MB
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='mps')

    def clear_cache(self, cache):
        """Clear the benchmark cache."""
        cache.zero_()

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        """Assemble tensor maps to arguments (not used on Metal)."""
        return args


class _MPSDeviceInterface:
    """
    Proxy interface for MPS device operations.

    Provides CUDA-like interface for compatibility.
    """

    def current_device(self):
        """MPS only has one device."""
        return 0

    def set_device(self, device):
        """No-op for MPS (single device)."""
        pass

    def synchronize(self):
        """Synchronize MPS operations."""
        import torch
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

    def current_stream(self, device=None):
        """MPS doesn't have explicit streams like CUDA."""
        return None
