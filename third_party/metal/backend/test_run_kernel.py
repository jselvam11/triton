#!/usr/bin/env python3
"""
Test running an actual Triton kernel on MPS.
"""

import os
import sys

# Use the development version of triton with the metal backend
triton_dev_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'python'
)
sys.path.insert(0, triton_dev_path)

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple vector addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_add_kernel_mps():
    """Test vector addition on MPS."""
    print("=" * 60)
    print("Test: Vector Addition on MPS")
    print("=" * 60)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return False

    # Create tensors
    n_elements = 1024
    x = torch.randn(n_elements, device='mps', dtype=torch.float32)
    y = torch.randn(n_elements, device='mps', dtype=torch.float32)
    output = torch.empty_like(x)

    print(f"Input tensors created on MPS:")
    print(f"  x: {x.shape}, device={x.device}")
    print(f"  y: {y.shape}, device={y.device}")

    # Compute expected result
    expected = x + y

    # Grid configuration
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    print(f"\nLaunch configuration:")
    print(f"  n_elements: {n_elements}")
    print(f"  BLOCK_SIZE: {BLOCK_SIZE}")
    print(f"  grid: {triton.cdiv(n_elements, BLOCK_SIZE)}")

    try:
        # Try to launch the kernel
        print("\nAttempting kernel launch...")
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        # Synchronize
        torch.mps.synchronize()

        # Verify result
        if torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
            print("SUCCESS: Output matches expected result!")
            return True
        else:
            print("FAILED: Output does not match expected")
            print(f"  Max diff: {(output - expected).abs().max().item()}")
            return False

    except Exception as e:
        print(f"ERROR during kernel launch: {e}")
        import traceback
        traceback.print_exc()

        # Try to get more details about the compilation
        print("\n" + "-" * 40)
        print("Attempting to debug compilation...")
        try:
            # Get the Metal backend
            from triton.backends import backends
            if 'metal' in backends:
                print("Metal backend is available")
                backend_cls = backends['metal'].compiler
                driver_cls = backends['metal'].driver

                print(f"Backend class: {backend_cls}")
                print(f"Driver class: {driver_cls}")
                print(f"Driver is_active: {driver_cls.is_active()}")

                if driver_cls.is_active():
                    driver = driver_cls()
                    target = driver.get_current_target()
                    print(f"Target: {target}")
        except Exception as e2:
            print(f"Debug error: {e2}")

        return False


def test_manual_compilation():
    """Test manual compilation of the kernel."""
    print("\n" + "=" * 60)
    print("Test: Manual Kernel Compilation")
    print("=" * 60)

    from triton.backends import backends
    from triton.backends.compiler import GPUTarget

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    backend_cls = backends['metal'].compiler
    target = GPUTarget("metal", "apple-m1", 32)
    backend = backend_cls(target)

    print(f"Backend: {backend}")
    print(f"Target: {target}")

    # Get options and stages
    opts = backend.parse_options({
        'num_warps': 4,
        'num_stages': 2,
    })

    stages = {}
    backend.add_stages(stages, opts)

    print(f"Stages: {list(stages.keys())}")

    # Generate stub MSL
    metadata = {}
    msl = backend._generate_stub_msl(metadata, opts)

    print(f"\nGenerated MSL ({len(msl)} chars):")
    print("-" * 40)
    print(msl[:500])
    print("-" * 40)

    # Compile to metallib
    metallib = backend.make_metallib(msl, metadata, opts)

    if metallib:
        print(f"\nSUCCESS: Compiled to metallib ({len(metallib)} bytes)")
        return True
    else:
        print("\nFAILED: Could not compile metallib")
        return False


def test_kernel_signature():
    """Test getting kernel signature info."""
    print("\n" + "=" * 60)
    print("Test: Kernel Signature Analysis")
    print("=" * 60)

    # Analyze the kernel
    print(f"Kernel function: {add_kernel.fn}")
    print(f"Kernel name: {add_kernel.fn.__name__}")

    # Get source
    import inspect
    source = inspect.getsource(add_kernel.fn)
    print(f"\nKernel source:")
    print("-" * 40)
    for line in source.split('\n'):
        print(f"  {line}")
    print("-" * 40)

    return True


def main():
    """Run all tests."""
    print("Metal Backend Kernel Execution Tests")
    print("=" * 60)
    print()

    # Check environment
    print("Environment:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    print()

    tests = [
        test_kernel_signature,
        test_manual_compilation,
        test_add_kernel_mps,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
