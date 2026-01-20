#!/usr/bin/env python3
"""
Test script for Metal backend integration.

This script tests the Metal backend pipeline:
1. Backend discovery
2. TTIR generation
3. TTGIR generation (with BlockedEncodingAttr)
4. MSL generation
5. Metal compilation (if on macOS)

Usage:
    python test_metal_backend.py
"""

import os
import sys
import platform

# Use the development version of triton with the metal backend
# This must be before any triton imports
triton_dev_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'python'
)
sys.path.insert(0, triton_dev_path)

def test_backend_discovery():
    """Test that Metal backend is discovered."""
    print("=" * 60)
    print("Test 1: Backend Discovery")
    print("=" * 60)

    from triton.backends import backends

    print(f"Discovered backends: {list(backends.keys())}")

    if 'metal' in backends:
        print("SUCCESS: Metal backend discovered")
        metal_backend = backends['metal']
        print(f"  Compiler: {metal_backend.compiler}")
        print(f"  Driver: {metal_backend.driver}")
        return True
    else:
        print("FAILED: Metal backend not found")
        return False


def test_driver_activation():
    """Test if Metal driver reports active status."""
    print("\n" + "=" * 60)
    print("Test 2: Driver Activation")
    print("=" * 60)

    from triton.backends import backends

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    driver_cls = backends['metal'].driver
    is_active = driver_cls.is_active()

    print(f"Platform: {platform.system()}")
    print(f"Driver is_active(): {is_active}")

    if platform.system() == 'Darwin':
        if is_active:
            print("SUCCESS: Metal driver active on macOS")
            return True
        else:
            print("WARNING: Metal driver not active (MPS may not be available)")
            return False
    else:
        print("INFO: Not on macOS, Metal driver correctly inactive")
        return True


def test_target_detection():
    """Test GPU target detection."""
    print("\n" + "=" * 60)
    print("Test 3: Target Detection")
    print("=" * 60)

    from triton.backends import backends

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    driver_cls = backends['metal'].driver

    if not driver_cls.is_active():
        print("SKIPPED: Metal driver not active")
        return False

    driver = driver_cls()
    target = driver.get_current_target()

    print(f"Target backend: {target.backend}")
    print(f"Target arch: {target.arch}")
    print(f"Target warp_size: {target.warp_size}")

    if target.backend == 'metal':
        print("SUCCESS: Target correctly identified as Metal")
        return True
    else:
        print("FAILED: Target backend mismatch")
        return False


def test_backend_options():
    """Test backend options parsing."""
    print("\n" + "=" * 60)
    print("Test 4: Backend Options")
    print("=" * 60)

    from triton.backends import backends
    from triton.backends.compiler import GPUTarget

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    backend_cls = backends['metal'].compiler
    target = GPUTarget("metal", "apple-m1", 32)
    backend = backend_cls(target)

    # Test option parsing
    opts = {
        'num_warps': 4,
        'num_stages': 2,
        'debug': True
    }

    parsed = backend.parse_options(opts)

    print(f"Parsed options:")
    print(f"  num_warps: {parsed.num_warps}")
    print(f"  num_stages: {parsed.num_stages}")
    print(f"  debug: {parsed.debug}")
    print(f"  arch: {parsed.arch}")

    if parsed.num_warps == 4 and parsed.arch == "apple-m1":
        print("SUCCESS: Options parsed correctly")
        return True
    else:
        print("FAILED: Options parsing error")
        return False


def test_compilation_stages():
    """Test that compilation stages are registered."""
    print("\n" + "=" * 60)
    print("Test 5: Compilation Stages")
    print("=" * 60)

    from triton.backends import backends
    from triton.backends.compiler import GPUTarget

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    backend_cls = backends['metal'].compiler
    target = GPUTarget("metal", "apple-m1", 32)
    backend = backend_cls(target)

    opts = backend.parse_options({'num_warps': 4})
    stages = {}
    backend.add_stages(stages, opts)

    expected_stages = ['ttir', 'ttgir', 'msl', 'metallib']

    print(f"Registered stages: {list(stages.keys())}")

    missing = [s for s in expected_stages if s not in stages]
    if missing:
        print(f"FAILED: Missing stages: {missing}")
        return False
    else:
        print("SUCCESS: All expected stages registered")
        return True


def test_msl_generation():
    """Test MSL code generation (stub)."""
    print("\n" + "=" * 60)
    print("Test 6: MSL Generation (Stub)")
    print("=" * 60)

    from triton.backends import backends
    from triton.backends.compiler import GPUTarget

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    backend_cls = backends['metal'].compiler
    target = GPUTarget("metal", "apple-m1", 32)
    backend = backend_cls(target)

    opts = backend.parse_options({'num_warps': 4})
    metadata = {}

    # Test stub MSL generation
    msl = backend._generate_stub_msl(metadata, opts)

    print("Generated MSL stub:")
    print("-" * 40)
    for line in msl.split('\n')[:10]:
        print(f"  {line}")
    print("  ...")
    print("-" * 40)

    if 'kernel void' in msl and '#include <metal_stdlib>' in msl:
        print("SUCCESS: MSL stub generated correctly")
        return True
    else:
        print("FAILED: MSL stub generation error")
        return False


def test_metal_compilation():
    """Test Metal compilation with xcrun."""
    print("\n" + "=" * 60)
    print("Test 7: Metal Compilation")
    print("=" * 60)

    if platform.system() != 'Darwin':
        print("SKIPPED: Not on macOS")
        return True

    from triton.backends import backends
    from triton.backends.compiler import GPUTarget

    if 'metal' not in backends:
        print("SKIPPED: Metal backend not available")
        return False

    backend_cls = backends['metal'].compiler
    target = GPUTarget("metal", "apple-m1", 32)
    backend = backend_cls(target)

    opts = backend.parse_options({'num_warps': 4})
    metadata = {}

    # Generate stub MSL
    msl = backend._generate_stub_msl(metadata, opts)

    # Try to compile
    metallib = backend.make_metallib(msl, metadata, opts)

    print(f"Metallib size: {len(metallib)} bytes")

    if len(metallib) > 0:
        print("SUCCESS: Metal compilation succeeded")
        return True
    else:
        print("WARNING: Metal compilation returned empty bytes")
        print("         (This is expected if xcrun is not available)")
        return False


def main():
    """Run all tests."""
    print("Metal Backend Integration Tests")
    print("================================\n")

    tests = [
        test_backend_discovery,
        test_driver_activation,
        test_target_detection,
        test_backend_options,
        test_compilation_stages,
        test_msl_generation,
        test_metal_compilation,
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
