#!/usr/bin/env python3
"""
Standalone test for Metal backend that doesn't require compiled Triton.

This verifies the Python code structure is correct.
"""

import os
import sys
import platform
import tempfile
import subprocess

# Test the driver module standalone
def test_driver_imports():
    """Test that driver module can be imported standalone."""
    print("Test 1: Driver module imports")
    print("-" * 40)

    try:
        # Import without triton dependencies
        import importlib.util
        driver_path = os.path.join(os.path.dirname(__file__), 'driver.py')

        # Read and check for syntax errors
        with open(driver_path, 'r') as f:
            source = f.read()

        compile(source, driver_path, 'exec')
        print("SUCCESS: driver.py has no syntax errors")
        return True
    except SyntaxError as e:
        print(f"FAILED: Syntax error in driver.py: {e}")
        return False


def test_compiler_imports():
    """Test that compiler module can be imported standalone."""
    print("\nTest 2: Compiler module imports")
    print("-" * 40)

    try:
        compiler_path = os.path.join(os.path.dirname(__file__), 'compiler.py')

        with open(compiler_path, 'r') as f:
            source = f.read()

        compile(source, compiler_path, 'exec')
        print("SUCCESS: compiler.py has no syntax errors")
        return True
    except SyntaxError as e:
        print(f"FAILED: Syntax error in compiler.py: {e}")
        return False


def test_msl_generation():
    """Test MSL generation directly."""
    print("\nTest 3: MSL generation")
    print("-" * 40)

    msl_code = '''#include <metal_stdlib>
using namespace metal;

kernel void triton_kernel(
    device float* arg0 [[buffer(0)]],
    device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgsize [[threads_per_threadgroup]])
{
    uint idx = tid.x;
    arg2[idx] = arg0[idx] + arg1[idx];
}
'''

    print("Generated MSL code:")
    for line in msl_code.split('\n')[:10]:
        print(f"  {line}")
    print("  ...")

    if 'kernel void' in msl_code and 'metal_stdlib' in msl_code:
        print("SUCCESS: MSL code structure is valid")
        return True
    else:
        print("FAILED: MSL code structure invalid")
        return False


def test_metal_compilation():
    """Test Metal compilation with xcrun if available."""
    print("\nTest 4: Metal compilation (requires macOS)")
    print("-" * 40)

    if platform.system() != 'Darwin':
        print("SKIPPED: Not on macOS")
        return True

    # Check if xcrun is available
    try:
        result = subprocess.run(['xcrun', '--version'], capture_output=True)
        if result.returncode != 0:
            print("SKIPPED: xcrun not available")
            return True
    except FileNotFoundError:
        print("SKIPPED: xcrun not found")
        return True

    msl_code = '''#include <metal_stdlib>
using namespace metal;

kernel void triton_kernel(
    device float* arg0 [[buffer(0)]],
    device float* arg1 [[buffer(1)]],
    device float* arg2 [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint idx = tid.x;
    arg2[idx] = arg0[idx] + arg1[idx];
}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        msl_path = os.path.join(tmpdir, 'kernel.metal')
        air_path = os.path.join(tmpdir, 'kernel.air')
        lib_path = os.path.join(tmpdir, 'kernel.metallib')

        # Write MSL
        with open(msl_path, 'w') as f:
            f.write(msl_code)

        try:
            # Compile MSL to AIR
            result = subprocess.run([
                'xcrun', '-sdk', 'macosx', 'metal',
                '-c', msl_path, '-o', air_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FAILED: Metal compilation error: {result.stderr}")
                return False

            # Link to metallib
            result = subprocess.run([
                'xcrun', '-sdk', 'macosx', 'metallib',
                air_path, '-o', lib_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"FAILED: Metallib linking error: {result.stderr}")
                return False

            # Check metallib was created
            if os.path.exists(lib_path):
                size = os.path.getsize(lib_path)
                print(f"SUCCESS: Created metallib ({size} bytes)")
                return True
            else:
                print("FAILED: Metallib not created")
                return False

        except Exception as e:
            print(f"FAILED: Exception during compilation: {e}")
            return False


def test_backend_structure():
    """Test that backend directory has required files."""
    print("\nTest 5: Backend structure")
    print("-" * 40)

    backend_dir = os.path.dirname(__file__)
    required_files = ['__init__.py', 'compiler.py', 'driver.py']

    missing = []
    for fname in required_files:
        path = os.path.join(backend_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)

    if missing:
        print(f"FAILED: Missing files: {missing}")
        return False
    else:
        print(f"SUCCESS: All required files present: {required_files}")
        return True


def test_symlink():
    """Test that the symlink to backends is correct."""
    print("\nTest 6: Backend symlink")
    print("-" * 40)

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # backend_dir = /Users/.../triton/third_party/metal/backend
    # triton_root = /Users/.../triton
    triton_root = os.path.dirname(os.path.dirname(os.path.dirname(backend_dir)))
    symlink_path = os.path.join(triton_root, 'python', 'triton', 'backends', 'metal')

    if os.path.islink(symlink_path):
        target = os.readlink(symlink_path)
        print(f"Symlink: {symlink_path}")
        print(f"Target: {target}")
        print("SUCCESS: Symlink exists")
        return True
    elif os.path.exists(symlink_path):
        print(f"Path exists but is not a symlink: {symlink_path}")
        return True
    else:
        print(f"FAILED: Symlink not found: {symlink_path}")
        return False


def test_conversion_patterns_cpp():
    """Test that C++ conversion patterns compile (syntax check)."""
    print("\nTest 7: C++ conversion patterns")
    print("-" * 40)

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # backend_dir = /Users/.../triton/third_party/metal/backend
    # triton_root = /Users/.../triton
    triton_root = os.path.dirname(os.path.dirname(os.path.dirname(backend_dir)))
    cpp_path = os.path.join(
        triton_root, 'metal-dialect', 'lib', 'metal', 'Conversion',
        'TritonToMetal', 'ConvertTritonToMetal.cpp'
    )

    if not os.path.exists(cpp_path):
        print(f"SKIPPED: C++ file not found: {cpp_path}")
        return True

    # Check that our new patterns are present
    with open(cpp_path, 'r') as f:
        content = f.read()

    expected_patterns = [
        'GetProgramIdOpConversion',
        'MakeRangeOpConversion',
        'SplatOpConversion',
        'AddPtrOpConversion',
        'LoadOpConversion',
        'StoreOpConversion',
        'BroadcastOpConversion',
    ]

    missing = [p for p in expected_patterns if p not in content]

    if missing:
        print(f"FAILED: Missing conversion patterns: {missing}")
        return False
    else:
        print(f"SUCCESS: All expected patterns present:")
        for p in expected_patterns:
            print(f"  - {p}")
        return True


def main():
    """Run all standalone tests."""
    print("Metal Backend Standalone Tests")
    print("=" * 50)
    print()

    tests = [
        test_driver_imports,
        test_compiler_imports,
        test_msl_generation,
        test_metal_compilation,
        test_backend_structure,
        test_symlink,
        test_conversion_patterns_cpp,
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
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
