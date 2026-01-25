import time
import tracemalloc

import numpy as np
from astropy import units as u

from gwexpy.fields import ScalarField


def profile_block(name, func, *args, **kwargs):
    """Run a function with profiling (time & memory)."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = end_time - start_time
    peak_mb = peak / 1024**2

    print(f"[{name}]")
    print(f"  Time : {elapsed:.4f} sec")
    print(f"  Peak Memory: {peak_mb:.2f} MB")
    return result


def benchmark_scalar_field():
    print("=== Benchmarking ScalarField ===")

    # 1. Initialization
    # Shape: (Time=1000, X=30, Y=30, Z=30) -> 27,000,000 elements -> ~216 MB (float64)
    # Reducing size slightly to be safe for quick test, aiming for ~100MB
    # Time=1000, X=25, Y=25, Z=20 -> 12.5M elements -> 100MB
    shape = (1000, 25, 25, 20)
    data = np.random.randn(*shape)

    print(f"Data Shape: {shape}")
    print(f"Data Size (approx): {data.nbytes / 1024**2:.2f} MB")

    # Create axis indices manually
    t_index = (np.arange(shape[0]) * 0.01) * u.s
    x_index = (np.arange(shape[1]) * 1.0) * u.m
    y_index = (np.arange(shape[2]) * 1.0) * u.m
    z_index = (np.arange(shape[3]) * 1.0) * u.m

    def create_field():
        return ScalarField(
            data, axis0=t_index, axis1=x_index, axis2=y_index, axis3=z_index, unit=u.V
        )

    field = profile_block("Initialization", create_field)

    # 2. Arithmetic (Multiplication)
    def op_mul():
        return field * 2.0

    _ = profile_block("Multiplication (Scalar)", op_mul)

    # 3. Time FFT
    # This involves iterating over spatial dimensions or using efficient numpy broadcasting
    def op_fft_time():
        return field.fft_time()

    fft_field = profile_block("FFT (Time axis=0)", op_fft_time)

    # 4. Space FFT
    # axis 1, 2, 3
    def op_fft_space():
        return field.fft_space()  # transforms all spatial axes

    _ = profile_block("FFT (Space axes=1,2,3)", op_fft_space)

    # 4b. Space FFT (Overwrite=True)
    def op_fft_space_overwrite():
        return field.fft_space(overwrite=True)

    fft_field = profile_block("FFT (Space, Overwrite=True)", op_fft_space_overwrite)

    # 4c. Space IFFT (Default)
    def op_ifft_space():
        return fft_field.ifft_space()

    _ = profile_block("IFFT (Space axes=1,2,3)", op_ifft_space)

    # 4d. Space IFFT (Overwrite=True)
    def op_ifft_space_overwrite():
        return fft_field.ifft_space(overwrite=True)

    _ = profile_block("IFFT (Space, Overwrite=True)", op_ifft_space_overwrite)

    # 5. Slicing (Preserving 4D)
    def op_slice():
        # Taking a chunk in time
        return field[100:200, 5:15, 5:15, :]

    _ = profile_block("Slicing (Sub-volume)", op_slice)

    # 6. Extract Points (Interpolation)
    def op_extract():
        # Extract at 10 random points
        # points must be list of tuples of Quantities
        pts = []
        for _ in range(10):
            pts.append(
                (
                    np.random.uniform(0, 20) * u.m,
                    np.random.uniform(0, 20) * u.m,
                    np.random.uniform(0, 15) * u.m,
                )
            )
        return field.extract_points(pts, interp="nearest")

    _ = profile_block("Extract Points (Linear Interp)", op_extract)


if __name__ == "__main__":
    benchmark_scalar_field()
