import numpy as np
from gwexpy.timeseries import TimeSeriesMatrix, TimeSeries
from astropy import units as u
from gwexpy.types.metadata import MetaData
import time
import sys

def verify_method(method_name, *args, **kwargs):
    print(f"--- Verifying {method_name} ---")
    
    # Create test data
    N, M, T = 2, 2, 1024
    data = np.random.randn(N, M, T)
    dt = 0.001
    tm = TimeSeriesMatrix(data, dt=dt, name="test")
    
    # Initialize metadata
    for i in range(N):
        for j in range(M):
            tm.meta[i, j] = MetaData(unit=u.dimensionless_unscaled, name=f"chan_{i}_{j}")

    vectorized_name = f"_vectorized_{method_name}"
    
    # Find which class in MRO has the vectorized method
    target_cls = None
    for cls in tm.__class__.mro():
        if vectorized_name in cls.__dict__:
            target_cls = cls
            break
            
    if not target_cls:
        # Check if it exists at all
        if hasattr(tm, vectorized_name):
             print(f"Error: {vectorized_name} exists on instance/MRO but not in any __dict__??")
             # Fallback: just delete from the class of the object
             target_cls = tm.__class__
        else:
             print(f"Error: {vectorized_name} not found!")
             return False
        
    orig_vec_impl = getattr(target_cls, vectorized_name)
    
    # 1. Get Loop Result
    # Temporarily hide the vectorized method to force loop
    try:
        delattr(target_cls, vectorized_name)
        print(f"  (Forcing loop by removing {vectorized_name} from {target_cls.__name__})")
        
        start_t = time.time()
        loop_res = getattr(tm, method_name)(*args, **kwargs)
        loop_time = time.time() - start_t
    except Exception as e:
        print(f"  Loop execution failed: {e}")
        import traceback
        traceback.print_exc()
        setattr(target_cls, vectorized_name, orig_vec_impl)
        return False
    finally:
        # Restore as soon as possible
        setattr(target_cls, vectorized_name, orig_vec_impl)
    
    # 2. Get Vectorized Result
    try:
        print(f"  (Running vectorized {method_name})")
        start_t = time.time()
        vec_res = getattr(tm, method_name)(*args, **kwargs)
        vec_time = time.time() - start_t
    except Exception as e:
        print(f"  Vectorized execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Compare Data
    try:
        l_arr = np.asarray(loop_res)
        v_arr = np.asarray(vec_res)
        
        if np.iscomplexobj(l_arr):
             data_match = np.allclose(l_arr.real, v_arr.real, atol=1e-11) and \
                          np.allclose(l_arr.imag, v_arr.imag, atol=1e-11)
        else:
             data_match = np.allclose(l_arr, v_arr, atol=1e-11)
             
        print(f"  Data Match: {data_match}")
        print(f"  Loop Time: {loop_time:.4f}s")
        print(f"  Vec Time:  {vec_time:.4f}s")
        print(f"  Speedup:   {loop_time/max(vec_time, 1e-9):.1f}x")
        
        unit_match = True
        if hasattr(loop_res, 'unit') and hasattr(vec_res, 'unit'):
             unit_match = (loop_res.unit == vec_res.unit)
             print(f"  Unit Match: {unit_match} (Loop: {loop_res.unit}, Vec: {vec_res.unit})")
        
        return data_match and unit_match
    except Exception as e:
        print(f"  Comparison failed: {e}")
        return False

if __name__ == "__main__":
    results = {}
    
    try:
        # Signal Processing
        results['detrend'] = verify_method('detrend', detrend='linear')
        results['taper'] = verify_method('taper', side='leftright')
        
        # Spectral
        results['fft'] = verify_method('fft')
        results['psd'] = verify_method('psd', fftlength=0.1)
        results['asd'] = verify_method('asd', fftlength=0.1)
        
        # Bivariate Spectral
        data2 = np.random.randn(2, 2, 1024)
        tm2 = TimeSeriesMatrix(data2, dt=0.001)
        results['csd'] = verify_method('csd', tm2, fftlength=0.1)
        results['coherence'] = verify_method('coherence', tm2, fftlength=0.1)
        
        # Transforms
        results['hilbert'] = verify_method('hilbert')
        results['radian'] = verify_method('radian')
        results['degree'] = verify_method('degree')

        print("\n" + "="*20)
        print("FINAL RESULTS")
        print("="*20)
        all_pass = True
        for m, res in results.items():
            print(f"{m:20}: {'PASS' if res else 'FAIL'}")
            if not res: all_pass = False
            
        if all_pass:
            print("\nAll vectorized methods verified successfully!")
        else:
            print("\nSome methods failed verification. Check logs.")
            sys.exit(1)
    except Exception as e:
        print(f"Fatal error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
