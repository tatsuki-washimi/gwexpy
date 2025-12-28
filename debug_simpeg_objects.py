
try:
    import simpeg
    from simpeg.electromagnetics import time_domain as tdem
    from simpeg.electromagnetics import frequency_domain as fdem
    import numpy as np
    
    # 1. Inspect TDEM Rx
    try:
        Rx = tdem.receivers.PointElectricField
        print(f"TDEM Rx init: {Rx.__init__}")
        
        times = np.linspace(0, 1, 11)
        rx = Rx(locations=np.array([[0., 0., 0.]]), times=times, orientation='x')
        print(f"TDEM Rx nD: {rx.nD}")
        
        src = tdem.sources.MagDipole(receivers=[rx], location=np.array([0., 0., 0.]))
        print(f"TDEM Src nD: {src.nD}")
        
        survey = tdem.Survey(source_list=[src])
        print(f"TDEM Survey nD: {survey.nD}")
        
    except Exception as e:
        print(f"TDEM Fail: {e}")

    # 2. Inspect FDEM Source
    try:
        Src = fdem.sources.MagDipole
        print(f"FDEM Src init: {Src.__init__}")
        # inspect signature
        import inspect
        print(f"FDEM Src signature: {inspect.signature(Src)}")
        
        Rx = fdem.receivers.PointMagneticFluxDensity
        rx = Rx(locations=np.array([[0,0,0]]), orientation='z')
        
        # Try creating src with frequency keyword
        try:
           src = Src(receivers=[rx], location=np.array([0,0,0]), frequency=100.0)
           print("FDEM Src created (kwarg frequency)")
        except TypeError as e:
           print(f"FDEM Src kwarg fail: {e}")
           # Try positional?
           try:
              src = Src([rx], 100.0, np.array([0,0,0]))
              print("FDEM Src created (positional)")
           except Exception as e2:
              print(f"FDEM Src positional fail: {e2}")

    except Exception as e:
        print(f"FDEM Fail: {e}")

except ImportError:
    print("SimPEG not installed")
