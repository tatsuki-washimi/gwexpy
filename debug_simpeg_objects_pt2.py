
try:
    import simpeg
    from simpeg.electromagnetics import time_domain as tdem
    import inspect
    
    Src = tdem.sources.MagDipole
    print(f"TDEM Src signature: {inspect.signature(Src)}")
    
    # Try creating with receiver_list
    Rx = tdem.receivers.PointElectricField
    times = [0.1, 0.2]
    rx = Rx(locations=[[0,0,0]], times=times, orientation='x')
    
    src = Src(receiver_list=[rx], location=[0,0,0])
    print(f"TDEM Src nD (with receiver_list): {src.nD}")
    
except ImportError:
    print("SimPEG not installed")
