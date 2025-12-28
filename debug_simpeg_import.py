
try:
    import simpeg
    print(f"SimPEG version: {simpeg.__version__}")
    print(f"Dir simpeg: {dir(simpeg)}")
    
    try:
        from simpeg import surveys
        print("Success: from simpeg import surveys")
    except ImportError as e:
        print(f"Fail: {e}")
        
    try:
        from simpeg import survey
        print("Success: from simpeg import survey")
    except ImportError as e:
        print(f"Fail: {e}")
        
except ImportError:
    print("SimPEG not installed")
