
import numpy as np
import gwpy.frequencyseries._fdcommon as fd
try:
    # Test with typical filter (e.g. zpk)
    filt = ([1], [1], 1)
    parsed = fd.parse_filter(filt)
    print(f"Type: {type(parsed)}")
    print(f"Value: {parsed}")
    if hasattr(parsed, 'freqresp'):
        print("Has freqresp method")
    
    # Test 2 args (b, a)
    # filt2 = ([1], [1])
    # parsed2 = fd.parse_filter(filt2)
    # print(f"Type 2: {type(parsed2)}")

except Exception as e:
    print(f"Error: {e}")

try:
    # Check if parse_filter handles analog keyword
    # It might be in kwargs or args
    pass
except:
    pass
