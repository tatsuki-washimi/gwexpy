from pathlib import Path
import xml.etree.ElementTree as ET
from gwexpy.timeseries import TimeSeriesDict, TimeSeries

def extract_xml_channels(filename: str) -> list:
    """
    Parse DTT XML to extract channel names and their Active status.
    Returns: list of dict {'name': str, 'active': bool}
    """
    channels = []
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # DTT XML typically stores parameters in <Param Name="MeasChn[i]" ...> and <Param Name="MeasActive[i]" ...>
        # or similar structure within <LIGO_LW Name="TestParameters">
        
        # We need to find the definition of channels. 
        # Structure is usually flattened arrays in Params or Columns in Table.
        # But DTT 'restore' logic reads Params.
        
        # Let's search for flattened params first.
        # In DTT XML, keys are like "MeasChn[0]", "MeasActive[0]" etc.
        
        params = {}
        for param in root.findall(".//Param"):
            name_attr = param.get('Name')
            if name_attr:
                # Value is text content, or sometimes Type attribute + content
                # DTT XML params usually have text content for value.
                val = param.text
                if val: val = val.strip()
                params[name_attr] = val
                
        # Now reconstruct the list
        # We look for MeasChn[i]
        i = 0
        while True:
            key_name = f"MeasChn[{i}]"
            # Note: Sometimes DTT uses specific formatting or nested params.
            # But mostly it follows simple object serialization.
            # Let's check simply.
            
            # Alternative: in LIGO_LW, it might be separate.
            # Let's try to match keys.
            
            if key_name not in params:
                 # Check if we exhausted sequential
                 # But maybe there are gaps? Usually not for arrays.
                 # Let's try up to 96 (max channels)
                 if i > 96: break
                 i += 1
                 continue
                 
            name = params[key_name]
            # Clean generic formatting if needed (sometimes "H1:..." sometimes just name)
            
            # Active status
            key_active = f"MeasActive[{i}]"
            active = True # Default
            if key_active in params:
                v = params[key_active]
                # XML boolean might be 'true', '1', 'false', '0'
                if v.lower() in ['false', '0']: active = False
            
            if name: # Only add if name is not empty
                channels.append({'name': name, 'active': active})
            
            i += 1
            
        # If the loop yields nothing, maybe the format is different (e.g. Table based)
        # But for 'TestParameters' restore, it is Param based.
            
    except Exception as e:
        print(f"XML Parsing Error: {e}")
        # Return empty list or minimal fallback
        pass
        
    return channels

def load_products(filename: str) -> dict:
    """
    Load data products from various file formats.
    Supported extensions: .xml (DTT XML), .gwf, .h5, .hdf5, .ffl, and generic formats supported by gwexpy.
    """
    if not isinstance(filename, str):
        return {}

    products = {}
    ext = Path(filename).suffix.lower()

    if ext == '.xml':
        from gwexpy.io.dttxml_common import load_dttxml_products
        products = load_dttxml_products(filename)
    elif ext in ['.gwf', '.h5', '.hdf5', '.ffl']:
        try:
            ts_dict = TimeSeriesDict.read(filename)
            products['TS'] = {str(k): v for k, v in ts_dict.items()}
        except Exception as e_dict:
             try:
                 ts = TimeSeries.read(filename)
                 products['TS'] = {ts.name or 'Channel0': ts}
             except Exception as e_single:
                 raise RuntimeError(f"Could not read as dict: {e_dict}\nCould not read as single: {e_single}")
    else:
         try:
            ts_dict = TimeSeriesDict.read(filename)
            products['TS'] = {str(k): v for k, v in ts_dict.items()}
         except Exception as e_dict:
            try:
                ts = TimeSeries.read(filename)
                name = ts.name if ts.name else "Channel0"
                products['TS'] = {name: ts}
            except Exception as e_single:
                raise RuntimeError(f"Unsupported file format or read error.\nDict Read: {e_dict}\nSingle Read: {e_single}")

    return products
