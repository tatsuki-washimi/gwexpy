
import numpy as np
from typing import Any, Optional
from ._optional import require_optional
from .base import to_plain_array

def _get_label(obj, unit, default_name="x"):
    name = getattr(obj, "name", None) or default_name
    unit_str = str(unit) if unit else ""
    if unit_str:
        return f"{name} [{unit_str}]"
    return str(name)

def _extract_error_array(series, error):
    """Internal helper to extract a matching error array from various types."""
    from astropy import units as u
    
    # If it's a gwpy/gwexpy Series
    if hasattr(error, "value") and hasattr(error, "xindex"):
        if len(error) != len(series):
            raise ValueError(f"Error series length ({len(error)}) does not match data length ({len(series)})")
        # Check xindex matching (optional check, maybe too strict if slightly shifted?)
        # For now, just extract value
        return to_plain_array(error.value)
    
    # If it's an astropy Quantity
    if isinstance(error, u.Quantity):
        if error.shape != series.shape:
            raise ValueError(f"Error shape {error.shape} does not match data shape {series.shape}")
        # Try to match units if series has unit
        if hasattr(series, "unit") and series.unit:
            try:
                return error.to(series.unit).value
            except u.UnitConversionError:
                # If not convertible, just use raw value but warn? 
                # User might be passing relative error or something. 
                # But requirement says "same unit".
                pass
        return error.value

    # If it's a plain numpy array or list
    err_arr = np.asarray(error)
    if err_arr.shape != series.shape:
        raise ValueError(f"Error shape {err_arr.shape} does not match data shape {series.shape}")
    return err_arr

def to_tgraph(series, error=None):
    """
    Convert 1D Series to ROOT TGraph or TGraphErrors.
    """
    ROOT = require_optional("ROOT")
    
    x = to_plain_array(series.xindex).astype(float)
    y = to_plain_array(series.value).astype(float)
    n = len(x)
    
    if error is not None:
        ey = _extract_error_array(series, error).astype(float)
        ex = np.zeros(n)
        graph = ROOT.TGraphErrors(n, x, y, ex, ey)
    else:
        graph = ROOT.TGraph(n, x, y)
        
    name = str(series.name or "graph")
    graph.SetName(name)
    graph.SetTitle(name)
    
    # Axis labels
    xunit = str(series.xunit) if hasattr(series, "xunit") else ""
    yunit = str(series.unit) if hasattr(series, "unit") else ""
    
    graph.GetXaxis().SetTitle(_get_label(series.xindex, series.xunit, default_name="x"))
    graph.GetYaxis().SetTitle(_get_label(series, series.unit, default_name="y"))
    
    return graph

def to_th1d(series, error=None):
    """
    Convert 1D Series to ROOT TH1D.
    """
    ROOT = require_optional("ROOT")
    
    x = to_plain_array(series.xindex).astype(float)
    y = to_plain_array(series.value).astype(float)
    n = len(x)
    
    if n < 2:
        raise ValueError("Series must have at least 2 points for TH1D")

    # Determine binning
    # TH1D requires bin edges. Series usually represents bin centers or samples.
    # We assume regular if possible, or use variable bins.
    
    dx_vals = np.diff(x)
    is_regular = np.allclose(dx_vals, dx_vals[0])
    
    name = str(series.name or "hist")
    title = name
    
    if is_regular:
        dx = dx_vals[0]
        xlow = x[0] - dx/2.0
        xup = x[-1] + dx/2.0
        hist = ROOT.TH1D(name, title, n, xlow, xup)
    else:
        # Variable bin widths
        # We construct edges halfway between centers
        edges = np.zeros(n + 1)
        edges[1:-1] = (x[:-1] + x[1:]) / 2.0
        # Extrapolate edges for first and last
        edges[0] = x[0] - (edges[1] - x[0])
        edges[-1] = x[-1] + (x[-1] - edges[-2])
        hist = ROOT.TH1D(name, title, n, edges)
        
    # Fill bins (TH1 bins are 1-indexed, 0 is underflow)
    for i in range(n):
        hist.SetBinContent(i + 1, y[i])
        
    if error is not None:
        ey = _extract_error_array(series, error).astype(float)
        for i in range(n):
            hist.SetBinError(i + 1, ey[i])
            
    # Labels
    xunit = str(series.xunit) if hasattr(series, "xunit") else ""
    yunit = str(series.unit) if hasattr(series, "unit") else ""
    hist.GetXaxis().SetTitle(_get_label(series.xindex, series.xunit, default_name="x"))
    hist.GetYaxis().SetTitle(_get_label(series, series.unit, default_name="y"))
    
    return hist

def to_th2d(spec, error=None):
    """
    Convert Spectrogram to ROOT TH2D.
    """
    ROOT = require_optional("ROOT")
    
    times = to_plain_array(spec.times).astype(float)
    freqs = to_plain_array(spec.frequencies).astype(float)
    data = to_plain_array(spec.value).astype(float)
    
    nt = len(times)
    nf = len(freqs)
    
    # helper for edges
    def _get_edges(arr):
        if len(arr) < 2:
             return np.array([arr[0]-0.5, arr[0]+0.5])
        dx = np.diff(arr)
        if np.allclose(dx, dx[0]):
             step = dx[0]
             return np.linspace(arr[0]-step/2, arr[-1]+step/2, len(arr)+1)
        else:
             edges = np.zeros(len(arr)+1)
             edges[1:-1] = (arr[:-1] + arr[1:]) / 2.0
             edges[0] = arr[0] - (edges[1] - arr[0])
             edges[-1] = arr[-1] + (arr[-1] - edges[-2])
             return edges

    t_edges = _get_edges(times)
    f_edges = _get_edges(freqs)
    
    name = str(spec.name or "spectrogram")
    hist = ROOT.TH2D(name, name, nt, t_edges, nf, f_edges)
    
    # Fill
    for i in range(nt):
        for j in range(nf):
            hist.SetBinContent(i+1, j+1, data[i, j])
            
    if error is not None:
        err_arr = np.asarray(error).astype(float)
        if err_arr.shape != data.shape:
             raise ValueError("Error shape mismatch")
        for i in range(nt):
            for j in range(nf):
                hist.SetBinError(i+1, j+1, err_arr[i, j])
                
    # Labels
    hist.GetXaxis().SetTitle(_get_label(spec.times, spec.times.unit, default_name="time"))
    hist.GetYaxis().SetTitle(_get_label(spec.frequencies, spec.frequencies.unit, default_name="frequency"))
    hist.GetZaxis().SetTitle(_get_label(spec, spec.unit, default_name="value"))
    
    return hist

def from_root(cls, obj, return_error=False):
    """
    Create Series (TimeSeries or FrequencySeries) from ROOT TGraph or TH1.
    """
    ROOT = require_optional("ROOT")
    
    # Check type
    is_hist = isinstance(obj, ROOT.TH1)
    is_hist2d = isinstance(obj, ROOT.TH2)
    is_graph = isinstance(obj, ROOT.TGraph)
    
    if not is_hist and not is_graph and not is_hist2d:
        raise TypeError(f"Object {obj} is neither TH1, TH2 nor TGraph")
        
    if is_hist2d:
        nx = obj.GetNbinsX()
        ny = obj.GetNbinsY()
        x = np.array([obj.GetXaxis().GetBinCenter(i+1) for i in range(nx)])
        y = np.array([obj.GetYaxis().GetBinCenter(j+1) for j in range(ny)])
        z = np.zeros((nx, ny))
        ez = np.zeros((nx, ny)) if return_error else None
        
        for i in range(nx):
            for j in range(ny):
                z[i, j] = obj.GetBinContent(i+1, j+1)
                if return_error:
                    ez[i, j] = obj.GetBinError(i+1, j+1)
        
        # In GWpy Spectrogram, typically shape is (Time, Freq)
        name = obj.GetName()
        unit = None
        # ... logic for unit extraction from Z axis if possible ...
        z_title = obj.GetZaxis().GetTitle()
        if "[" in z_title and "]" in z_title:
             import re
             match = re.search(r"\[(.*?)\]", z_title)
             if match: unit = match.group(1)
             
        res = cls(z, times=x, frequencies=y, unit=unit, name=name)
        if return_error:
             err_res = cls(ez, times=x, frequencies=y, unit=unit, name=f"{name}_error")
             return res, err_res
        return res

    if is_hist:
        n = obj.GetNbinsX()
        x = np.array([obj.GetBinCenter(i+1) for i in range(n)])
        y = np.array([obj.GetBinContent(i+1) for i in range(n)])
        ey = np.array([obj.GetBinError(i+1) for i in range(n)]) if return_error else None
    else: # is_graph
        n = obj.GetN()
        # buffer access is faster
        x = np.frombuffer(obj.GetX(), dtype=np.float64, count=n).copy()
        y = np.frombuffer(obj.GetY(), dtype=np.float64, count=n).copy()
        if return_error:
             if hasattr(obj, "GetEY"):
                 ey = np.frombuffer(obj.GetEY(), dtype=np.float64, count=n).copy()
             else:
                 ey = np.zeros(n)
        else:
             ey = None
             
    # Try to extract name and unit
    name = obj.GetName()
    title = obj.GetYaxis().GetTitle()
    unit = None
    if "[" in title and "]" in title:
        # Simple parser "Name [Unit]"
        import re
        match = re.search(r"\[(.*?)\]", title)
        if match:
            unit = match.group(1)
            
    # Regularity check
    if n > 1:
        dx_vals = np.diff(x)
        if np.allclose(dx_vals, dx_vals[0]):
            res = cls(y, x0=float(x[0]), dx=float(dx_vals[0]), unit=unit, name=name)
        else:
            if "Frequency" in cls.__name__:
                res = cls(y, frequencies=x, unit=unit, name=name)
            else:
                res = cls(y, times=x, unit=unit, name=name)
    else:
        res = cls(y, x0=float(x[0]) if n==1 else 0, unit=unit, name=name)
        
    if return_error:
        # Create a matching series for error
        if "Frequency" in cls.__name__:
            err_res = cls(ey, frequencies=x, unit=unit, name=f"{name}_error")
        else:
            err_res = cls(ey, times=x, unit=unit, name=f"{name}_error")
        return res, err_res
        
    return res

def to_tmultigraph(collection, name: Optional[str] = None) -> Any:
    """
    Convert a collection of Series to a ROOT TMultiGraph.
    """
    ROOT = require_optional("ROOT")
    mg = ROOT.TMultiGraph()
    title = name or getattr(collection, "name", "multigraph")
    mg.SetName(str(title))
    mg.SetTitle(str(title))
    
    # Handle dict or list
    if hasattr(collection, "items"):
        items = collection.items()
    else:
        items = enumerate(collection)
        
    for i, (key, series) in enumerate(items):
        if hasattr(series, "to_tgraph"):
            g = series.to_tgraph()
        else:
            g = to_tgraph(series)
             
        # Set default colors (ROOT color cycle: 1=Black, 2=Red, 3=Green, 4=Blue, ...)
        # Skip 0 (White) and handle large i
        color = (i % 9) + 1
        g.SetLineColor(color)
        g.SetMarkerColor(color)
        
        # Ensure it has a meaningful name in the legend
        if g.GetName() in ["graph", ""]:
             g.SetName(str(key))
             g.SetTitle(str(key))
             
        mg.Add(g)
        
    return mg

def write_root_file(collection, filename: str, **kwargs: Any) -> None:
    """
    Write a collection of Series to a ROOT TFile.
    """
    ROOT = require_optional("ROOT")
    mode = kwargs.get("mode", "recreate")
    
    # Save current directory to restore later
    old_dir = ROOT.gDirectory.GetDirectory("")
    
    f = ROOT.TFile.Open(filename, mode)
    if not f or f.IsZombie():
        raise IOError(f"Failed to open {filename} for writing")
        
    if hasattr(collection, "items"):
        items = collection.items()
    else:
        items = enumerate(collection)
        
    for key, series in items:
        if hasattr(series, "to_th2d"):
             obj = series.to_th2d()
        elif hasattr(series, "to_tgraph"):
             obj = series.to_tgraph()
        else:
             obj = to_tgraph(series)
        
        # Determine a good name for the object in the ROOT file
        obj_name = str(key)
        if isinstance(key, int):
             # For lists, try to use the object's own name if it has one
             obj_name = getattr(series, "name", None) or str(key)
             
        obj.SetName(str(obj_name))
        obj.Write()
        
    f.Close()
    if old_dir:
        old_dir.cd()
