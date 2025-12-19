
from ._optional import require_optional

# MT handling is complex, placeholder
# xarray interop is key

# MT handling is complex, minimal implementation
# xarray interop is key

def to_mth5(series_or_collection, mth5_obj, station=None, run=None, channel_type="electric"):
    """
    Write a TimeSeries to an MTH5 file/object.

    Parameters
    ----------
    series_or_collection : TimeSeries
        Data to write. P1 scope covers single TimeSeries.
    mth5_obj : mth5.mth5.MTH5 or str
        Open MTH5 object or path to filename.
    station : str
        Station name.
    run : str
        Run name.
    channel_type : str
        'electric', 'magnetic', 'auxiliary'.
    """
    from ._optional import require_optional
    mth5 = require_optional("mth5")
    
    # Handle filename
    file_managed = False
    if isinstance(mth5_obj, str):
         # It's a path
         filename = mth5_obj
         mth5_obj = mth5.mth5.MTH5()
         mth5_obj.open_mth5(filename, mode='a')
         file_managed = True
         
    try:
         # Ensure station exists
         if station is None:
              station = "Station01"
         if not mth5_obj.has_station(station):
              st_group = mth5_obj.add_station(station)
         else:
              st_group = mth5_obj.get_station(station)
              
         # Ensure run exists
         if run is None:
              run = "Run01"
         if not st_group.has_run(run):
              run_group = st_group.add_run(run)
         else:
              run_group = st_group.get_run(run)
              
         # Add Channel
         # Map TimeSeries metadata
         ts = series_or_collection
         
         # Identify component
         comp = ts.name if ts.name else "Ex"
         
         # Sample rate
         if hasattr(ts, 'sample_rate'):
             sr = ts.sample_rate.value
         elif hasattr(ts, 'dt'):
             sr = 1.0 / ts.dt.to('s').value
         else:
             sr = 1.0
             
         data = ts.value
         
         # MTH5 add_channel expects xarray or numpy?
         # It usually takes a numpy array and metadata.
         run_group.add_channel(
             comp, 
             channel_type, 
             data=data, 
             sample_rate=sr,
             start=ts.t0.to('s').value if ts.t0 else 0
         )
         
    finally:
         if file_managed:
              mth5_obj.close_mth5()

def from_mth5(mth5_obj, station, run, channel):
    """
    Read a channel from MTH5 to TimeSeries.
    """
    from ._optional import require_optional
    mth5 = require_optional("mth5")
    from gwexpy.timeseries import TimeSeries
    import astropy.units as u
    
    file_managed = False
    if isinstance(mth5_obj, str):
         filename = mth5_obj
         mth5_obj = mth5.mth5.MTH5()
         mth5_obj.open_mth5(filename, mode='r')
         file_managed = True
         
    try:
         st_group = mth5_obj.get_station(station)
         run_group = st_group.get_run(run)
         ch_obj = run_group.get_channel(channel)
         
         # Convert to TimeSeries
         # ch_obj.to_xarray() often handy
         # Or access hdf5 dataset directly: ch_obj.hdf5_dataset[()]
         # mth5 channel object wraps dataset.
         
         # mth5 > 0.x provides .ts (mth5.timeseries.ChannelTS)
         # Using raw data fetch
         data = ch_obj.time_slice(0, ch_obj.n_samples) # Get all?
         # Wait, time_slice isn't standard? 
         # ch_obj.hdf5_dataset is h5py dataset
         val = ch_obj.hdf5_dataset[:]
         
         # Meta
         sr = ch_obj.sample_rate
         dt = (1.0 / sr) * u.s
         start = ch_obj.start
         # start might be string or float GPS? MTH5 uses ISO string usually.
         # We need to parse.
         from astropy.time import Time
         try:
              t0 = Time(start).gps * u.s
         except:
              t0 = 0 * u.s
              
         # Unit?
         try:
              unit_str = ch_obj.units
              unit = u.Unit(unit_str)
         except:
              unit = None
              
         return TimeSeries(val, dt=dt, t0=t0, unit=unit, name=channel)
         
    finally:
         if file_managed:
              mth5_obj.close_mth5()
