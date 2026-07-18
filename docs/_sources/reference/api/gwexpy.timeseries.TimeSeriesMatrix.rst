gwexpy.timeseries.TimeSeriesMatrix
==================================

.. currentmodule:: gwexpy.timeseries

.. autoclass:: TimeSeriesMatrix

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~TimeSeriesMatrix.__init__
      ~TimeSeriesMatrix.abs
      ~TimeSeriesMatrix.all
      ~TimeSeriesMatrix.angle
      ~TimeSeriesMatrix.any
      ~TimeSeriesMatrix.append
      ~TimeSeriesMatrix.append_exact
      ~TimeSeriesMatrix.argmax
      ~TimeSeriesMatrix.argmin
      ~TimeSeriesMatrix.argpartition
      ~TimeSeriesMatrix.argsort
      ~TimeSeriesMatrix.asd
      ~TimeSeriesMatrix.astype
      ~TimeSeriesMatrix.auto_coherence
      ~TimeSeriesMatrix.bandpass
      ~TimeSeriesMatrix.byteswap
      ~TimeSeriesMatrix.choose
      ~TimeSeriesMatrix.clip
      ~TimeSeriesMatrix.coherence
      ~TimeSeriesMatrix.coherence_ranking
      ~TimeSeriesMatrix.col_index
      ~TimeSeriesMatrix.col_keys
      ~TimeSeriesMatrix.compress
      ~TimeSeriesMatrix.conj
      ~TimeSeriesMatrix.conjugate
      ~TimeSeriesMatrix.copy
      ~TimeSeriesMatrix.correlation
      ~TimeSeriesMatrix.correlation_vector
      ~TimeSeriesMatrix.crop
      ~TimeSeriesMatrix.csd
      ~TimeSeriesMatrix.cumprod
      ~TimeSeriesMatrix.cumsum
      ~TimeSeriesMatrix.degree
      ~TimeSeriesMatrix.det
      ~TimeSeriesMatrix.detrend
      ~TimeSeriesMatrix.diagonal
      ~TimeSeriesMatrix.diff
      ~TimeSeriesMatrix.distance_correlation
      ~TimeSeriesMatrix.dot
      ~TimeSeriesMatrix.dump
      ~TimeSeriesMatrix.dumps
      ~TimeSeriesMatrix.fft
      ~TimeSeriesMatrix.fill
      ~TimeSeriesMatrix.filter
      ~TimeSeriesMatrix.flatten
      ~TimeSeriesMatrix.from_neo
      ~TimeSeriesMatrix.get_index
      ~TimeSeriesMatrix.getfield
      ~TimeSeriesMatrix.highpass
      ~TimeSeriesMatrix.hilbert
      ~TimeSeriesMatrix.ica
      ~TimeSeriesMatrix.ica_fit
      ~TimeSeriesMatrix.ica_inverse_transform
      ~TimeSeriesMatrix.ica_transform
      ~TimeSeriesMatrix.impute
      ~TimeSeriesMatrix.interpolate
      ~TimeSeriesMatrix.inv
      ~TimeSeriesMatrix.is_compatible
      ~TimeSeriesMatrix.is_compatible_exact
      ~TimeSeriesMatrix.is_contiguous
      ~TimeSeriesMatrix.is_contiguous_exact
      ~TimeSeriesMatrix.item
      ~TimeSeriesMatrix.keys
      ~TimeSeriesMatrix.ktau
      ~TimeSeriesMatrix.kurtosis
      ~TimeSeriesMatrix.lock_in
      ~TimeSeriesMatrix.lowpass
      ~TimeSeriesMatrix.max
      ~TimeSeriesMatrix.mean
      ~TimeSeriesMatrix.median
      ~TimeSeriesMatrix.mic
      ~TimeSeriesMatrix.min
      ~TimeSeriesMatrix.nonzero
      ~TimeSeriesMatrix.notch
      ~TimeSeriesMatrix.pad
      ~TimeSeriesMatrix.partial_correlation_matrix
      ~TimeSeriesMatrix.partition
      ~TimeSeriesMatrix.pca
      ~TimeSeriesMatrix.pca_fit
      ~TimeSeriesMatrix.pca_inverse_transform
      ~TimeSeriesMatrix.pca_transform
      ~TimeSeriesMatrix.pcc
      ~TimeSeriesMatrix.phase
      ~TimeSeriesMatrix.plot
      ~TimeSeriesMatrix.prepend
      ~TimeSeriesMatrix.prepend_exact
      ~TimeSeriesMatrix.prod
      ~TimeSeriesMatrix.psd
      ~TimeSeriesMatrix.put
      ~TimeSeriesMatrix.q_transform
      ~TimeSeriesMatrix.radian
      ~TimeSeriesMatrix.ravel
      ~TimeSeriesMatrix.read
      ~TimeSeriesMatrix.repeat
      ~TimeSeriesMatrix.resample
      ~TimeSeriesMatrix.reshape
      ~TimeSeriesMatrix.resize
      ~TimeSeriesMatrix.rms
      ~TimeSeriesMatrix.rolling_max
      ~TimeSeriesMatrix.rolling_mean
      ~TimeSeriesMatrix.rolling_median
      ~TimeSeriesMatrix.rolling_min
      ~TimeSeriesMatrix.rolling_std
      ~TimeSeriesMatrix.round
      ~TimeSeriesMatrix.row_index
      ~TimeSeriesMatrix.row_keys
      ~TimeSeriesMatrix.schur
      ~TimeSeriesMatrix.searchsorted
      ~TimeSeriesMatrix.setfield
      ~TimeSeriesMatrix.setflags
      ~TimeSeriesMatrix.shift
      ~TimeSeriesMatrix.skewness
      ~TimeSeriesMatrix.sort
      ~TimeSeriesMatrix.spectrogram
      ~TimeSeriesMatrix.spectrogram2
      ~TimeSeriesMatrix.squeeze
      ~TimeSeriesMatrix.standardize
      ~TimeSeriesMatrix.std
      ~TimeSeriesMatrix.step
      ~TimeSeriesMatrix.submatrix
      ~TimeSeriesMatrix.sum
      ~TimeSeriesMatrix.swapaxes
      ~TimeSeriesMatrix.take
      ~TimeSeriesMatrix.taper
      ~TimeSeriesMatrix.to_cupy
      ~TimeSeriesMatrix.to_dask
      ~TimeSeriesMatrix.to_device
      ~TimeSeriesMatrix.to_dict
      ~TimeSeriesMatrix.to_dict_flat
      ~TimeSeriesMatrix.to_hdf5
      ~TimeSeriesMatrix.to_jax
      ~TimeSeriesMatrix.to_list
      ~TimeSeriesMatrix.to_mne
      ~TimeSeriesMatrix.to_neo
      ~TimeSeriesMatrix.to_pandas
      ~TimeSeriesMatrix.to_series_1Dlist
      ~TimeSeriesMatrix.to_series_2Dlist
      ~TimeSeriesMatrix.to_tensorflow
      ~TimeSeriesMatrix.to_torch
      ~TimeSeriesMatrix.to_zarr
      ~TimeSeriesMatrix.tobytes
      ~TimeSeriesMatrix.tofile
      ~TimeSeriesMatrix.tolist
      ~TimeSeriesMatrix.trace
      ~TimeSeriesMatrix.transfer_function
      ~TimeSeriesMatrix.transpose
      ~TimeSeriesMatrix.update
      ~TimeSeriesMatrix.value_at
      ~TimeSeriesMatrix.var
      ~TimeSeriesMatrix.view
      ~TimeSeriesMatrix.whiten
      ~TimeSeriesMatrix.whiten_channels
      ~TimeSeriesMatrix.write
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~TimeSeriesMatrix.MetaDataMatrix
      ~TimeSeriesMatrix.N_samples
      ~TimeSeriesMatrix.T
      ~TimeSeriesMatrix.base
      ~TimeSeriesMatrix.channel_names
      ~TimeSeriesMatrix.channels
      ~TimeSeriesMatrix.ctypes
      ~TimeSeriesMatrix.data
      ~TimeSeriesMatrix.default_xunit
      ~TimeSeriesMatrix.default_yunit
      ~TimeSeriesMatrix.device
      ~TimeSeriesMatrix.dt
      ~TimeSeriesMatrix.dtype
      ~TimeSeriesMatrix.duration
      ~TimeSeriesMatrix.dx
      ~TimeSeriesMatrix.flags
      ~TimeSeriesMatrix.flat
      ~TimeSeriesMatrix.imag
      ~TimeSeriesMatrix.is_regular
      ~TimeSeriesMatrix.itemsize
      ~TimeSeriesMatrix.loc
      ~TimeSeriesMatrix.mT
      ~TimeSeriesMatrix.names
      ~TimeSeriesMatrix.nbytes
      ~TimeSeriesMatrix.ndim
      ~TimeSeriesMatrix.real
      ~TimeSeriesMatrix.sample_rate
      ~TimeSeriesMatrix.series_type
      ~TimeSeriesMatrix.shape
      ~TimeSeriesMatrix.shape3D
      ~TimeSeriesMatrix.size
      ~TimeSeriesMatrix.span
      ~TimeSeriesMatrix.strides
      ~TimeSeriesMatrix.t0
      ~TimeSeriesMatrix.times
      ~TimeSeriesMatrix.units
      ~TimeSeriesMatrix.value
      ~TimeSeriesMatrix.x0
      ~TimeSeriesMatrix.xarray
      ~TimeSeriesMatrix.xindex
      ~TimeSeriesMatrix.xspan
      ~TimeSeriesMatrix.xunit
   
   