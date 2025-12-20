from .spectrogram import SpectrogramList, SpectrogramDict, SpectrogramMatrix

# Dynamic import from gwpy
import gwpy.spectrogram
for key in dir(gwpy.spectrogram):
    if not key.startswith("_") and key not in locals():
        locals()[key] = getattr(gwpy.spectrogram, key)

__all__ = [k for k in locals().keys() if not k.startswith("_") and k != "gwpy"]
