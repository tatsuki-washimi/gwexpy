"""Generate two synthetic channels and save their ASDs to asd.png."""

# quickstart-begin
from gwexpy.noise.wave import gaussian, sine
from gwexpy.timeseries import TimeSeriesDict

settings = dict(duration=16, sample_rate=512, t0=0, unit="V")
tone = sine(frequency=40, **settings)
channels = TimeSeriesDict(
    {
        "Sensor A": tone + gaussian(std=0.3, seed=10, **settings),
        "Sensor B": tone + gaussian(std=0.8, seed=20, **settings),
    }
)
spectra = channels.asd(fftlength=2, overlap=1, window="hann", method="welch")
plot = spectra.plot(xlim=(1, 256), ylabel=r"ASD [V/$\sqrt{\mathrm{Hz}}$]")
plot.gca().legend()
plot.savefig("asd.png")
# quickstart-end
