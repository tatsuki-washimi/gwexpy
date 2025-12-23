
import numpy as np
import matplotlib.pyplot as plt
from gwexpy.timeseries import TimeSeries

# 1. Generate Synthetic Data
np.random.seed(42)
x = np.linspace(0, 10, 100)
true_params = {'a': 2.5, 'b': 1.2}
y = true_params['a'] * x + true_params['b']
y_data = y + np.random.normal(0, 0.5, size=len(x))

# Create TimeSeries
ts = TimeSeries(y_data, x0=0, dx=x[1]-x[0], name='MCMC Test Data', unit='V')

# 2. Define Model
def linear_model(x, a, b):
    return a * x + b

# 3. Fit using iminuit
print("Running Least Squares Fit...")
# Initial guess
p0 = {'a': 1.0, 'b': 1.0}
result = ts.fit(linear_model, p0=p0)

print("Fit Result:")
print(result.params)
print("Chi2/ndof:", result.reduced_chi2)

# Verify Fit Plot
try:
    fig, ax = plt.subplots()
    result.plot(ax=ax)
    plt.close(fig)
    print("Fit plot created successfully.")
except Exception as e:
    print(f"Fit plot failed: {e}")

# 4. Run MCMC
print("\nRunning MCMC...")
try:
    sampler = result.run_mcmc(n_walkers=32, n_steps=500, burn_in=100, progress=False)
    print("MCMC finished.")
    print("Samples shape:", result.samples.shape)
    
    # 5. Corner Plot
    print("Creating Corner Plot...")
    fig = result.plot_corner()
    plt.close(fig)
    print("Corner plot created successfully.")
    
except ImportError as e:
    print(f"Skipping MCMC test: {e}")
except Exception as e:
    print(f"MCMC failed: {e}")
    import traceback
    traceback.print_exc()

