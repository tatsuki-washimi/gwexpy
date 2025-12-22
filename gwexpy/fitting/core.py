import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.util import describe, make_func_code
from .models import get_model

# Optional imports for MCMC
try:
    import emcee
    import corner
except ImportError:
    emcee = None
    corner = None

class ComplexLeastSquares:
    """
    Least Squares cost function for complex-valued data.
    Minimizes the sum of squared residuals for both Real and Imaginary parts.
    """
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, dy, model):
        self.x = x
        self.y = y  # Complex data
        self.dy = dy # Error (assumed isotropic for real/imag unless specified otherwise)
        self.model = model
        
        # Determine parameters from model (skipping 'x')
        # describe returns a list of parameter names
        params = describe(model)[1:] 
        self.func_code = make_func_code(params)
        
    def __call__(self, *args):
        # Calculate model prediction
        ym = self.model(self.x, *args)
        
        # Calculate residuals for Real and Imag parts
        res_real = (self.y.real - ym.real) / self.dy
        res_imag = (self.y.imag - ym.imag) / self.dy
        
        # Sum of squared residuals
        chi2 = np.sum(res_real**2 + res_imag**2)
        return chi2

    @property
    def ndata(self):
        # Effectively 2 * len(x) data points
        return 2 * len(self.x)


class RealLeastSquares:
    """
    Least Squares cost function for real-valued data.

    This is a small, dependency-light replacement for `iminuit.cost.LeastSquares`
    to avoid optional JIT/caching side effects in some environments.
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, dy, model):
        self.x = x
        self.y = y
        self.dy = dy
        self.model = model

        params = describe(model)[1:]
        self.func_code = make_func_code(params)

    def __call__(self, *args):
        ym = self.model(self.x, *args)
        res = (self.y - ym) / self.dy
        return np.sum(res**2)

    @property
    def ndata(self):
        return len(self.x)


class FitResult:
    def __init__(self, minuit_obj, model, x, y, dy=None, cost_func=None, x_label=None, y_label=None):
        self.minuit = minuit_obj
        self.model = model
        self.x = x
        self.y = y
        self.dy = dy
        self.cost_func = cost_func
        self.x_label = x_label
        self.y_label = y_label
        self.sampler = None
        self.samples = None
        self.mcmc_labels = None
        
    @property
    def params(self):
        """Best fit parameters (dict)."""
        return {name: self.minuit.values[name] for name in self.minuit.parameters}
        
    @property
    def errors(self):
        """Parameter errors (dict)."""
        return {name: self.minuit.errors[name] for name in self.minuit.parameters}
        
    @property
    def chi2(self):
        """Chi-square value (valid only for LeastSquares-like costs)."""
        # Both LeastSquares and ComplexLeastSquares return chi2 as fval
        return self.minuit.fval
        
    @property
    def ndof(self):
        """Number of degrees of freedom."""
        n_data = len(self.x)
        if np.iscomplexobj(self.y):
            # For complex data, we have Real and Imag parts, so effectively 2 * N data points
            n_data *= 2
        n_params = self.minuit.nfit
        return max(0, n_data - n_params)

    @property
    def reduced_chi2(self):
        """Reduced Chi-square value."""
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def __str__(self):
        """Delegate to Minuit's pretty printer."""
        return str(self.minuit)
        
    def _repr_html_(self):
        """Jupyter notebook integration."""
        return self.minuit._repr_html_()

    def plot(self, ax=None, num_points=1000, **kwargs):
        """
        Plot data and best-fit curve. 
        For complex data, delegates to bode_plot().
        """
        is_complex = np.iscomplexobj(self.y)
        
        if is_complex:
            return self.bode_plot(ax=ax, num_points=num_points, **kwargs)

        # Real Plot
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot Model (First)
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('zorder', 5)
        x_plot = np.linspace(min(self.x), max(self.x), num_points)
        y_plot = self.model(x_plot, **self.params)
        ax.plot(x_plot, y_plot, label='Fit', **kwargs)

        # Plot Data (Second)
        if self.dy is not None:
            ax.errorbar(self.x, self.y, yerr=self.dy, fmt='.', label='Data', color='black')
        else:
            ax.plot(self.x, self.y, '.', label='Data', color='black')
            
        if self.x_label:
            ax.set_xlabel(self.x_label)
        if self.y_label:
            ax.set_ylabel(self.y_label)
            
        ax.legend()
        return ax

    def bode_plot(self, ax=None, num_points=1000, **kwargs):
        """
        Create a Bode plot (Magnitude and Phase) for the fit result.
        """
        if ax is None:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 2:
            ax_mag, ax_phase = ax
        else:
            raise ValueError("For bode_plot, 'ax' must be a list/tuple of 2 axes (mag, phase), or None.")
        
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('zorder', 5)

        # --- Magnitude ---
        # Model (First)
        # Note: log scale requires positive values for x usually, check if min(x) > 0?
        # Assuming typical frequency, min(x) > 0. If linear x includes 0, log scale breaks.
        # But Bode is typically log X.
        x_start = min(self.x) if min(self.x) > 0 else (max(self.x) * 1e-3 if max(self.x) > 0 else 1e-1)
        # If x_range provided, use it. But here we just use data range.
        if x_start <= 0: x_start = 1e-1 # Fallback
            
        x_plot = np.logspace(np.log10(x_start), np.log10(max(self.x)), num_points)
        ym_plot = self.model(x_plot, **self.params)
        
        ax_mag.plot(x_plot, np.abs(ym_plot), label='Fit', **kwargs)
        
        # Data (Second)
        mag_data = np.abs(self.y)
        if self.dy is not None:
                ax_mag.errorbar(self.x, mag_data, yerr=self.dy, fmt='.', label='Data', color='black')
        else:
                ax_mag.plot(self.x, mag_data, '.', label='Data', color='black')

        ax_mag.set_ylabel('Magnitude')
        ax_mag.legend()
        ax_mag.grid(True, which='both', alpha=0.3)
        ax_mag.set_xscale('log')
        ax_mag.set_yscale('log')
        
        # --- Phase ---
        # Model (First)
        ax_phase.plot(x_plot, np.angle(ym_plot, deg=True), label='Fit', **kwargs)
        
        # Data (Second)
        phase_data = np.angle(self.y, deg=True) # Degrees
        ax_phase.plot(self.x, phase_data, '.', label='Data', color='black')
        
        ax_phase.set_ylabel('Phase [deg]')
        if self.x_label:
            ax_phase.set_xlabel(self.x_label)
        else:
            ax_phase.set_xlabel('Frequency / Time')
            
        ax_phase.grid(True, which='both', alpha=0.3)
        ax_phase.set_xscale('log')
        
        return (ax_mag, ax_phase)

    def run_mcmc(self, n_walkers=32, n_steps=3000, burn_in=500, progress=True):
        """
        Run MCMC using emcee starting from the best-fit parameters.
        """
        if emcee is None:
            raise ImportError("Please install 'emcee' and 'corner' to use MCMC features.")

        # 1. Parameter Information Extraction
        # Filter out fixed parameters for MCMC
        float_params = [p for p in self.minuit.parameters if not self.minuit.fixed[p]]
        ndim = len(float_params)
        
        # Dictionary of fixed parameters
        fixed_params = {p: self.minuit.values[p] for p in self.minuit.parameters if self.minuit.fixed[p]}

        # Log Probability Function
        def log_prob(theta):
            # theta: array of float values for float_params
            
            # Construct full parameter dictionary
            current_params = fixed_params.copy()
            for name, val in zip(float_params, theta):
                current_params[name] = val
                
                # Check limits defined in minuit
                if name in self.minuit.limits:
                    vmin, vmax = self.minuit.limits[name]
                    if not (vmin <= val <= vmax):
                        return -np.inf

            # iminuit Cost (Chi2) -> log_prob = -0.5 * Chi2
            # LeastSquares(*args) expects arguments in order
            try:
                args = [current_params[p] for p in self.minuit.parameters]
                
                # Support custom cost functions (like ComplexLeastSquares)
                # Note: iminuit costs usually take *args
                chi2 = self.cost_func(*args)
                return -0.5 * chi2
            except Exception:
                return -np.inf

        # Initial state: small ball around minuit result
        p0_float = np.array([self.minuit.values[p] for p in float_params])
        # Use hessian errors for initialization spread, or small value if fixed/zero
        stds = np.array([self.minuit.errors[p] if self.minuit.errors[p] > 0 else 1e-4 * abs(v) + 1e-8
                         for p, v in zip(float_params, p0_float)])
        
        pos = p0_float + stds * 1e-1 * np.random.randn(n_walkers, ndim)
        
        # Run emcee
        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        self.sampler.run_mcmc(pos, n_steps, progress=progress)
        
        # Save flattened samples (discarding burn-in)
        self.samples = self.sampler.get_chain(discard=burn_in, flat=True)
        self.mcmc_labels = float_params
        
        return self.sampler

    def plot_corner(self, **kwargs):
        """Plot corner plot of MCMC samples."""
        if corner is None: 
            raise ImportError("Please install 'corner' to use plot_corner.")
        if self.samples is None: 
            raise RuntimeError("Run .run_mcmc() first.")
        
        # Show BestFit truth lines
        if self.mcmc_labels:
            truths = [self.minuit.values[p] for p in self.mcmc_labels]
            kwargs.setdefault('truths', truths)
            kwargs.setdefault('labels', self.mcmc_labels)
        
        return corner.corner(self.samples, **kwargs)

def fit_series(series, model, x_range=None, sigma=None, 
               p0=None, limits=None, fixed=None, **kwargs):
    """
    Fit a Series object using iminuit.
    Supports real and complex valued Series (simultaneous Re/Im fit).
    """
    # 0. モデルの解決
    if isinstance(model, str):
        model_name = model
        model = get_model(model_name)
        
    # 1. データの準備 (Crop & Unit Stripping)
    target = series.crop(*x_range) if x_range else series
    
    # x軸の取得
    x_label = 'x'
    y_label = 'y'
    
    if hasattr(target, 'frequencies'):
        x = target.frequencies.value
        x_label = 'Frequency'
        if hasattr(target, 'xunit') and str(target.xunit) != 'dimensionless':
             x_label += f" [{target.xunit}]"
    elif hasattr(target, 'times'):
        x = target.times.value
        x_label = 'Time'
        if hasattr(target, 'xunit') and str(target.xunit) != 'dimensionless':
             x_label += f" [{target.xunit}]"
    else:
        x = target.xindex.value
        if hasattr(target, 'xunit') and str(target.xunit) != 'dimensionless':
             x_label += f" [{target.xunit}]"
        
    y = target.value
    
    # Determine y-label
    if hasattr(target, 'unit') and str(target.unit) != 'dimensionless':
        y_label_unit = f"[{target.unit}]"
    else:
        y_label_unit = ""

    if hasattr(target, 'name') and target.name:
        y_label = f"{target.name}"
        if y_label_unit:
            y_label += f" {y_label_unit}"
    elif y_label_unit:
        y_label = f"Amplitude {y_label_unit}"

    is_complex = np.iscomplexobj(y)
    
    # 誤差の処理
    if sigma is None:
        # 重みなし最小二乗 (Error=1.0)
        dy = np.ones(len(y)) # Real array
    else:
        dy = np.asarray(sigma)
        if len(dy) != len(y):
            raise ValueError("Sigma length mismatch")

    # 2. Cost Function
    if is_complex:
        cost = ComplexLeastSquares(x, y, dy, model)
    else:
        cost = RealLeastSquares(x, y, dy, model)
    
    # 3. Minuit 初期化
    init_params = p0 if p0 else {}
    m = Minuit(cost, **init_params)
    
    # 4. Limit / Fix の適用
    if limits:
        for name, (vmin, vmax) in limits.items():
            m.limits[name] = (vmin, vmax)
            
    if fixed:
        for name in fixed:
            m.fixed[name] = True
            
    # 5. 実行
    m.migrad()
    m.hesse()
    
    return FitResult(m, model, x, y, dy, cost, x_label=x_label, y_label=y_label)
