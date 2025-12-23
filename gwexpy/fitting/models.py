import numpy as np
from inspect import signature, Parameter

def gaussian(x, A, mu, sigma):
    """
    Gaussian distribution.
    f(x) = A * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def exponential(x, A, tau):
    """
    Exponential decay.
    f(x) = A * exp(-x / tau)
    """
    return A * np.exp(-x / tau)

def power_law(x, A, alpha):
    """
    Power law.
    f(x) = A * x^alpha
    """
    return A * x ** alpha

def damped_oscillation(x, A, tau, f, phi=0):
    """
    Damped oscillation.
    f(x) = A * exp(-x / tau) * sin(2 * pi * f * x + phi)
    """
    return A * np.exp(-x / tau) * np.sin(2 * np.pi * f * x + phi)

def landau(x, A, mu, sigma):
    """
    Landau distribution approximation (Moyal distribution).
    f(x) = A * exp(-0.5 * (lambda + exp(-lambda))), where lambda = (x - mu) / sigma
    """
    l = (x - mu) / sigma
    return A * np.exp(-0.5 * (l + np.exp(-l)))

def make_pol_func(n):
    """
    Create a polynomial function of degree n with signature (x, p0, p1, ..., pn).
    f(x) = p0 + p1*x + ... + pn*x^n
    """
    param_names = [f'p{i}' for i in range(n + 1)]
    
    def pol_func(x, *args, **kwargs):
        # Handle both positional and keyword arguments
        if kwargs:
            p_vals = [kwargs[name] for name in param_names]
        else:
            p_vals = args
            
        res = 0
        for i, p in enumerate(p_vals):
            res += p * x**i
        return res
    
    # Construct the function signature manually so iminuit can detect parameter names
    parameters = [Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)]
    for name in param_names:
        parameters.append(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD))
    
    pol_func.__signature__ = signature(lambda: None).replace(parameters=parameters)
    pol_func.__doc__ = f"Polynomial of degree {n}: p0 + p1*x + ... + p{n}*x^{n}"
    return pol_func

# Predefined dictionary
MODELS = {
    'gaus': gaussian,
    'gaussian': gaussian,
    'exp': exponential,
    'expo': exponential,  # ROOT style
    'exponential': exponential,
    'landau': landau,
    'power_law': power_law,
    'damped_oscillation': damped_oscillation,
}

# Add pol0 to pol9
for i in range(10):
    MODELS[f'pol{i}'] = make_pol_func(i)

def get_model(name):
    """
    Retrieve a model function by name (case-insensitive).
    """
    if not isinstance(name, str):
        return None
    
    key = name.lower()
    if key in MODELS:
        return MODELS[key]
    
    # Handle polN for N >= 10 if needed, though usually not recommended
    if key.startswith('pol') and key[3:].isdigit():
        n = int(key[3:])
        return make_pol_func(n)
        
    raise ValueError(f"Unknown model name: {name}. Available: {list(MODELS.keys())}")
