import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee
import corner



# Define the log likelihood function for emcee
def log_likelihood(params, x, y, yerr):
    model = composite_model(x, *params)
    resid = y - model
    log_like = -0.5 * np.sum((resid / yerr)**2 + np.log(2 * np.pi * yerr**2))
    return log_like

# Define the composite model function
def composite_model(x, slope_linear, length_scale_rbf, amplitude_rq, length_scale_rq, alpha_rq, amplitude_exp, length_scale_exp, period_exp):
    k_linear = slope_linear * x
    k_rbf = amplitude_rq**2 * np.exp(-0.5 * (x / length_scale_rbf)**2)
    k_rq = amplitude_rq**2 * (1 + (x**2 / (2 * alpha_rq * length_scale_rq**2)))**(-alpha_rq)
    k_exp = amplitude_exp**2 * np.exp(-0.5 * (x / length_scale_exp)**2) * np.cos(2 * np.pi * x / period_exp)

    return k_linear + k_rbf + k_rq + k_exp



def classic_trapfit(X,y,yerr = None, make_plots = False, output_fp = None):

    def plotter(output_fp = None):
        # Plotting the posterior distributions of parameters
        if output_fp is not None: 
            # Plotting the posterior distributions of parameters
            labels = ["slope_linear", "length_scale_rbf", "amplitude_rq", "length_scale_rq", "alpha_rq", "amplitude_exp", "length_scale_exp", "period_exp"]
            corner.corner(samples, labels=labels, truths=fit_params)
            plt.show()

            # Predict using the optimized model
            t_pred = np.linspace(0, 5, 500)
            y_pred = composite_model(t_pred, *fit_params)

            # Plotting the fitted composite model
            plt.figure(figsize=(10, 6))
            plt.errorbar(X, y, yerr=y_err, fmt='o', label='Data')
            plt.plot(t_pred, y_pred, 'b', label='Fitted Model')
            plt.plot(X, y_true, 'g', label='True Model')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Composite Model Fitting using curve_fit and MCMC')
            plt.legend()
            plt.show()


	# Fit the composite model using curve_fit
	p0 = [0.1, 0.5, 0.5, 0.5, 1.5, 0.5, 1.0, 1.0]  # Initial parameter guesses
	fit_params, _ = curve_fit(composite_model, X, y, p0=p0, sigma=y_err)

	# Set up the emcee sampler
	nwalkers = 32
	ndim = len(fit_params)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(X, y, y_err))

	# Burn-in phase
	pos, _, _ = sampler.run_mcmc(np.random.randn(nwalkers, ndim) * 1e-3 + fit_params, 1000)

	# Production phase
	sampler.reset()
	sampler.run_mcmc(pos, 2000, progress=True)

	# Extract the posterior samples
	samples = sampler.get_chain(flat=True)


    # Extract the parameter estimates and uncertainties from the samples
    parameter_estimates = np.median(samples, axis=0)
    parameter_errors = np.percentile(samples, [16, 84], axis=0) - parameter_estimates

    if make_plots:
        plotter(output_fp)

    outputs = []
    # Print parameter estimates and errors
    for i, parameter_name in enumerate(labels):
        estimate = parameter_estimates[i]
        error_lower = parameter_errors[0, i]
        error_upper = parameter_errors[1, i]
        outputs.append([estimate, error_lower, error_upper])
    

