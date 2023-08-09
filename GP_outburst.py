import numpy as np
import matplotlib.pyplot as plt
from celerite.modeling import Model
import celerite
import emcee
import corner


# Define the log likelihood function for emcee
def log_likelihood(params, t, y):
    model.set_parameter_vector(params)
    gp.compute(t)
    return -0.5 * np.sum((y - model.get_value(t)) ** 2 / gp.get_variance(y))

def log_likelihood_err(params, t, y, y_err):
    model.set_parameter_vector(params)
    gp.compute(t)
    y_pred = model.get_value(t)
    resid = y - y_pred
    log_like = -0.5 * np.sum((resid / y_err)**2 + np.log(2 * np.pi * y_err**2))
    return log_like




# Define the log prior for emcee
def log_prior(params):
    # Define appropriate prior distributions for each parameter
    return 0.0  # Flat prior for simplicity

# Define the log probability for emcee
def log_probability(params, t, y):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, y)



# Define a custom kernel for your composite kernel
class CompositeKernel(celerite.terms.Term):
    parameter_names = ("slope_linear", "length_scale_rbf", "amplitude_rq", "length_scale_rq", "alpha_rq", "amplitude_exp", "length_scale_exp", "period_exp")

    def get_real_coefficients(self):
        return np.array([0.0])

    def get_complex_coefficients(self):
        slope_linear = self.slope_linear
        length_scale_rbf = self.length_scale_rbf
        amplitude_rq = self.amplitude_rq
        length_scale_rq = self.length_scale_rq
        alpha_rq = self.alpha_rq
        amplitude_exp = self.amplitude_exp
        length_scale_exp = self.length_scale_exp
        period_exp = self.period_exp

        kernel_value = slope_linear * celerite.terms.RealTerm(1.0, 0.0) + \
                       amplitude_rq * celerite.terms.Matern32Term(length_scale=length_scale_rbf) + \
                       amplitude_rq * celerite.terms.Matern52Term(length_scale=length_scale_rq) + \
                       amplitude_exp * celerite.terms.ExpSine2Term(length_scale=length_scale_exp, period=period_exp)

        return kernel_value






def GP_trapfit(X,y,yerr = None, make_plots = False, output_fp = None):

    def plotter(output_fp = None):
        # Plotting the posterior distributions of parameters
        if output_fp is not None: 
            labels = ["slope_linear", "length_scale_rbf", "amplitude_rq", "length_scale_rq", "alpha_rq", "amplitude_exp", "length_scale_exp", "period_exp"]
            corner.corner(samples, labels=labels, truths=initial_params)
            plt.savefig(output_fp)

            # Predict using the optimized model
            t_pred = np.linspace(0, 5, 500)
            y_pred, pred_var = gp.predict(y, t_pred, return_var=True)

            # Plotting the GP predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, c='r', label='Data')
            plt.plot(t_pred, y_pred, 'b', label='GP Prediction')
            plt.fill_between(t_pred, y_pred - np.sqrt(pred_var), y_pred + np.sqrt(pred_var), alpha=0.2, color='blue')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Gaussian Process Regression using celerite with MCMC')
            plt.legend()
            plt.savefig(output_fp)




    # Initialize the custom kernel
    kernel = CompositeKernel()

    # Set up the celerite GP
    gp = celerite.GP(kernel, fit_mean=False)
    gp.compute(X)


    # Initial parameter guesses
    initial_params = model.get_parameter_vector()





    # Set up the emcee sampler
    nwalkers = 32
    ndim = len(initial_params)
    if yerr == None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(X, y))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_err, args=(X, y, y_err))

    # Burn-in phase
    pos, _, _ = sampler.run_mcmc(np.random.randn(nwalkers, ndim) * 1e-3 + initial_params, 1000)

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
    
    return outputs




