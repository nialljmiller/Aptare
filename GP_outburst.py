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




# Define a custom model class for your composite kernel
class CompositeModel(Model):
    parameter_names = ("slope_linear", "length_scale_rbf", "amplitude_rq", "length_scale_rq", "alpha_rq", "amplitude_exp", "length_scale_exp", "period_exp")

    def get_value(self, t):
        # Interpret parameters
        slope_linear = self.slope_linear
        length_scale_rbf = self.length_scale_rbf
        amplitude_rq = self.amplitude_rq
        length_scale_rq = self.length_scale_rq
        alpha_rq = self.alpha_rq
        amplitude_exp = self.amplitude_exp
        length_scale_exp = self.length_scale_exp
        period_exp = self.period_exp

        # Calculate the composite kernel value
        k_linear = slope_linear * t
        k_rbf = amplitude_rq**2 * np.exp(-0.5 * (t / length_scale_rbf)**2)
        k_rq = amplitude_rq**2 * (1 + (t**2 / (2 * alpha_rq * length_scale_rq**2)))**(-alpha_rq)
        k_exp = amplitude_exp**2 * np.exp(-0.5 * (t / length_scale_exp)**2) * np.cos(2 * np.pi * t / period_exp)

        return k_linear + k_rbf + k_rq + k_exp




def GP_trapfit(X,y, make_plots = False, output_fp = None):


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



    # Initialize the custom model
    model = CompositeModel(slope_linear=0.1, length_scale_rbf=0.5, amplitude_rq=0.5, length_scale_rq=0.5, alpha_rq=1.5, amplitude_exp=0.5, length_scale_exp=1.0, period_exp=1.0)

    # Set up the celerite GP
    gp = celerite.GP(model, fit_mean=False)
    gp.compute(X)

    # Initial parameter guesses
    initial_params = model.get_parameter_vector()

    # Set up the emcee sampler
    nwalkers = 32
    ndim = len(initial_params)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(X, y))

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




