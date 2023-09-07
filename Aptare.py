import numpy as np
import celerite
from celerite import terms
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
import classic_outburst
import GP_outburst
#       _ _   _           _                                        _                                 _                       
#      (_) | | |         | |                                      | |                               (_)                      
#  __ _ _| |_| |__  _   _| |__     ___ __ _ _ __    ___ _   _  ___| | __  _ __ ___  _   _  __      ___ _ __   __ _ _   _ ___ 
# / _` | | __| '_ \| | | | '_ \   / __/ _` | '_ \  / __| | | |/ __| |/ / | '_ ` _ \| | | | \ \ /\ / / | '_ \ / _` | | | / __|
#| (_| | | |_| | | | |_| | |_) | | (_| (_| | | | | \__ \ |_| | (__|   <  | | | | | | |_| |  \ V  V /| | | | | (_| | |_| \__ \
# \__, |_|\__|_| |_|\__,_|_.__/   \___\__,_|_| |_| |___/\__,_|\___|_|\_\ |_| |_| |_|\__, |   \_/\_/ |_|_| |_|\__, |\__,_|___/
#  __/ |                                                                             __/ |                    __/ |          
# |___/                                                                             |___/                    |___/           

def min_max_norm(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
	

def z_norm(arr):
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    normalized_arr = (arr - mean_val) / std_val
    return normalized_arr

def phaser(self, time, period):
    # this is to mitigate against div 0
    if period == 0:
        period = 1 
    phase = np.array(time) * 0.0
    for i in range(0, len(time)):
         phase[i] = (time[i])/period - np.floor((time[i])/period)
         if (phase[i] >= 1):
           phase[i] = phase[i]-1.
         if (phase[i] <= 0):
           phase[i] = phase[i]+1.
    return phase

def calculate_rmse(y_observed, y_predicted):
    """
    Calculates the root mean square error (RMSE) between observed and predicted values.

    Args:
        y_observed (numpy.ndarray): Observed values.
        y_predicted (numpy.ndarray): Predicted values.

    Returns:
        float: Root mean square error.
    """
    residuals = y_observed - y_predicted
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse

def calculate_weighted_rmse(y_observed, y_predicted, uncertainties):
    """
    Calculates the weighted root mean square error (RMSE) between observed and predicted values
    using the uncertainties (errors) associated with the observed values.

    Args:
        y_observed (numpy.ndarray): Observed values.
        y_predicted (numpy.ndarray): Predicted values.
        uncertainties (numpy.ndarray): Uncertainties (errors) associated with the observed values.

    Returns:
        float: Weighted root mean square error.
    """
    squared_residuals = (y_observed - y_predicted)**2
    weighted_squared_residuals = squared_residuals / uncertainties**2
    weighted_rmse = np.sqrt(np.sum(weighted_squared_residuals) / len(y_observed))
    return weighted_rmse


def squared_exp_kernel(log_sigma, log_rho):
    kernel = terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
    return kernel

def log_likelihood(params, x, y, yerr):
    log_sigma, log_rho = params
    kernel = squared_exp_kernel(log_sigma, log_rho)
    gp = celerite.GP(kernel)
    gp.compute(x, yerr)
    return gp.log_likelihood(y)



    
    
    
    
    

def linear_rise_slope(t, depth, transit_time, input_time):
    """
    Calculates the linear rise slope.

    Args:
        t (float): Time value.
        depth (float): The depth or amplitude of the waveform.
        transit_time (float): The time at maximum depth.
        input_time (float): The time at which the waveform starts to rise.

    Returns:
        float: The rise slope value at time t.
    """
    return depth * (t - input_time) / transit_time

def sigmoid_rise_slope(t, depth, transit_time, input_time):
    """
    Calculates the rise slope using a sigmoid function.

    Args:
        t (float): Time value.
        depth (float): The depth or amplitude of the waveform.
        transit_time (float): The time at maximum depth.
        input_time (float): The time at which the waveform starts to rise.

    Returns:
        float: The rise slope value at time t.
    """
    return depth / (1 + np.exp(-2 * (t - input_time) / transit_time))

def tanh_rise_slope(t, depth, transit_time, input_time):
    """
    Calculates the rise slope using a tanh function.

    Args:
        t (float): Time value.
        depth (float): The depth or amplitude of the waveform.
        transit_time (float): The time at maximum depth.
        input_time (float): The time at which the waveform starts to rise.

    Returns:
        float: The rise slope value at time t.
    """
    return depth * np.tanh((t - input_time) / transit_time)

def trapezoid_waveform(x, base, depth, transit_time, input_time, output_time, rise_slope_func=linear_rise_slope):
    """
    Defines a trapezoidal waveform based on the given parameters.

    Args:
        x (numpy.ndarray): The x values.
        base (float): The base line of the waveform.
        depth (float): The depth or amplitude of the waveform.
        transit_time (float): The time at maximum depth.
        input_time (float): The time at which the waveform starts to rise.
        output_time (float): The time at which the waveform starts to fall.
        rise_slope_func (function): The function to calculate the rise slope.
                                    Can be linear_rise_slope, sigmoid_rise_slope, or tanh_rise_slope.

    Returns:
        y (numpy.ndarray): The corresponding waveform values.
    """
    y = np.zeros_like(x)

    # Calculate the rising slope using the provided function
    rise_slope = rise_slope_func(x, depth, transit_time, input_time)

    # Generate the waveform
    for i in range(len(x)):
        if x[i] < input_time:  # Input time range (flat base line)
            y[i] = base
        elif x[i] < input_time + transit_time:  # Rising edge range
            # Using the specified rise slope function for the rising edge
            y[i] = base + rise_slope[i]
        elif x[i] < x.max() - output_time:  # Flat top range
            # Waveform stays at maximum depth during the flat top range
            y[i] = base + depth
        else:  # Falling edge range
            # If falling edge is not required to fall, just use the last depth value
            if not output_time:
                y[i] = base + depth
            else:
                # Linearly decreasing waveform during the falling edge
                y[i] = base + depth + (x[i] - (x.max() - output_time)) * (-depth / output_time)

    return y





def savitzky_golay(x, y, window_size, order, deriv=0, rate=1):
    """
    Apply Savitzky-Golay filter on the data.

    Parameters:
        x (array-like): Input data's X-axis values.
        y (array-like): Input data's Y-axis values.
        window_size (int): The size of the window used for the Savitzky-Golay filter.
        order (int): The polynomial order used for the Savitzky-Golay filter.
        deriv (int): The order of derivative. Default is 0 (no derivative).
        rate (int): The sampling rate of the data. Default is 1.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed Y-axis data.

    Note:
        - The input data (x and y) should be sorted in ascending order of X-axis values.
        - The X-axis values should be normalized before passing to this function.
        - The Y-axis values should also be normalized before passing to this function.
    """
    half_window = (window_size - 1) // 2
    order_range = range(order + 1)
    coeffs = [(-1)**k * np.math.factorial(order) // (np.math.factorial(k) * np.math.factorial(order - k)) for k in order_range]
    kernel = np.outer(coeffs, np.ones(window_size))

    # Interpolate the data to a regular grid
    x_regular = np.linspace(x[0], x[-1], len(x))
    y_regular = np.interp(x_regular, x, y)

    smoothed = np.convolve(y_regular, kernel[deriv], mode='same')
    smoothed /= rate**deriv

    # Interpolate the smoothed data back to the original X-axis values
    smoothed_at_original_x = np.interp(x, x_regular, smoothed)

    return smoothed_at_original_x


def knn_smooth(x, y, k=5, output_length=None):
    """
    Apply modified K-Nearest Neighbors smoothing on the data with stochastic sampling.

    Parameters:
        x (array-like): Input data's X-axis values.
        y (array-like): Input data's Y-axis values.
        k (int): The number of nearest neighbors to consider for smoothing. Default is 5.
        output_length (int): Length of the output smoothed data. If None, uses the length of input data.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed Y-axis data.

    Note:
        - The input data (x and y) should be sorted in ascending order of X-axis values.
        - The X-axis values should be normalized before passing to this function.
        - The Y-axis values should also be normalized before passing to this function.
    """
    if output_length is None:
        output_length = len(x)

    smoothed_x = np.zeros(output_length)
    smoothed_y = np.zeros(output_length)

    for i in range(output_length):
        target_x = (i / (output_length - 1)) * (x[-1] - x[0]) + x[0]  # Interpolate target x value
        distances = np.abs(x - target_x)
        sorted_indices = np.argsort(distances)
        neighbors = y[sorted_indices[:k]]

        # Compute weights based on distance from the target point
        weights = 1.0 / (1.0 + distances[sorted_indices[:k]])

        # Calculate weighted average of neighbors
        smoothed_x[i] = target_x
        smoothed_y[i] = np.sum(weights * neighbors) / np.sum(weights)

    return smoothed_x, smoothed_y


def delta_detector(x, y, window_size=11, poly_order=3, threshold=0.1, k=5, method='sav_gol'):
    """
    Detect significant changes in a dataset using a filter.

    Parameters:
        x (array-like): Input data's X-axis values.
        y (array-like): Input data's Y-axis values.
        window_size (int): The size of the window used for the Savitzky-Golay filter. Default is 11.
        poly_order (int): The polynomial order used for the Savitzky-Golay filter. Default is 3.
        threshold (float): Threshold to identify significant changes in the smoothed data. Default is 0.1.
        k (int): The number of nearest neighbors to consider for KNN smoothing. Default is 5.
        method (str): Smoothing method ('sav_gol' or 'knn'). Default is 'sav_gol'.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed Y-axis data.
        significant_changes (numpy.ndarray): X-axis values of significant changes in the data.

    Note:
        - The input data (x and y) should be sorted in ascending order of X-axis values.
        - The X-axis values should be normalized before passing to this function.
        - The Y-axis values should also be normalized before passing to this function.
    """
    x_sort = np.argsort(x)
    x = x[x_sort]
    y = y[x_sort]

    x_data = min_max_norm(x)
    y_data = min_max_norm(y)
    
    if method.lower() in ['knn', 'k-nearest neighbors']:
        y_data_smoothed = knn_smooth(x_data, y_data, k=k)
    else:
        y_data_smoothed = savitzky_golay(x_data, y_data, window_size, poly_order)

    significant_changes = np.where(np.abs(np.gradient(y_data_smoothed)) > threshold)[0]
    return y_data_smoothed, x[significant_changes]



def gp_traps(x,y,make_plots = False, output_fp = None):
	GP_outburst.GP_trapfit(x,y, make_plots, output_fp)



def classic_traps(x,y,make_plots = False, output_fp = None):
	classic_outburst.classic_trapfit(x,y, make_plots, output_fp)




def fit_trap_model(phase, mag, mag_error, rise_slope = 'Linear', output_fp = None, norm_x = False, norm_y = False, initial_guess = [0.3,0.1,0.1], do_MCMC = False):
       
    # Define the log likelihood function for the MCMC fitting
    def trap_log_likelihood(params, x, y, y_err):
        model = trapezoid_waveform_partial(x, *params)
        return -0.5 * np.sum(((y - model) / y_err) ** 2)

    # Define priors for the parameters. This is where you can set your initial guesses and ranges.
    def trap_log_prior(params):
        base_line, wave_depth, transit_time, input_time, output_time = params

        # Define uniform priors for each parameter
        if (0 < wave_depth < 1000) and (0 < transit_time < 1000) and (0 < input_time < 1000) and (0 < output_time < 1000):
            return 0.0

        return -np.inf

    # Combine the log likelihood and log prior to get the log posterior
    def trap_log_probability(params, x, y, y_err):
        lp = trap_log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + trap_log_likelihood(params, x, y, y_err)

    x_sort = np.argsort(phase)
    x_data = phase[x_sort]
    y_data = mag[x_sort]
    y_error = mag_error[x_sort]
    
    if norm_x:
        x_data = min_max_norm(x_data)
    if norm_y:        
        y_error = y_error/y_data
        y_data = min_max_norm(y_data)


    # Example usage with curve fitting
    Q1,Q5,Q25,Q99 = np.percentile(y_data, [1,5,50,99])
    base_line = Q5
    wave_depth = abs(Q99-Q1)
    transit_time = initial_guess[0]
    input_time = initial_guess[1]
    output_time = initial_guess[2]



    if rise_slope == 'Tanh' or 'Than':
        rise_slope_func = tanh_rise_slope
    if rise_slope == 'Sigmoid':
        rise_slope_func = sigmoid_rise_slope
    if rise_slope == 'Linear':
        rise_slope_func=linear_rise_slope
    else:
        rise_slope_func=linear_rise_slope

    # Create a partial function with fixed arguments (rise_slope_func)
    trapezoid_waveform_partial = partial(trapezoid_waveform, rise_slope_func=rise_slope_func)

    # Perform curve fitting using scipy.optimize.curve_fit with weights
    p0 = [base_line, wave_depth, transit_time, input_time, output_time]  # Initial guess for parameters
    popt, pcov = curve_fit(trapezoid_waveform_partial, x_data, y_data, p0=p0, sigma=y_error, absolute_sigma=True, maxfev = 9999999)

    # Extract the fitted parameters
    fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time = popt

    if do_MCMC: 
        
        # Perform MCMC fitting
        nwalkers = 100
        ndim = 5
        nsteps = 5000

        # Initialize the walkers around the initial guess
        initial_guess = popt
        pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
        # Set up the emcee sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, trap_log_probability, args=(x_data, y_data, y_error))
        # Run the MCMC sampler
        sampler.run_mcmc(pos, nsteps, progress=True)
        # Extract the fitted parameters from the samples
        burn_in = 1000  # You may need to adjust this value depending on the convergence of the chains
        samples = sampler.get_chain(discard=burn_in, flat=True)
        # Extract the fitted parameters from the samples
        fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time = np.median(samples, axis=0)
  
    # Generate the fitted waveform using the fitted parameters
    fitted_waveform = trapezoid_waveform_partial(x_data, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time)

    # Generate the fitted waveform using the fitted parameters
    plt_x_fitted = np.linspace(0, 1, 1000)  # Higher sampling rate for visualization
    plt_fitted_waveform = trapezoid_waveform_partial(plt_x_fitted, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time)


    rmse = calculate_rmse(y_data, fitted_waveform)
    weighted_rmse = calculate_weighted_rmse(y_data, fitted_waveform, y_error)

    if output_fp != None:
        # Plotting the original data, the noisy data, and the fitted waveform
   
        # Plotting the original data, the noisy data, and the fitted waveform
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Top panel - Original data and fitted waveform
        ax1.errorbar(x_data, y_data, yerr=y_error, fmt='.', alpha=0.7, label='Observed Data', c = 'k')
        ax1.plot(plt_x_fitted, plt_fitted_waveform, label='Fitted Trapezoid', c = 'g')
        ax1.errorbar(x_data + 1, y_data, yerr=y_error, fmt='.', alpha=0.7, c = 'black')
        ax1.plot(plt_x_fitted + 1, plt_fitted_waveform, c = 'green')
        ax1.invert_yaxis()
        ax1.set_ylabel('Mag')
        ax1.legend()
        ax1.grid(True)

        # Bottom panel - Residuals
        residuals = y_data - fitted_waveform
        ax2.plot(x_data, residuals, '.', alpha=0.7, c = 'blue')
        ax2.plot(x_data + 1, residuals, '.', alpha=0.7, c = 'blue')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)

        # Set the x-axis label only on the bottom plot
        plt.xlabel('Phase')

        # Set the title with RMSE and weighted RMSE
        plt.suptitle('Trapezoid Fit\nRMSE: {:.4f} - Weighted RMSE: {:.4f}'.format(rmse, weighted_rmse))

        # Adjust the spacing between subplots
        plt.tight_layout()
        plt.savefig(output_fp, dpi = 300)

    return rmse, weighted_rmse, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time
    
    
    
    

def phil_rise_slope_norm(time, peak, tau, t_half, base_mag, decay_mag):
    """
    Calculates the FUor-iness of a light curve using an exponential function and linear decay.

    Args:
        time (array): Array of dates or times.
        peak (float): The depth or amplitude of the waveform.
        tau (float): 1/2 of the time the burst ends.
        t_half (float): Time at halfway point of eruptive event.
        base_mag (float): Quiescent magnitude.
        decay_mag (float): Final magnitude.

    Returns:
        list: List of calculated values for the light curve.
    """
    line = []
    decay_time = time.max() - (t_half + 2 * tau)
    grad = (decay_mag - peak) / decay_time

    for t in time:
        if t < t_half:
            line.append(base_mag + ((base_mag + peak) / (1 + np.exp(-1 * (t - t_half) / tau))))
        elif t <= (t_half + 2 * tau):
            line.append(base_mag + ((base_mag + peak) * (0.5 + 0.5 * ((t - t_half) / (2 * tau)))))
        else:
            line.append((grad * (time.max() - t)) + decay_mag)

    return line





def fit_FUor_model(phase, mag, mag_error, output_fp=None, norm_x=False, norm_y=False, initial_guess=[0.1, 0.3], do_MCMC=False):

    # Sort the data
    x_sort = np.argsort(phase)
    x_data = phase[x_sort]
    y_data = mag[x_sort]
    y_error = mag_error[x_sort]

    # Normalize data if requested
    if norm_x:
        x_data = min_max_norm(x_data)
    if norm_y:
        y_error = y_error / y_data
        y_data = min_max_norm(y_data)

    # Example usage with curve fitting
    Q1, Q5, Q25, Q99 = np.percentile(y_data, [1, 5, 50, 99])
    base_mag = Q5
    peak = abs(Q99 - Q1)
    tau = initial_guess[0]
    t_half = initial_guess[1]
    decay_mag = y_data[-1]

    p0 = [peak, tau, t_half, base_mag, decay_mag]  # Initial guess for parameters
    popt, pcov = curve_fit(phil_rise_slope_norm, x_data, y_data, p0=p0, sigma=y_error, absolute_sigma=True, maxfev=9999999)
    fitted_peak, fitted_tau, fitted_t_half, fitted_base_mag, fitted_decay_mag = popt

    if do_MCMC:

        # Define the log likelihood function for MCMC
        def ln_likelihood(theta, x, y, y_err):
            peak, tau, t_half, base_mag, decay_mag = theta
            model = phil_rise_slope_norm(x, peak, tau, t_half, base_mag, decay_mag)
            chi2 = np.sum(((y - model) / y_err) ** 2)
            return -0.5 * chi2

        # Define the log prior function (uniform priors in this example)
        def ln_prior(theta):
            peak, tau, t_half, base_mag, decay_mag = theta
            if 0.0 < peak < 10.0 and 0.0 < tau < 10.0 and 0.0 < t_half < 10.0 and 0.0 < base_mag < 10.0 and 0.0 < decay_mag < 10.0:
                return 0.0
            return -np.inf

        # Define the log probability function
        def ln_probability(theta, x, y, y_err):
            lp = ln_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + ln_likelihood(theta, x, y, y_err)

        # Perform MCMC fitting
        nwalkers = 100
        ndim = len(p0)
        nsteps = 5000

        # Initialize the walkers around the initial guess
        initial_guess = popt
        pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        # Set up the emcee sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_probability, args=(x_data, y_data, y_error))

        # Run the MCMC sampler
        sampler.run_mcmc(pos, nsteps, progress=True)

        # Extract the fitted parameters from the samples
        burn_in = 1000  # You may need to adjust this value depending on the convergence of the chains
        samples = sampler.get_chain(discard=burn_in, flat=True)

        # Extract the fitted parameters from the samples
        fitted_peak, fitted_tau, fitted_t_half, fitted_base_mag, fitted_decay_mag = np.median(samples, axis=0)

    fitted_waveform = phil_rise_slope_norm(x_data, fitted_peak, fitted_tau, fitted_t_half, fitted_base_mag, fitted_decay_mag)           

    # Generate the fitted waveform using the fitted parameters
    plt_x_fitted = np.linspace(0, 1, 1000)  # Higher sampling rate for visualization
    plt_fitted_waveform = phil_rise_slope_norm(plt_x_fitted, fitted_peak, fitted_tau, fitted_t_half, fitted_base_mag, fitted_decay_mag)

    # Calculate RMSE and weighted RMSE
    rmse = np.sqrt(np.mean((y_data - fitted_waveform)**2))
    weighted_rmse = np.sqrt(np.mean(((y_data - fitted_waveform) / y_error)**2))

    if output_fp is not None:
        # Plotting the original data, the noisy data, and the fitted waveform
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Top panel - Original data and fitted waveform
        ax1.errorbar(x_data, y_data, yerr=y_error, fmt='.', alpha=0.7, label='Observed Data', c='k')
        ax1.plot(plt_x_fitted, plt_fitted_waveform, label='Fitted Trapezoid', c='g')
        ax1.errorbar(x_data + 1, y_data, yerr=y_error, fmt='.', alpha=0.7, c='black')
        ax1.plot(plt_x_fitted + 1, plt_fitted_waveform, c='green')
        ax1.set_ylabel('Mag')
        ax1.legend()
        ax1.grid(True)

        # Bottom panel - Residuals
        residuals = y_data - fitted_waveform
        ax2.plot(x_data, residuals, '.', alpha=0.7, c='blue')
        ax2.plot(x_data + 1, residuals, '.', alpha=0.7, c='blue')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)

        # Set the x-axis label only on the bottom plot
        plt.xlabel('Phase')

        # Set the title with RMSE and weighted RMSE
        plt.suptitle('Trapezoid Fit\nRMSE: {:.4f} - Weighted RMSE: {:.4f}'.format(rmse, weighted_rmse))

        # Adjust the spacing between subplots
        plt.tight_layout()
        plt.savefig(output_fp, dpi=300)

    return rmse, weighted_rmse, fitted_peak, fitted_tau, fitted_t_half, fitted_base_mag, fitted_decay_mag


def calculate_boxiness(phase, mag, mag_err):
    # Normalize phase to range [0, 1]
    #phase /= np.max(phase)

    # Sort the phase, magnitude, and magnitude error arrays based on the phase values
    sorted_indices = np.argsort(phase)
    phase_sorted = phase[sorted_indices]
    mag_sorted = mag[sorted_indices]
    mag_err_sorted = mag_err[sorted_indices]

    # Set up the MCMC sampling
    ndim = 2
    nwalkers = 32
    nsteps = 1000
    pos = [np.random.randn(ndim) for _ in range(nwalkers)]

    # Run the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(phase_sorted, mag_sorted, mag_err_sorted))
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Extract the posterior samples
    samples = sampler.get_chain(discard=100, flat=True)

    # Calculate the median of each parameter
    median_params = np.median(samples, axis=0)

    # Generate the GP with the median parameters
    kernel = squared_exp_kernel(*median_params)
    gp = celerite.GP(kernel)
    gp.compute(phase_sorted, mag_err_sorted)

    # Calculate the residuals
    residuals = mag_sorted - gp.predict(mag_sorted, phase_sorted)[0]

    # Calculate the boxiness metric
    #boxiness_metric = np.corrcoef(residuals, np.ones_like(residuals))[0, 1]

    return np.mean(residuals)



    
    
    


def fit_sine_model(self, mag, time, period, IO = False, output_fp = '.'):
    def sinus(x, A, B, C): # this is your 'straight line' y=f(x)
        return (A * np.sin((2.*np.pi*x)+C))+B

    y = np.array(mag)        # to fix order when called, (better to always to mag, time) CONSISTENCY!!!!!!!!!)
    x = phaser(time, period)
    popt, pcov = curve_fit(sinus, x, y, bounds=((true_amplitude*0.3, mag_avg*0.3, -2), (true_amplitude*3.0, mag_avg*3, 2)))#, method = 'lm') # your data x, y to fit
    
    y_fit = sinus(x, popt[0], popt[1], popt[2])
    #compare
    y_diff = y - y_fit
    #residual sum of squares
    ss_res = np.sum((y_diff) ** 2)
    #total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)        #coefficient of determination
    
    amplitude = popt[0]
    offset = popt[1]
    phase_shift = popt[2]
    if IO:

        plt.clf()
        
        text_font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 8, }
        text_pos = 0.001
        sort = np.argsort(x)
        m = np.array(y)[sort]            
        
        plt.title('Amplitude = '+str(round(popt[0], 3))+'  Offset = '+str(round(popt[1], 3))+'  Phase Shift = '+str(round(popt[2], 3)))
                        
        x_line = np.arange(0, 1.01, 0.01)
        plt.plot(x, y, 'rx', markersize=4, label = 'Before')
        plt.plot(x+1., y, 'rx', markersize=4)

        plt.plot(x_line, sinus(x_line, popt[0], popt[1], popt[2]), 'k', label="$R^2\, =\, $"+str(round(r2,3)))
        plt.plot(x_line+1., sinus(x_line, popt[0], popt[1], popt[2]), 'k')

        plt.plot(x, y_diff+np.median(y), 'b+', label = 'After')
        plt.plot(x+1., y_diff+np.median(y), 'b+')
        
        plt.xlabel('Phase')
        plt.ylabel('Magnitude [mag]')
        plt.grid()
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(output_fd + '_sine.png', format = 'png', dpi = image_dpi)

    return r2, amplitude, offset, phase_shift

     
    
    
    

def fit_spline_model(mag, magerr, time, IO = False, output_fp = '.'):

    def sl(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B
            
    y = np.array(mag)    
    yerr = np.array(magerr)
    x = np.array(time)

    res = 10
    rq50 = np.empty(res)
    rq25 = np.empty(res)
    rq75 = np.empty(res)
    Q75, Q25 = np.percentile(y, [75, 25])
    rx = np.linspace(min(x), max(x), res)
    rdelta = (max(x) - min(x))/(2*res)

    ##bin need to have X points
    for i in range(res):
        check = []
        rdelta_temp = rdelta                        
        while len(check) < 1:
            check = np.where((x < rx[i]+rdelta_temp) & (x > rx[i]-rdelta_temp))[0]
            rdelta_temp = rdelta_temp + 0.2*rdelta
        rq50[i] = np.median(y[check])
        try:
            rq75[i], rq25[i] = np.percentile(y[check], [75, 25])
        except:
            rq75[i], rq25[i] = rq75[i-1], rq25[i-1]


    RQ75, RQ25 = np.percentile(rq50, [75, 25])
    RIQR = abs(RQ75 - RQ25)

    
    #if the range of IQR of binned data changes alot when a single bin is removed, its probably transient
    IQRs = []
    for i in range(1,res):
        tq75, tq25 = np.percentile(np.delete(rq50,i), [75, 25])
        IQRs.append(abs(tq75-tq25))
    
    if abs(max(IQRs)-min(IQRs)) > 0.1 * RIQR:
        trans_flag = 1 


    popt, pcov = curve_fit(sl, rx, rq50) # your data x, y to fit
    grad = popt[0]
    intercept = popt[1]
    #generate fit

    y_fit = sl(x, popt[0], popt[1])
    #compare
    y_diff = y - y_fit
    #residual sum of squares
    ss_res = np.sum((y_diff) ** 2)
    #total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    if IO:

        print('\t~~~~~~~SPLINE FIT~~~~~~~~~')
        print("\tCoefficient of determination = ", r2)
        print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~')

        plt.clf()
        x_line = np.arange(min(x), max(x))
        plt.plot(x_line, sl(x_line, grad, intercept), 'k', label="$R^2\, =\, $"+str(r2))
        plt.plot(x, y, 'rx', label = 'Before')
        plt.errorbar(rx, rq50, yerr = (rq75-rq50, rq50-rq25), xerr = rdelta, ecolor = 'g', label = 'Running Median')            
        plt.plot(x, y_diff+np.median(y), 'b+', label = 'After')
        plt.xlabel('Time [JD]')
        plt.ylabel('Magnitude [mag]')
        plt.grid()
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(output_fp + '_spline.png', format = 'png', dpi = image_dpi) 
        plt.clf()

    return r2, grad, intercept
