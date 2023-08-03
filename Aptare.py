import numpy as np
import celerite
from celerite import terms
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#       _ _   _           _                                        _                                 _                       
#      (_) | | |         | |                                      | |                               (_)                      
#  __ _ _| |_| |__  _   _| |__     ___ __ _ _ __    ___ _   _  ___| | __  _ __ ___  _   _  __      ___ _ __   __ _ _   _ ___ 
# / _` | | __| '_ \| | | | '_ \   / __/ _` | '_ \  / __| | | |/ __| |/ / | '_ ` _ \| | | | \ \ /\ / / | '_ \ / _` | | | / __|
#| (_| | | |_| | | | |_| | |_) | | (_| (_| | | | | \__ \ |_| | (__|   <  | | | | | | |_| |  \ V  V /| | | | | (_| | |_| \__ \
# \__, |_|\__|_| |_|\__,_|_.__/   \___\__,_|_| |_| |___/\__,_|\___|_|\_\ |_| |_| |_|\__, |   \_/\_/ |_|_| |_|\__, |\__,_|___/
#  __/ |                                                                             __/ |                    __/ |          
# |___/                                                                             |___/                    |___/           




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



def fit_trap_model(phase, mag, mag_error, rise_slope = 'Linear', output_fp = None):

    x_sort = np.argsort(phase)
    x_data = phase[x_sort]
    y_data = mag[x_sort]
    y_error = mag_error[x_sort]

    # Example usage with curve fitting
    Q1,Q5,Q25,Q99 = np.percentile(y, [1,5,50,99])
    base_line = Q5
    wave_depth = abs(Q99-Q1)
    transit_time = 0.3
    input_time = 0.1
    output_time = 0.1



    if rise_slope == 'Tanh' or 'Than':
        rise_slope_func = tanh_rise_slope
    if rise_slope == 'Sigmoid':
        rise_slope_func = sigmoid_rise_slope
    if rise_slope == 'Linear':
        rise_slope_func=linear_rise_slope
    else:
        rise_slope_func=linear_rise_slope
    # Perform curve fitting using scipy.optimize.curve_fit with weights
    p0 = [base_line, wave_depth, transit_time, input_time, output_time, rise_slope_func]  # Initial guess for parameters
    popt, pcov = curve_fit(trapezoid_waveform, x_data, y_data, p0=p0, sigma=y_error, absolute_sigma=True, maxfev = 9999999)

    # Extract the fitted parameters
    fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time = popt

    # Generate the fitted waveform using the fitted parameters
    fitted_waveform = trapezoid_waveform(x_data, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time)

    # Generate the fitted waveform using the fitted parameters
    plt_x_fitted = np.linspace(0, 1, 1000)  # Higher sampling rate for visualization
    plt_fitted_waveform = trapezoid_waveform(plt_x_fitted, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time)



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



    
    
    


def sine_fit(self, mag, time, period, IO = False, output_fp = '.'):
    try:
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

     
    
    
    

def spline_fit(mag, magerr, time, IO = False, output_fp = '.'):
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

    try:
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
