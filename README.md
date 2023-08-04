!!!README WRITTEN BY CHATGPT!!!
# Aptare - Generic Light Curve Fitting Package

Aptare is a versatile and user-friendly Python package designed to fit light curves using various models and techniques. It allows researchers and data analysts to accurately model and fit light curves from a wide range of sources, including astronomical data, sensor data, and more. Aptare provides functionality for fitting different types of light curves, including spline fits, sinusoids, trapeziums with various rise shapes, and Gaussian Processes (GP) with squared exponential kernels.

## Features

Aptare offers the following key features:

- **Spline Fits:** Aptare can fit light curves using splines, both binned and unbinned. Splines are useful for capturing smooth and flexible representations of light curve data.

- **Sinusoid Fits:** Aptare can model periodic phenomena using sinusoidal functions. This feature is particularly useful when analyzing light curves with periodic patterns, such as those arising from celestial objects.

- **Trapezium Fits:** Aptare allows fitting trapezium-shaped light curves with different rise shapes, such as linear, tanh, and sigmoid. Trapezium fits are valuable for studying transient events and quick variations in light intensity.

- **Gaussian Processes (GP):** Aptare supports fitting light curves using Gaussian Processes with squared exponential kernels. GP models provide a powerful framework for capturing complex correlations and uncertainties in the data.

## Getting Started

To get started with Aptare, follow these steps:

1. **Installation:** Install Aptare using `pip`:

   ```bash
   pip install aptare
   ```

2. **Usage:** Import Aptare in your Python script or notebook and use the appropriate functions to fit your light curves:

## Fit Trap Model - Giddy up your Light Curves!

Yeehaw! The `fit_trap_model` function in Aptare is your trusty steed for wranglin' them trapezoid waveforms! Saddle up, and let's see how this function can fit your data.

### Parameters 🎯

- `phase` 🔍: NumPy array or list - The phase values (x-axis) corresponding to the magnitude data.
- `mag` 🌟: NumPy array or list - The magnitude (y-axis) data of your light curve.
- `mag_error` 🔧: NumPy array or list - The magnitude error (uncertainty) for each data point in the light curve.
- `rise_slope` 📈 (optional) - The type of rise slope function to use for the trapezoid fit. Available options are 'Linear', 'Tanh', 'Sigmoid', and 'Than'. Default is 'Linear'.
- `output_fp` 🖼️ (optional) - The file path to save the output plot showing the fitted waveform and residuals. If not provided, the plot will not be saved.
- `norm_x` 📐 (optional) - Set to `True` to normalize the phase values to the range [0, 1]. Default is `False`.
- `norm_y` 📏 (optional) - Set to `True` to normalize the magnitude values to the range [0, 1]. Default is `False`.
- `initial_guess` 🎯 (optional) - The initial guess for the trapezoid model parameters [transit_time, input_time, output_time]. Default is [0.3, 0.1, 0.1].

### Usage 🤠

```python
from aptare import fit_trap_model

# Example data
phase = [0.1, 0.3, 0.5, 0.7, 0.9]
mag = [10.0, 9.8, 11.2, 9.5, 10.5]
mag_error = [0.1, 0.2, 0.15, 0.12, 0.18]

# Fit the trapezoid waveform to the data
rmse, weighted_rmse, fitted_base, fitted_depth, fitted_transit_time, fitted_input_time, fitted_output_time = fit_trap_model(phase, mag, mag_error, rise_slope='Tanh', output_fp='fit_result.png', norm_x=True, norm_y=True, initial_guess=[0.2, 0.05, 0.05])

print("Root Mean Squared Error (RMSE):", rmse)
print("Weighted RMSE:", weighted_rmse)
print("Fitted Baseline:", fitted_base)
print("Fitted Depth:", fitted_depth)
print("Fitted Transit Time:", fitted_transit_time)
print("Fitted Input Time (Rise Time):", fitted_input_time)
print("Fitted Output Time (Fall Time):", fitted_output_time)
```

### Output 🌄

Y'all can save an output plot by providing the `output_fp` parameter, and it'll show the original data points, the fitted trapezoid waveform, and the residuals after fitting. Giddy up and take a look at the fit_result.png to see how well the trapezoid corralled them data points!

### Let's Ride!

So there you have it, partner! Saddle up and use the `fit_trap_model` function to lasso them trapezoid fits! 🐎 Remember to pass the right data and watch out for them optional parameters to customize your fit. Yeehaw! 🌵🌟🔧📈🎯📐📏🖼️🌄
3. **Documentation:** For more details on the available functions and their usage, please refer to the documentation [link to documentation].


Sure! Here's a small GitHub-style README with instructions on how to use the `savgol_delta_detector` function:

## SavGol Delta Detector

The `savgol_delta_detector` is a Python function designed to detect significant changes in a dataset using the Savitzky-Golay filter. It provides a way to identify regions of interest where the data shows significant deviations from its smoothed trend.

### Dependencies

This function requires the following Python libraries:

- `numpy`: For numerical computations and array manipulations.
- `matplotlib`: For data visualization.

### Function Signature

```python
def savgol_delta_detector(x, y, window_size=11, poly_order=3, threshold=0.1):
    """
    Detect significant changes in a dataset using the Savitzky-Golay filter.

    Parameters:
        x (array-like): Input data's X-axis values.
        y (array-like): Input data's Y-axis values.
        window_size (int): The size of the window used for the Savitzky-Golay filter. Default is 11.
        poly_order (int): The polynomial order used for the Savitzky-Golay filter. Default is 3.
        threshold (float): Threshold to identify significant changes in the smoothed data. Default is 0.1.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed Y-axis data.
        significant_changes (numpy.ndarray): X-axis values of significant changes in the data.

    Note:
        - The input data (x and y) should be sorted in ascending order of X-axis values.
        - The X-axis values should be normalized before passing to this function.
        - The Y-axis values should also be normalized before passing to this function.

    Example:
        # Sample XY data (replace with your actual data)
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data) + np.random.normal(0, 0.1, size=len(x_data))

        # Detect significant changes
        smoothed_data, significant_changes = savgol_delta_detector(x_data, y_data)

    """
```


### Example

```python
import numpy as np
import matplotlib.pyplot as plt
from aptare import savgol_delta_detector

# Sample XY data (replace with your actual data)
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, size=len(x_data))

# Detect significant changes
smoothed_data, significant_changes = savgol_delta_detector(x_data, y_data)

# Plot the original data and the smoothed data with significant changes
plt.plot(x_data, y_data, label='Original Data', alpha=0.5)
plt.plot(x_data, smoothed_data, label='Smoothed Data')
plt.scatter(significant_changes, [y_data[i] for i in significant_changes], color='red', label='Significant Changes')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data and Smoothed Data with Significant Changes')
plt.show()
```


## Contributing

Contributions to Aptare are welcome! If you find any issues or have ideas for improvements, feel free to raise an issue or submit a pull request. Check out the contribution guidelines [link to contribution guidelines] for more information.

## License

Aptare is open-source software and is distributed under the MIT License. See the LICENSE file [link to license file] for more details.

## Contact

For any questions or inquiries, you can reach out to the project maintainer [maintainer's email].

---

Next, please provide me with the names of the important functions you want to be described, and I'll gladly write the respective sections for each function in the README.
