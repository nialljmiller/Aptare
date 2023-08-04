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


# Savitzky-Golay and KNN Smooth: A Pirate's Guide

Ahoy there, landlubbers! If ye find yerself strugglin' with noisy data that be jumpin' and divin' like a kraken, fear not! With the help of the Savitzky-Golay and K-Nearest Neighbors (KNN) smoothing techniques, we shall tame those unruly waves and find the buried treasures hidden within!

## Installations

Before we embark on this voyage, make sure ye have installed the necessary libraries: `numpy` and `matplotlib`. If ye haven't installed 'em yet, run the following command:

```
pip install numpy matplotlib
```

## Savitzky-Golay Smooth

Our first mate, Savitzky-Golay, be a powerful smoothing filter that helps us sail through the rough seas of noisy data. It be a polynomial regression-based filter that can smooth yer data and reveal the trends hidd'n beneath.

To use Savitzky-Golay, import the `savitzky_golay` function and pass it yer X and Y data along with the window size and polynomial order:

```python
from your_code_file import savitzky_golay

# Sample XY data (replace with yer own data)
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0.1, 0.5, 0.2, 0.8, 0.9, 0.4]

# Set the window size and polynomial order
window_size = 3
poly_order = 2

# Apply Savitzky-Golay smoothing
smoothed_data = savitzky_golay(x_data, y_data, window_size, poly_order)
```

Ah, ye be seein' the magic of Savitzky-Golay in action! The `smoothed_data` will be a beautiful, smoother version of yer original `y_data`. But hold on, there be more!

## K-Nearest Neighbors (KNN) Smooth

Sometimes, the seas be too treacherous, and the Savitzky-Golay may falter. In such cases, we need the aid of our trusty K-Nearest Neighbors (KNN) to navigate the choppy waters.

KNN Smooth be a modified version of the original KNN algorithm. It be like sailors sharing their tales with each other and smoothing out the rough edges. Ye can use the `knn_smooth` function to apply this magic:

```python
from your_code_file import knn_smooth

# Sample XY data (replace with yer own data)
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0.1, 0.5, 0.2, 0.8, 0.9, 0.4]

# Set the number of nearest neighbors to consider
k_neighbors = 3

# Apply KNN smoothing
smoothed_data_knn = knn_smooth(x_data, y_data, k=k_neighbors)
```

Arrr, KNN Smooth to the rescue! The `smoothed_data_knn` be the KNN-smoothed version of yer original `y_data`, showin' a different perspective on yer data.

## Delta Detector: Unearthin' Significant Changes

Now that ye be familiar with the powers of Savitzky-Golay and KNN Smooth, we be ready to sail towards our treasure—significant changes in the data!

Use the `delta_detector` function to detect the significant changes in yer data:

```python
from your_code_file import delta_detector

# Sample XY data (replace with yer own data)
x_data = [0, 1, 2, 3, 4, 5]
y_data = [0.1, 0.5, 0.2, 0.8, 0.9, 0.4]

# Set the window size, polynomial order, and threshold for detecting changes
window_size = 3
poly_order = 2
threshold = 0.2

# Detect significant changes
smoothed_data, significant_changes = delta_detector(x_data, y_data, window_size, poly_order, threshold)
```

Arrr, behold! The `smoothed_data` be the smoothed version of yer `y_data`, while `significant_changes` be the indices where significant changes be detected. Plot them on a chart to visualize the changes:

```python
import matplotlib.pyplot as plt

# Plot the original data and the smoothed data with significant changes
plt.plot(x_data, y_data, label='Original Data', marker='o')
plt.plot(x_data, smoothed_data, label='Smoothed Data', linestyle='dashed')
plt.scatter(x_data[significant_changes], y_data[significant_changes], color='red', label='Significant Changes', marker='x')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Significant Changes in the Data')
plt.show()
```

Shiver me timbers! Ye now be seein' the significant changes marked in red, revealin' the treasures hidden in the depths of the data!

## A Word of Caution

Beware, matey! When usin' Savitzky-Golay and KNN Smooth, the window size and polynomial order be crucial. A too large window may smooth out important details, and a too small window may leave ye sailin' on choppy waters. Experiment wisely!

## Conclusion

Ye be now equipped with the knowledge to harness the powers of Savitzky-Golay and KNN Smooth, detectin' significant changes like a seasoned pirate of the data seas. Sail forth, me hearty, and may yer data explorations be fruitful and yer code always bug-free!

Fair winds and following seas! Yo ho, yo ho! 🏴‍☠️

## Contributing

Contributions to Aptare are welcome! If you find any issues or have ideas for improvements, feel free to raise an issue or submit a pull request. Check out the contribution guidelines [link to contribution guidelines] for more information.

## License

Aptare is open-source software and is distributed under the MIT License. See the LICENSE file [link to license file] for more details.

## Contact

For any questions or inquiries, you can reach out to the project maintainer [maintainer's email].

---

Next, please provide me with the names of the important functions you want to be described, and I'll gladly write the respective sections for each function in the README.
