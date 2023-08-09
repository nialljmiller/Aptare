import numpy as np
import matplotlib.pyplot as plt
from celerite.modeling import Model
import celerite

# Simulated data generation
np.random.seed(42)
X = np.sort(5 * np.random.rand(80))
y = np.sin(X) + np.random.normal(0, 0.1, X.shape[0])

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

# Initialize the custom model
model = CompositeModel(slope_linear=0.1, length_scale_rbf=0.5, amplitude_rq=0.5, length_scale_rq=0.5, alpha_rq=1.5, amplitude_exp=0.5, length_scale_exp=1.0, period_exp=1.0)

# Fit the model using celerite
gp = celerite.GP(model, fit_mean=False)
gp.compute(X)

# Optimize the parameters
gp.optimize(X, y)

# Predict using the optimized model
t_pred = np.linspace(0, 5, 500)
y_pred, pred_var = gp.predict(y, t_pred, return_var=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='r', label='Data')
plt.plot(t_pred, y_pred, 'b', label='GP Prediction')
plt.fill_between(t_pred, y_pred - np.sqrt(pred_var), y_pred + np.sqrt(pred_var), alpha=0.2, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Process Regression using celerite')
plt.legend()
plt.show()

