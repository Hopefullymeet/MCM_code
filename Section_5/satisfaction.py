import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

file_path = 'data.csv'
data = pd.read_csv(file_path)

columns = [
    "Visitor_Number",
    "Infrastructure_Investment_%",
    "Resident_Satisfaction",
    "Total_Income_million",
    "Unemployment_Rate",
]
data_cleaned = data[columns].dropna()
data_cleaned.columns = ["TourNum", "I_infr", "Satisfaction", "Income", "Unemployment"]

TourNum_max = data_cleaned["TourNum"].max()
I_infr_base = data_cleaned["I_infr"].mean()
data_cleaned["TourNum_norm"] = data_cleaned["TourNum"] / TourNum_max
data_cleaned["I_infr_norm"] = data_cleaned["I_infr"] / I_infr_base

def income_model(X, omega0, omega1, omega2):
    I_infr_norm, TourNum_norm = X
    return omega0 + omega1 * I_infr_norm + omega2 * TourNum_norm

X_income_data = (
    data_cleaned["I_infr_norm"].values,
    data_cleaned["TourNum_norm"].values,
)
y_income_data = data_cleaned["Income"].values

initial_guess_income = [1, 1, 1]
params_income, _ = curve_fit(income_model, X_income_data, y_income_data, p0=initial_guess_income)

omega0, omega1, omega2 = params_income

def unemployment_model(X, psi0, psi1, psi2):
    I_infr_norm, TourNum_norm = X
    return psi0 - psi1 * I_infr_norm + psi2 * TourNum_norm

y_unemployment_data = data_cleaned["Unemployment"].values

initial_guess_unemployment = [1, 1, 1]
params_unemployment, _ = curve_fit(unemployment_model, X_income_data, y_unemployment_data, p0=initial_guess_unemployment)

psi0, psi1, psi2 = params_unemployment

def satisfaction_model(X, theta1, theta2, theta3):
    I_infr_norm, TourNum_norm = X
    return theta1 * I_infr_norm - theta2 * TourNum_norm + theta3

y_satisfaction_data = data_cleaned["Satisfaction"].values

initial_guess_satisfaction = [1, 1, 1]
params_satisfaction, _ = curve_fit(satisfaction_model, X_income_data, y_satisfaction_data, p0=initial_guess_satisfaction)

theta1, theta2, theta3 = params_satisfaction

print("Income Model Coefficients:")
print("Omega0 (ω0):", omega0)
print("Omega1 (ω1):", omega1)
print("Omega2 (ω2):", omega2)

print("\nUnemployment Model Coefficients:")
print("Psi0 (ψ0):", psi0)
print("Psi1 (ψ1):", psi1)
print("Psi2 (ψ2):", psi2)

print("\nSatisfaction Model Coefficients:")
print("Theta1 (θ1):", theta1)
print("Theta2 (θ2):", theta2)
print("Theta3 (θ3):", theta3)

data_cleaned["Satisfaction_predicted"] = satisfaction_model(X_income_data, theta1, theta2, theta3)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    data_cleaned["TourNum_norm"],
    data_cleaned["I_infr_norm"],
    y_satisfaction_data,
    color='#154a08',
    label='Actual Satisfaction',
    alpha=0.8
)

TourNum_grid, I_infr_grid = np.meshgrid(
    np.linspace(data_cleaned["TourNum_norm"].min(), data_cleaned["TourNum_norm"].max(), 50),
    np.linspace(data_cleaned["I_infr_norm"].min(), data_cleaned["I_infr_norm"].max(), 50),
)
Satisfaction_grid = satisfaction_model(
    (I_infr_grid, TourNum_grid), theta1, theta2, theta3
)
ax.plot_surface(
    TourNum_grid, I_infr_grid, Satisfaction_grid, color='lightgreen', alpha=0.5
)

ax.set_xlabel('TourNum (Normalized)')
ax.set_ylabel('I_infr (Normalized)')
ax.set_zlabel('Resident Satisfaction (S)')
ax.set_title('Resident Satisfaction Model Fit')
ax.legend()

plt.show()

plot_path = 'resident_satisfaction_3d_fit.png'
fig.savefig(plot_path)

print("Satisfaction 3D plot saved at:", plot_path)
