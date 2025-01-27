import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'data.csv'
data = pd.read_csv(file_path)

aligned_data = data[['Visitor_Number', 'Environment_Investment_%',
                    'Climate_Index_Arctic_Oscillation', 'Carbon_Emissions_kt_CO2_',
                    'Glacier_Changes _Mendenhall_Glacier']].dropna()

tour_num = aligned_data['Visitor_Number'].values.reshape(-1, 1)
env_investment = aligned_data['Environment_Investment_%'].values.reshape(-1, 1)
climate_index = aligned_data['Climate_Index_Arctic_Oscillation'].values.reshape(-1, 1)
carbon_emissions = aligned_data['Carbon_Emissions_kt_CO2_'].values.reshape(-1, 1)
glacier_changes = aligned_data['Glacier_Changes _Mendenhall_Glacier'].values.reshape(-1, 1)

X_E = np.hstack((tour_num, env_investment, climate_index))
model_E = LinearRegression().fit(X_E, carbon_emissions)
alpha_coefficients = (model_E.intercept_[0], *model_E.coef_[0])

X_G = np.hstack((carbon_emissions, climate_index))
model_G = LinearRegression().fit(X_G, glacier_changes)
gamma_coefficients = (model_G.intercept_[0], *model_G.coef_[0])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

tour_range = np.linspace(tour_num.min(), tour_num.max(), 50)
env_range = np.linspace(env_investment.min(), env_investment.max(), 50)
tour_grid, env_grid = np.meshgrid(tour_range, env_range)

E_pred = (
    alpha_coefficients[0]
    + alpha_coefficients[1] * tour_grid
    + alpha_coefficients[2] * env_grid
    + alpha_coefficients[3] * climate_index.mean()
)

ax.plot_surface(tour_grid, env_grid, E_pred, alpha=0.5, color='#8CCF7C')
ax.scatter(tour_num, env_investment, carbon_emissions, c='#154a08', s=15, label='Actual Data')

ax.set_xlabel('TourNum', fontsize=12)
ax.set_ylabel('I_env (%)', fontsize=12)
ax.set_zlabel('Carbon Emissions (kt CO2)', fontsize=12)
ax.set_title('3D Fit: Carbon Emissions = f(TourNum, Environment Investment)', fontsize=16)
ax.legend()

plt.savefig('3D_Fit_for_E.png')
plt.show()
plt.close()

fig, ax2 = plt.subplots(figsize=(12, 8))

E_range = np.linspace(carbon_emissions.min(), carbon_emissions.max(), 50)

G_pred = (
    gamma_coefficients[0]
    + gamma_coefficients[1] * E_range
    + gamma_coefficients[2] * climate_index.mean()
)

ax2.plot(E_range, G_pred, label='Fitted Model', color='#8CCF7C')
ax2.scatter(carbon_emissions, glacier_changes, c='#154a08', s=15, label='Actual Data')

ax2.set_xlabel('Carbon Emissions (kt CO2)', fontsize=12)
ax2.set_ylabel('Glacier Changes (Mendenhall Glacier)', fontsize=12)
ax2.set_title('Fit: Glacier Changes = f(Carbon Emissions)', fontsize=16)
ax2.legend()

plt.savefig('Fit_for_G.png')
plt.show()
plt.close()

print("Alpha coefficients (E equation):", [float(f'{c:.6f}') for c in alpha_coefficients])
print("Gamma coefficients (G equation):", [float(f'{c:.6f}') for c in gamma_coefficients])
