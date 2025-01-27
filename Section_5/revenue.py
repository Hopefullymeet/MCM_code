import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

data_path = 'data.csv'
data = pd.read_csv(data_path)

def revenue_model(inputs, beta_0, beta_1, beta_2):
    TourNum, TaxRate = inputs
    return beta_0 * (TourNum ** beta_1) * (TaxRate ** beta_2)

TourNum = data['Visitor_Number'].values
TaxRate = data['Bed_Tax_%'].values
Revenue = data['Total_Income_million'].values

valid_indices = ~np.isnan(TourNum) & ~np.isnan(TaxRate) & ~np.isnan(Revenue)
TourNum = TourNum[valid_indices]
TaxRate = TaxRate[valid_indices]
Revenue = Revenue[valid_indices]

valid_indices = ~np.isinf(TourNum) & ~np.isinf(TaxRate) & ~np.isinf(Revenue)
TourNum = TourNum[valid_indices]
TaxRate = TaxRate[valid_indices]
Revenue = Revenue[valid_indices]

params, covariance = curve_fit(revenue_model, (TourNum, TaxRate), Revenue, p0=[1, 1, 1])

beta_0, beta_1, beta_2 = params

TourNum_range = np.linspace(min(TourNum), max(TourNum), 100)
TaxRate_range = np.linspace(min(TaxRate), max(TaxRate), 100)
TourNum_grid, TaxRate_grid = np.meshgrid(TourNum_range, TaxRate_range)
Revenue_grid = revenue_model((TourNum_grid, TaxRate_grid), beta_0, beta_1, beta_2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(TourNum, TaxRate, Revenue, color='#154a08', label='Actual Revenue', s=20)

ax.plot_surface(TourNum_grid, TaxRate_grid, Revenue_grid, color='lightgreen', alpha=0.7)

ax.set_title('Fitted Revenue Model', fontsize=16)
ax.set_xlabel('TourNum', fontsize=12)
ax.set_ylabel('TaxRate (%)', fontsize=12)
ax.set_zlabel('Total Income (million)', fontsize=12)
ax.legend()

plt.savefig('revenue_model_3d_plot.png')
plt.show()

model_params = pd.DataFrame({
    'Parameter': ['beta_0', 'beta_1', 'beta_2'],
    'Value': [beta_0, beta_1, beta_2]
})

model_params.to_csv('fitted_model_parameters.csv', index=False)
