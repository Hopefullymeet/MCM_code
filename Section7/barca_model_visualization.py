import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_path = "1.xlsx"
data = pd.ExcelFile(data_path)
sheet_data = data.parse("Sheet1")

filtered_data = sheet_data[(sheet_data["Year"] >= 2012) & (sheet_data["Year"] <= 2023)]
filtered_data = filtered_data.dropna(subset=[
    "Visitor_Number", "RevenueM&euro;", "Climate_Index", "Carbon_Emissions_kt_CO2_", "Resident_Satisfaction"
])

filtered_data["Infrastructure_Investment_Normalized"] = (
    filtered_data["Infrastructure_Investment_%"] / 100
)
filtered_data["Visitor_Number_Normalized"] = (
    filtered_data["Visitor_Number"] / filtered_data["Visitor_Number"].max()
)

X_environment = filtered_data[["Visitor_Number", "Environment_Investment_%", "Climate_Index"]]
y_environment = filtered_data["Carbon_Emissions_kt_CO2_"]

X_revenue = filtered_data[["Visitor_Number", "Bed Tax%"]]
y_revenue = filtered_data["RevenueM&euro;"]

X_satisfaction = filtered_data[[
    "Infrastructure_Investment_Normalized", "Visitor_Number_Normalized", "Poverty_Rate%", "Unemployment_Rate%"
]]
y_satisfaction = filtered_data["Resident_Satisfaction"]

env_model = LinearRegression()
env_model.fit(X_environment, y_environment)
y_env_pred = env_model.predict(X_environment)

X_revenue_log = np.log(X_revenue)
y_revenue_log = np.log(y_revenue)
rev_model = LinearRegression()
rev_model.fit(X_revenue_log, y_revenue_log)
y_rev_pred = rev_model.predict(X_revenue_log)

sat_model = LinearRegression()
sat_model.fit(X_satisfaction, y_satisfaction)
y_sat_pred = sat_model.predict(X_satisfaction)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].scatter(y_environment, y_env_pred, alpha=0.7, label="Predicted vs Actual")
axs[0].plot([y_environment.min(), y_environment.max()], [y_environment.min(), y_environment.max()], 'r--', label="Perfect Fit")
axs[0].set_title("Environmental Impact Model")
axs[0].set_xlabel("Actual Carbon Emissions")
axs[0].set_ylabel("Predicted Carbon Emissions")
axs[0].legend()

axs[1].scatter(y_revenue_log, y_rev_pred, alpha=0.7, label="Predicted vs Actual (Log Scale)")
axs[1].plot([y_revenue_log.min(), y_revenue_log.max()], [y_revenue_log.min(), y_revenue_log.max()], 'r--', label="Perfect Fit")
axs[1].set_title("Revenue Model (Log Scale)")
axs[1].set_xlabel("Actual Log(Revenue)")
axs[1].set_ylabel("Predicted Log(Revenue)")
axs[1].legend()

axs[2].scatter(y_satisfaction, y_sat_pred, alpha=0.7, label="Predicted vs Actual")
axs[2].plot([y_satisfaction.min(), y_satisfaction.max()], [y_satisfaction.min(), y_satisfaction.max()], 'r--', label="Perfect Fit")
axs[2].set_title("Resident Satisfaction Model")
axs[2].set_xlabel("Actual Satisfaction")
axs[2].set_ylabel("Predicted Satisfaction")
axs[2].legend()

plt.tight_layout()
plt.savefig("model_results.png", dpi=300)
plt.show()
