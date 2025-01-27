import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data_path = "1.xlsx"
data = pd.ExcelFile(data_path)
sheet_data = data.parse("Sheet1")

filtered_data = sheet_data[(sheet_data["Year"] >= 2012) & (sheet_data["Year"] <= 2023)]
filtered_data = filtered_data.dropna(subset=[
    "Visitor_Number", "RevenueM€", "Climate_Index", "Carbon_Emissions_kt_CO2_", "Resident_Satisfaction"
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
y_revenue = filtered_data["RevenueM€"]

X_satisfaction = filtered_data[[
    "Infrastructure_Investment_Normalized", "Visitor_Number_Normalized", "Poverty_Rate%", "Unemployment_Rate%"
]]
y_satisfaction = filtered_data["Resident_Satisfaction"]

X_income = filtered_data[["Infrastructure_Investment_Normalized", "Visitor_Number_Normalized"]]
y_income = np.log(filtered_data["RevenueM€"])

X_unemployment = filtered_data[["Infrastructure_Investment_Normalized", "Visitor_Number_Normalized"]]
y_unemployment = filtered_data["Unemployment_Rate%"]

env_model = LinearRegression()
env_model.fit(X_environment, y_environment)

environment_coefficients = {
    "alpha_0 (Intercept)": env_model.intercept_,
    "alpha_1 (TourNum)": env_model.coef_[0],
    "alpha_2 (Ienv)": env_model.coef_[1],
    "alpha_3 (ClimateIndex)": env_model.coef_[2],
}

X_revenue_log = np.log(X_revenue)
y_revenue_log = np.log(y_revenue)
rev_model = LinearRegression()
rev_model.fit(X_revenue_log, y_revenue_log)

revenue_coefficients = {
    "beta_0 (Intercept in exp)": np.exp(rev_model.intercept_),
    "beta_1 (TourNum)": rev_model.coef_[0],
    "beta_2 (TaxRate)": rev_model.coef_[1],
}

sat_model = LinearRegression()
sat_model.fit(X_satisfaction, y_satisfaction)

satisfaction_coefficients = {
    "theta_1 (Iinfr)": sat_model.coef_[0],
    "theta_2 (Crowding)": -sat_model.coef_[1],
    "theta_3 (Income proxy)": sat_model.coef_[2],
    "theta_4 (Unemployment)": -sat_model.coef_[3],
}

income_model = LinearRegression()
income_model.fit(X_income, y_income)

income_coefficients = {
    "omega_0 (Intercept)": income_model.intercept_,
    "omega_1 (Iinfr)": income_model.coef_[0],
    "omega_2 (TourNum)": income_model.coef_[1],
}

unemployment_model = LinearRegression()
unemployment_model.fit(X_unemployment, y_unemployment)

unemployment_coefficients = {
    "psi_0 (Intercept)": unemployment_model.intercept_,
    "psi_1 (Iinfr)": unemployment_model.coef_[0],
    "psi_2 (TourNum)": unemployment_model.coef_[1],
}

environment_coefficients, revenue_coefficients, satisfaction_coefficients, income_coefficients, unemployment_coefficients
