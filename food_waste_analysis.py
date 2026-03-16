import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset for FreshFood analytics project
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Food_Supplied_Tonnes": [120, 135, 128, 140, 150, 160],
    "Food_Wasted_Tonnes": [18, 20, 17, 22, 21, 24],
    "Redistributed_Tonnes": [8, 9, 10, 11, 12, 13],
    "NGO_Collections": [15, 17, 16, 20, 22, 24]
}

df = pd.DataFrame(data)

# Data cleaning check
print("Missing values:")
print(df.isnull().sum())
print()

# KPI calculations
df["Waste_Percentage"] = (df["Food_Wasted_Tonnes"] / df["Food_Supplied_Tonnes"]) * 100
df["Recovery_Percentage"] = (df["Redistributed_Tonnes"] / df["Food_Wasted_Tonnes"]) * 100

print("FreshFood KPI Summary")
print(df[["Month", "Waste_Percentage", "Recovery_Percentage"]])
print()

print("Average Waste Percentage: {:.2f}%".format(df["Waste_Percentage"].mean()))
print("Average Recovery Percentage: {:.2f}%".format(df["Recovery_Percentage"].mean()))
print()

# Simple prediction model
df["Month_Number"] = [1, 2, 3, 4, 5, 6]
X = df[["Month_Number"]]
y = df["Food_Wasted_Tonnes"]

model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_months = pd.DataFrame({"Month_Number": [7, 8, 9]})
predictions = model.predict(future_months)

future_results = pd.DataFrame({
    "Month_Number": [7, 8, 9],
    "Predicted_Food_Waste_Tonnes": predictions
})

print("Predicted Food Waste for Next 3 Months")
print(future_results)
print()

# Plot actual waste
plt.figure(figsize=(8, 5))
plt.plot(df["Month"], df["Food_Wasted_Tonnes"], marker="o")
plt.title("Monthly Food Waste Trend")
plt.xlabel("Month")
plt.ylabel("Food Wasted (Tonnes)")
plt.tight_layout()
plt.show()
