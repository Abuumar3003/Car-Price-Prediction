import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("car_data.csv")

# Feature Engineering
df['Car_Age'] = 2024 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model AND feature names
joblib.dump({
    "model": model,
    "features": X.columns.tolist()
}, "car_prediction_model.pkl")

print("Model trained and saved successfully!")