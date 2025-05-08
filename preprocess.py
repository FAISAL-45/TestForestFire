import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Correct column names
column_names = [
    'day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain',
    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes'
]

# Load dataset
df = pd.read_csv("Algerian_forest_fires_dataset.csv", skiprows=1, names=column_names)

# Clean and preprocess
df['Classes'] = df['Classes'].str.strip().map({'not fire': 0, 'fire': 1})
df.dropna(inplace=True)

# Define features and target
X = df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
y = df['Classes']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create output folder
os.makedirs('models', exist_ok=True)

# Save model and scaler
with open('models/fire_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved to 'models/' directory.")
