import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


file_path = "insulin dosage.csv" 
df = pd.read_csv(file_path)


df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


target_column = "insulin_dose"
if target_column not in df.columns:
    raise ValueError(f"Error: '{target_column}' column not found in dataset!")

X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

joblib.dump(scaler, "model/scaler.pkl")

print(" Data preprocessing complete!")
