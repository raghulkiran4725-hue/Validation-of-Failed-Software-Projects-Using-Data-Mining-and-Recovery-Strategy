import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("datasets/software_projects.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Normalize features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.iloc[:, 1:-1])

df_scaled = pd.DataFrame(scaled, columns=df.columns[1:-1])
df_scaled["label"] = df["label"]

# Save preprocessed data
df_scaled.to_csv("datasets/preprocessed.csv", index=False)

print("✅ Preprocessing completed successfully!")
