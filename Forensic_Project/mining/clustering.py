import pandas as pd
from sklearn.cluster import KMeans

# Load preprocessed data
df = pd.read_csv("datasets/preprocessed.csv")

X = df.drop("label", axis=1)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Save clustered data
df.to_csv("datasets/clustered.csv", index=False)

print("✅ Clustering completed successfully!")
