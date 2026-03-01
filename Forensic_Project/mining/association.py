import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load clustered data
df = pd.read_csv("datasets/clustered.csv")

# Convert numerical values to binary risk indicators
binary_df = pd.DataFrame()
binary_df["high_open_issues"] = df["open_issues"] > 0.5
binary_df["low_contributors"] = df["contributors"] < 0.3
binary_df["long_inactive"] = df["months_inactive"] > 0.6

# Apply Apriori algorithm
frequent_itemsets = apriori(binary_df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Save rules
rules.to_csv("report/association_rules.csv", index=False)

print("✅ Association rules generated successfully!")
