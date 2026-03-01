import pandas as pd
import random

df = pd.read_csv("datasets/software_projects.csv")

failed_projects = []

for i in range(15):
    failed_projects.append({
        "project": f"failed_project_{i}",
        "open_issues": random.randint(500, 5000),
        "total_issues": random.randint(100, 300),
        "contributors": random.randint(1, 5),
        "months_inactive": random.randint(12, 48),
        "label": 0
    })

df_failed = pd.DataFrame(failed_projects)

final_df = pd.concat([df, df_failed], ignore_index=True)
final_df.to_csv("datasets/software_projects_final.csv", index=False)

print("✅ Dataset balanced with failed projects")
print(final_df["label"].value_counts())
