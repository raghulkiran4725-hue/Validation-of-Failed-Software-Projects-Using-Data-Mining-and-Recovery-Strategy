import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/software_projects_final.csv")

# Failure distribution
plt.figure()
df["label"].value_counts().plot(kind="bar")
plt.xticks([0,1], ["Failed", "Successful"], rotation=0)
plt.title("Project Outcome Distribution")
plt.show()

# Inactivity vs failure
plt.figure()
plt.scatter(df["months_inactive"], df["open_issues"])
plt.xlabel("Months Inactive")
plt.ylabel("Open Issues")
plt.title("Failure Risk Pattern")
plt.show()
