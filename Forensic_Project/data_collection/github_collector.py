from github import Github, Auth
import pandas as pd
import os
from datetime import datetime

token = os.getenv("GITHUB_TOKEN")
if not token:
    raise ValueError("❌ GITHUB_TOKEN not found.")

auth = Auth.Token(token)
g = Github(auth=auth)

repos = [
    "facebook/react",
    "tensorflow/tensorflow",
    "torvalds/linux",
    "psf/requests",
    "angular/angular",
    "django/django"
]

data = []

def is_failed(repo):
    conditions = 0

    if repo.open_issues_count > 200:
        conditions += 1

    last_commit = repo.get_commits()[0].commit.author.date
    if (datetime.now() - last_commit).days > 365:
        conditions += 1

    if repo.stargazers_count < 500:
        conditions += 1

    if repo.get_contributors().totalCount < 5:
        conditions += 1

    if repo.forks_count < 50:
        conditions += 1

    return 1 if conditions >= 2 else 0


for repo_name in repos:
    print(f"🔍 Collecting data for {repo_name}...")
    repo = g.get_repo(repo_name)

    failure = is_failed(repo)

    data.append({
        "name": repo.name,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "issues": repo.open_issues_count,
        "contributors": repo.get_contributors().totalCount,
        "failed": failure
    })

df = pd.DataFrame(data)
os.makedirs("datasets", exist_ok=True)
df.to_csv("datasets/software_projects.csv", index=False)

print("✅ Dataset saved successfully")
