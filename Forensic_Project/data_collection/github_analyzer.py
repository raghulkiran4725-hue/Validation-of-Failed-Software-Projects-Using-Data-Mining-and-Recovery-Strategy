from github import Github
from urllib.parse import urlparse
from datetime import datetime, timezone

def analyze_github_repo(github_url):
    g = Github()  # public repos only
    path = urlparse(github_url).path.strip("/")
    repo = g.get_repo(path)

    # Get last commit
    commits = repo.get_commits()
    if commits.totalCount > 0:
        last_commit = commits[0].commit.author.date
        # Convert to naive UTC
        last_commit = last_commit.replace(tzinfo=None)
    else:
        last_commit = datetime.now()

    # Calculate months inactive
    months_inactive = (datetime.now() - last_commit).days // 30

    # Contributors
    contributors_count = repo.get_contributors().totalCount

    # Total issues
    total_issues_count = repo.get_issues(state="all").totalCount

    return {
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "watchers": repo.watchers_count,
        "size": repo.size,
        "total_issues": total_issues_count,
        "contributors": contributors_count,
        "months_inactive": months_inactive
    }
