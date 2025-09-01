# Git History Cleanup Instructions

If the `archive/` directory was previously committed and is inflating the repository size, you can use the following commands to purge it from Git history.

## Prerequisites

- Install [git-filter-repo](https://github.com/newren/git-filter-repo) (recommended) or [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- Make a backup of your repository before proceeding

## Using git-filter-repo (Recommended)

```bash
# Clone a fresh copy of the repository
git clone --mirror <repository-url> repo-mirror
cd repo-mirror

# Remove archive/ directory from all of history
git filter-repo --path archive/ --invert-paths

# Push the changes back to the remote repository
git push --force
```

## Using BFG Repo-Cleaner (Alternative)

```bash
# Clone a fresh copy of the repository
git clone --mirror <repository-url> repo-mirror
cd repo-mirror

# Remove archive/ directory from all of history
java -jar bfg.jar --delete-folders archive

# Clean up and update the repository
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push the changes back to the remote repository
git push --force
```

## Important Notes

- These operations rewrite Git history and require a force push
- All collaborators will need to re-clone the repository or perform specific steps to update their local copies
- Only perform this operation if repository size is a significant issue
- Coordinate with all team members before executing these commands