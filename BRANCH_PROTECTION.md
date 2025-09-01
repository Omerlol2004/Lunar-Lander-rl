# Branch Protection Configuration

## Recommended Settings

To ensure code quality and maintain performance standards, configure branch protection for the `main` branch with the following settings:

1. Navigate to your repository on GitHub
2. Go to Settings > Branches
3. Add a branch protection rule for `main`
4. Configure the following settings:

### Required Settings

- ✅ **Require a pull request before merging**
- ✅ **Require status checks to pass before merging**
  - Select the `Repository Verification and Performance Testing` workflow as a required status check
- ✅ **Require branches to be up to date before merging**

### Additional Recommended Settings

- ✅ **Require linear history**
- ✅ **Include administrators**

## Benefits

These branch protection rules ensure:

1. All code changes go through pull requests
2. The CI/CD pipeline verifies repository structure and performance metrics
3. Performance regressions are caught before they reach the main branch
4. The repository maintains a clean, linear history

## Performance Guardrails

The GitHub Actions workflow enforces performance standards within the tolerances specified in the RUNBOOK.md:

- Clean environment: 227-307 reward range
- Robust environment: 200-271 reward range
- Extreme environment: 146-198 reward range

Any pull request that causes performance to fall outside these ranges will fail the CI check and be prevented from merging.