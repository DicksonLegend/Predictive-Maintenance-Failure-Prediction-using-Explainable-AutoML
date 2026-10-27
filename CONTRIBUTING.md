# Contributing Guidelines & GitFlow Governance

Thank you for contributing to the **Explainable AutoML-based Machine Failure Prediction and Root-Cause Analysis for Industry 4.0** project. This document outlines our branching strategy, commit conventions, code review standards, and quality gates.

---

## 1. Branching Strategy (GitFlow)

We employ a structured **GitFlow** branching workflow designed for high-consequence industrial ML systems:

```text
main (Production / Protected)
  ▲
  │ (Release PR after CI validation)
develop (Integration / Staging)
  ▲
  ├── feature/auto-tuning-enhancements
  ├── feature/k8s-autoscaling-policy
  ├── bugfix/shap-surrogate-alignment
  └── hotfix/critical-security-patch ────► (Direct to main & develop)
```

- **`main` (Production):** The production release branch. Strictly protected. Directly mapped to production container image tagging and cluster deployments. Direct commits and force pushes are blocked.
- **`develop` (Integration):** Nightly integration branch for tested feature increments. All feature branches merge here via Pull Request.
- **`feature/<name>`:** Branch off `develop`. Dedicated to isolated feature implementations (e.g., `feature/prometheus-metrics`, `feature/xai-plots`).
- **`bugfix/<name>`:** Branch off `develop`. Resolves non-critical defects and regressions.
- **`hotfix/<name>`:** Branch off `main`. Critical production remediation merged back into both `main` and `develop`.

---

## 2. Commit Message Standards (Conventional Commits)

Commit messages must adhere strictly to the **Conventional Commits v1.0.0** format:

```text
<type>(<scope>): <short description in imperative present tense>

[optional body explaining motivation and technical approach]

[optional footer referencing issues, e.g. Fixes #42]
```

### Approved Types
- `feat`: A new feature or capability (e.g., `feat(api): add /metrics prometheus instrumentation`)
- `fix`: A bug fix (e.g., `fix(xai): resolve surrogate feature alignment for autogluon`)
- `ml`: Changes to ML models, tuning grids, or training loops (e.g., `ml(phase3): run final 300s autogluon benchmark`)
- `ops`: Infrastructure, containerization, or Kubernetes changes (e.g., `ops(k8s): add rolling update and resource limits`)
- `docs`: Documentation updates (e.g., `docs(setup): update devops runbook and screenshots`)
- `test`: Adding or refactoring test suites (e.g., `test(api): add test_11_metrics_endpoint`)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `chore`: Maintenance tasks, dependencies, or tool configurations

---

## 3. Pull Request & Code Review Workflow

1. **Branch Creation:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```
2. **Local Pre-Flight Checks:** Run quality gates before pushing:
   ```powershell
   .venv\Scripts\python -m ruff check app src scripts tests
   .venv\Scripts\python -m bandit -r app src -x tests,.venv -ll
   .venv\Scripts\python -m pytest tests/ -q
   ```
3. **Open Pull Request:** Target `develop` (or `main` for hotfixes). Use the automated `.github/PULL_REQUEST_TEMPLATE.md`.
4. **Automated CI Validation:** The GitHub Actions pipeline will run all 3 stages:
   - Quality Gate: `ruff`, `bandit`, `gitleaks`, `pytest`.
   - Containerization: Multi-stage Docker build, Trivy security scan.
   - Deployment Verification: Ephemeral KinD cluster rollout and health checks.

---

## 4. Definition of Done (DoD)

A pull request or task is considered **Done** only when:
1. All unit and integration tests pass (100% test pass rate).
2. All regression validation scripts pass:
   - `validate_phase2.py`: 31/31 PASS
   - `validate_phase3.py`: 39/39 PASS
   - `validate_phase4.py`: 43/43 PASS
   - `validate_phase5.py`: All Phase 5 checks PASS
3. `ruff` reports zero errors.
4. `bandit` reports zero Medium or High severity issues.
5. All secrets scanning (`gitleaks`) reports clean status.
6. Documentation, Model Card, and setup runbooks reflect all architectural changes.

---

## 5. Code Ownership & Security Governance

- **Maintainers / Code Owners:** Senior MLOps & Platform Engineering Team.
- **Credentials Policy:** Never commit plain-text credentials, tokens, or connection strings. Kubernetes secrets must use placeholders in source control and cloud KMS in production environments.
