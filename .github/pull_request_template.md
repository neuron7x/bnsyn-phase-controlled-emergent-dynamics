## Description

Описати зміни

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test improvement

## Pre-Merge Checklist

**ОБОВ'ЯЗКОВО перед створенням PR:**

- [ ] Локально запустив: `pre-commit run --all-files` ✅
- [ ] Локально запустив: `make check` ✅
- [ ] Локально запустив: `pytest -m "not validation" --cov=src/bnsyn --cov-fail-under=85` ✅
- [ ] Покриття коду ≥85% ✅
- [ ] Жодних lint-помилок ✅
- [ ] mypy --strict пройшов без помилок ✅
- [ ] SSOT перевірки пройшли (bibliography, claims, normative tags) ✅
- [ ] Security audits пройшли (gitleaks, pip-audit, bandit) ✅
- [ ] Тести детермінізму пройшли (однаковий seed = однакові результати) ✅
- [ ] Всі нові функції мають docstrings ✅
- [ ] Додав тести для нового коду ✅

## CI Status

> Автоматично заповнюється CI

## Related Issues

Closes #(issue number)

## Testing

Описати як було протестовано локально

## Screenshots (if applicable)
