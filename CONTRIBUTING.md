# Contributing to Trading System

Thank you for your interest in contributing to the trading system! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- IBKR Gateway (for testing)
- Git

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd trader

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Environment
```bash
# Activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Code Style

We use several tools to maintain code quality:

### Pre-commit Hooks
The project uses pre-commit hooks to ensure code quality. These run automatically on commit:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and code quality
- **MyPy**: Type checking

### Manual Code Quality Checks
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
ruff check .

# Type checking
mypy .

# Run all checks
pre-commit run --all-files
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=strategies --cov=features --cov=brokers

# Run specific test file
pytest tests/test_strategies.py

# Run with verbose output
pytest -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (IBKR, yfinance)

Example test structure:
```python
def test_strategy_generates_signals():
    """Test that strategy generates valid signals."""
    # Arrange
    strategy = RegimeAwareEnsembleStrategy(params)
    data = create_test_data()
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert len(signals) > 0
    assert signals.dtype == float
```

## Commit Style

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

### Commit Message Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

### Examples
```bash
# Feature
git commit -m "feat(strategies): add momentum strategy"

# Bug fix
git commit -m "fix(ibkr): handle connection timeout gracefully"

# Documentation
git commit -m "docs: update README with installation instructions"

# Refactor
git commit -m "refactor(core): simplify regime detection logic"

# Test
git commit -m "test(strategies): add unit tests for ensemble strategy"
```

### Breaking Changes
For breaking changes, add `!` after the type and include a footer:
```bash
git commit -m "feat!(api): change strategy interface

BREAKING CHANGE: Strategy.generate_signals() now returns DataFrame instead of Series"
```

## Pull Request Process

### Before Submitting
1. **Update tests**: Add tests for new functionality
2. **Update documentation**: Update relevant documentation
3. **Run checks**: Ensure all pre-commit hooks pass
4. **Test locally**: Verify the system works as expected

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Code Review Guidelines

### For Reviewers
- Be constructive and respectful
- Focus on code quality and functionality
- Check for security implications
- Ensure tests are adequate
- Verify documentation is updated

### For Authors
- Respond to review comments promptly
- Make requested changes or explain why not
- Update PR description if needed
- Re-request review after changes

## Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Create release tag
6. Merge to main branch

## Getting Help

### Questions and Issues
- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check existing documentation first

### Communication
- Be respectful and inclusive
- Use clear, concise language
- Provide context for issues
- Include relevant code examples

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

## Code of Conduct

Please read and follow our Code of Conduct to ensure a welcoming and inclusive environment for all contributors.
