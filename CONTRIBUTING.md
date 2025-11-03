# Contributing to MDITRE

# Contributing to MDITRE v1.0.1

Thank you for your interest in contributing to MDITRE! This guide will help you set up your development environment and submit your contributions.

## 🎯 Ways to Contribute

- 🐛 **Bug Reports**: Submit issues for bugs you've found
- ✨ **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add documentation
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 🧪 **Testing**: Add or improve test coverage
- 📊 **Examples**: Create tutorials or usage examples

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mditre.git
cd mditre

# Add upstream remote
git remote add upstream https://github.com/melhzy/mditre.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 3. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout master
git merge upstream/master

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 💻 Development Workflow

### Code Style

We follow these standards:

- **Formatter**: [Black](https://black.readthedocs.io/) with 100 character line length
- **Import Sorting**: [isort](https://pycqa.github.io/isort/)
- **Linter**: [Flake8](https://flake8.pycqa.org/)
- **Type Hints**: Encouraged for all public APIs

**Format your code:**
```bash
# Format with Black
black mditre/ tests/

# Sort imports
isort mditre/ tests/

# Check with Flake8
flake8 mditre/ tests/ --max-line-length=100
```

### Testing

We maintain comprehensive test coverage. All contributions should include tests.

**Current Test Status (v1.0.1)**:
- ✅ Python: 39/39 tests passing (100% coverage)
- ✅ R: 39/39 tests passing (100% coverage)  
- ✅ Cross-Platform: 3/3 verification tests passing
- ✅ Total: 81/81 tests passing

**Run tests:**
```bash
# Quick verification (< 1 second)
python scripts/verify_cross_platform.py

# Run all Python tests
pytest

# Run with coverage
pytest --cov=mditre --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests with specific marker
pytest -m architecture
```

**Test Requirements:**
- All new features must include tests
- Bug fixes should include regression tests
- Aim for >80% code coverage
- Tests should be fast (<1s per test when possible)

### Type Hints

Add type hints to all public functions:

```python
from typing import Optional, Tuple, Dict
import torch
from torch import nn

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train MDITRE model.
    
    Args:
        model: MDITRE model instance
        train_loader: Training data loader
        epochs: Number of training epochs
        device: Training device (CPU/CUDA)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    pass
```

### Documentation

- Use **Google-style docstrings** for all public APIs
- Include examples in docstrings when helpful
- Update relevant documentation files in `docs/`
- Keep `README.md` up to date

**Example docstring:**
```python
def extract_rules(
    model: nn.Module,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Extract interpretable rules from trained MDITRE model.
    
    Analyzes model weights and thresholds to generate human-readable
    IF-THEN rules describing temporal microbiome patterns.
    
    Args:
        model: Trained MDITRE model instance
        threshold: Minimum rule weight threshold (0.0-1.0)
        
    Returns:
        List of rule dictionaries with keys:
            - 'taxa': List of microbe names
            - 'temporal_pattern': 'increasing', 'decreasing', or 'stable'
            - 'time_window': (start_time, end_time) tuple
            - 'weight': Rule importance score
            
    Example:
        >>> from mditre.models import MDITRE
        >>> model = MDITRE.load('model.pth')
        >>> rules = extract_rules(model, threshold=0.7)
        >>> print(rules[0])
        {
            'taxa': ['Bacteroides', 'Firmicutes'],
            'temporal_pattern': 'increasing',
            'time_window': (0, 14),
            'weight': 0.85
        }
        
    Note:
        Rules are sorted by weight in descending order.
    """
    pass
```

## 📋 Contribution Guidelines

### Pull Request Process

1. **Update Documentation**: Ensure README.md and relevant docs are updated
2. **Add Tests**: Include tests for new features or bug fixes
3. **Update CHANGELOG**: Add entry under "Unreleased" section
4. **Pass CI Checks**: Ensure all tests and linting pass
5. **Clean Commits**: Use clear, descriptive commit messages

**Good commit messages:**
```
✅ Add phylogenetic distance caching for faster training
✅ Fix memory leak in data loader batching
✅ Docs: Add tutorial for custom layer development
```

**Poor commit messages:**
```
❌ fix bug
❌ update
❌ changes
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new features
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Commented complex code sections
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] CHANGELOG.md updated
```

### Code Review

All submissions require review. Reviewers will check:

- ✅ Code quality and style adherence
- ✅ Test coverage and passing tests
- ✅ Documentation completeness
- ✅ No breaking changes (or properly documented)
- ✅ Performance impact (if applicable)

## 🐛 Bug Reports

**Use GitHub Issues** with the following information:

1. **Description**: Clear description of the bug
2. **Reproduction Steps**: Minimal code to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS: (e.g., Ubuntu 24.04, Windows 11)
   - Python version: (e.g., 3.10.12)
   - MDITRE version: (e.g., 1.0.0)
   - PyTorch version: (e.g., 2.5.1)
   - CUDA version: (if using GPU)

**Example:**
```markdown
**Bug**: Memory leak during long training runs

**To Reproduce:**
```python
from mditre.models import MDITRE
model = MDITRE(...)
trainer.train_model(model, epochs=1000)  # Memory grows continuously
```

**Expected**: Memory stays constant during training

**Actual**: Memory increases by ~100MB per epoch

**Environment**:
- OS: Ubuntu 24.04
- Python: 3.10.12
- MDITRE: 1.0.0
- PyTorch: 2.5.1+cu121
- GPU: NVIDIA RTX 4090 (16GB)
```

## ✨ Feature Requests

**Use GitHub Issues** with:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions considered
4. **Additional Context**: Examples, mockups, etc.

## 🏗️ Architecture Guidelines

### Current Structure

```
mditre/
├── core/           # Base classes and utilities
├── layers/         # Modular 5-layer architecture (NEW, recommended)
├── data_loader/    # Modern data loading (NEW, recommended)
├── models.py       # Legacy monolithic models (deprecated)
├── trainer.py      # Training utilities (to be refactored)
└── seeding.py      # Reproducibility utilities
```

### Design Principles

1. **Modularity**: Keep components independent and reusable
2. **Extensibility**: Easy to add new layers, data loaders, etc.
3. **Testability**: Each component should be unit-testable
4. **Documentation**: Public APIs must be well-documented
5. **Backward Compatibility**: Maintain for v1.x releases

### Adding New Components

**New Layer Example:**
```python
# mditre/layers/layer6_attention/attention.py
from mditre.core import BaseLayer
import torch.nn as nn

class AttentionLayer(BaseLayer):
    """Multi-head attention for temporal patterns.
    
    This layer applies attention mechanisms to identify critical
    time points in microbiome trajectories.
    """
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        
    def forward(self, x):
        """Forward pass with attention."""
        # Implementation
        pass
```

**New Data Loader Example:**
```python
# mditre/data_loader/loaders/qiime2_loader.py
from mditre.data_loader.base_loader import BaseDataLoader

class QIIME2Loader(BaseDataLoader):
    """Load data from QIIME2 artifacts."""
    
    def load_data(self, artifact_path: str):
        """Load QIIME2 .qza artifact."""
        # Implementation
        pass
```

## 📚 Resources

- **Main Repository**: https://github.com/melhzy/mditre
- **Issue Tracker**: https://github.com/melhzy/mditre/issues
- **Documentation**: https://github.com/melhzy/mditre/blob/master/README.md

## 🤝 Code of Conduct

### Our Standards

- ✅ Be respectful and inclusive
- ✅ Welcome newcomers and help them learn
- ✅ Focus on constructive feedback
- ✅ Assume good intentions

### Unacceptable Behavior

- ❌ Harassment or discriminatory language
- ❌ Personal attacks or trolling
- ❌ Spam or off-topic content

## 📄 License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0 (GPL-3.0).

## ❓ Questions?

- **General Questions**: Open a GitHub Issue with the "question" label
- **Private Inquiries**: Contact maintainers directly

## 🙏 Thank You!

Your contributions make MDITRE better for everyone. We appreciate your time and effort!

---

*Last Updated: November 1, 2025*
