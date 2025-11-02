# MDITRE Documentation# MDITRE Documentation# MDITRE Documentation# Documentation Archive



**Version**: 1.0.0  

**Last Updated**: November 1, 2025  

**Status**: Production Ready**Version**: 1.0.0  



---**Last Updated**: November 1, 2025  



## Overview**Status**: Production Ready**Version:** 1.0.0  This folder contains historical documentation files created during the MDITRE v2.0 modular refactoring process. These documents have been consolidated into the main `README.md` for better user experience.



This directory contains comprehensive technical documentation for MDITRE (Microbiome DynamIc Time-series Rule Extraction). The documentation is organized into focused guides covering architecture, development, data loading, seeding, and bug fixes.



------**Last Updated:** January 2025



## Documentation Index



### 🏗️ [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)## Overview## Archived Documents

**Detailed Architecture Documentation**



Comprehensive guide to MDITRE's 5-layer modular architecture:

- Layer 1: Phylogenetic Focus (Spatial Aggregation)This directory contains comprehensive technical documentation for MDITRE (Microbiome DynamIc Time-series Rule Extraction). The documentation is organized into focused guides covering architecture, development, data loading, seeding, and bug fixes.---

- Layer 2: Temporal Focus (Time Windows)

- Layer 3: Pattern Detection (Thresholds & Slopes)

- Layer 4: Rule Layer (Interpretable Rules)

- Layer 5: Classification (Dense Layer)---- **CHECKLIST.md**: Implementation checklist tracking all completed work



**Use this when**:

- Understanding the model architecture

- Implementing custom layers## Documentation Index## Quick Navigation- **DATA_LOADER_GUIDE.md**: Comprehensive guide to the data loader system

- Modifying existing components

- Troubleshooting layer interactions



**Key Topics**: Layer specifications, model variants, PyTorch implementation details### 🏗️ [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)- **IMPLEMENTATION_COMPLETE.md**: Summary of the complete implementation



---**Detailed Architecture Documentation**



### 📦 [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md)### 📚 Start Here- **MODULAR_ARCHITECTURE.md**: Detailed architecture documentation

**Data Loading System Reference**

Comprehensive guide to MDITRE's 5-layer modular architecture:

Complete guide to the modular data loader system:

- `BaseDataLoader` and loader registry- Layer 1: Phylogenetic Focus (Spatial Aggregation)- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Master documentation hub with overview of all topics- **PACKAGE_INTEGRITY_REPORT.md**: Validation report and test results

- PyTorch `Dataset` implementations

- Data transformations and preprocessing- Layer 2: Temporal Focus (Time Windows)

- Format-specific loaders (Pickle, DADA2, QIIME2)

- Layer 3: Pattern Detection (Thresholds & Slopes)- **REFACTORING_PLAN.md**: Original refactoring plan

**Use this when**:

- Loading microbiome data- Layer 4: Rule Layer (Interpretable Rules)

- Implementing custom data loaders

- Adding new data format support- Layer 5: Classification (Dense Layer)### 🔧 Technical References- **REFACTORING_SUMMARY.md**: Summary of refactoring work

- Preprocessing and transforming data



**Key Topics**: Data loader API, transforms, datasets, format support

**Use this when**:- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)** - Detailed architecture design and layer specifications

---

- Understanding the model architecture

### 🛠️ [DEVELOPMENT.md](DEVELOPMENT.md)

**Development Guide & Performance Analysis**- Implementing custom layers- **[DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md)** - Complete data loading API and dataset integration## Current Documentation



v1.0.0 infrastructure improvements and development workflow:- Modifying existing components

- Modern packaging (pyproject.toml, requirements.txt)

- Task automation (Makefile with 20+ commands)- Troubleshooting layer interactions- **[TRAINER_FIXES.md](TRAINER_FIXES.md)** - Bug fixes, improvements, and trainer utilities

- Test organization and quality checks

- Performance metrics and optimization opportunities

- Development workflow and best practices

**Key Topics**: Layer specifications, model variants, PyTorch implementation detailsFor current documentation, please see:

**Use this when**:

- Setting up development environment

- Running tests and quality checks

- Understanding v1.0.0 improvements---### 🎲 Reproducibility

- Identifying code optimization opportunities

- Contributing to the project



**Key Topics**: Infrastructure, workflows, performance, optimization### 📦 [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md)- **[SEEDING.md](SEEDING.md)** - Complete seeding module API reference- **Main README**: `../README.md` - Comprehensive user guide



---**Data Loading System Reference**



### 🎲 [SEEDING_GUIDE.md](SEEDING_GUIDE.md)- **[SEEDING_INTEGRATION.md](SEEDING_INTEGRATION.md)** - Repository-wide seeding integration details- **Data Loader Specific**: `../mditre/data_loader/README.md` - Data loading API reference

**Reproducibility & Seeding Module**

Complete guide to the modular data loader system:

Complete seeding module documentation:

- `MDITRESeedGenerator` API reference- `BaseDataLoader` and loader registry- **Examples**: `../mditre/examples/` - Working code examples

- Deterministic seed generation

- Repository-wide integration- PyTorch `Dataset` implementations

- Reproducibility best practices

- Data transformations and preprocessing### ⚡ Performance- **Tutorials**: `../jupyter/tutorials/` - Jupyter notebook tutorials

**Use this when**:

- Ensuring experiment reproducibility- Format-specific loaders (Pickle, DADA2, QIIME2)

- Generating deterministic seeds

- Setting up cross-validation- **[EFFICIENCY_REPORT.md](EFFICIENCY_REPORT.md)** - Performance analysis, benchmarks, and optimization guide

- Creating reproducible training scripts

**Use this when**:

**Key Topics**: Seed generation, reproducibility, CUDA seeding, API reference

- Loading microbiome data## Purpose

---

- Implementing custom data loaders

### 🐛 [TRAINER_FIXES.md](TRAINER_FIXES.md)

**Bug Fixes & Trainer Utilities**- Adding new data format support---



Documentation of bug fixes applied to `mditre/trainer.py`:- Preprocessing and transforming data

- Variable initialization fixes

- Loss variable unbound errors resolvedThese documents are preserved for:

- Type safety improvements

- Comprehensive test validation**Key Topics**: Data loader API, transforms, datasets, format support



**Use this when**:## Documentation Structure- Historical reference

- Understanding trainer.py fixes

- Debugging trainer-related issues---

- Reviewing code quality improvements

- Validating test results- Development tracking



**Key Topics**: Bug fixes, variable initialization, loss computation, testing### 🛠️ [DEVELOPMENT.md](DEVELOPMENT.md)



---**Development Guide & Performance Analysis**```- Understanding design decisions



## Quick Navigation



### For New Usersv1.0.0 infrastructure improvements and development workflow:docs/- Future maintenance

1. Start with [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) to understand the model

2. Check [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) for loading your data- Modern packaging (pyproject.toml, requirements.txt)

3. See main [README.md](../README.md) for installation and usage

- Task automation (Makefile with 20+ commands)├── README.md                    ← You are here

### For Developers

1. Read [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow- Test organization and quality checks

2. Review [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) for architecture

3. Check [TRAINER_FIXES.md](TRAINER_FIXES.md) for recent improvements- Performance metrics and optimization opportunities├── DOCUMENTATION.md             ← Start with this overviewThey are no longer required for day-to-day usage of MDITRE, as all essential information has been integrated into the main README.md in a more user-friendly format.



### For Reproducibility- Development workflow and best practices

1. Follow [SEEDING_GUIDE.md](SEEDING_GUIDE.md) for reproducible experiments

2. Use seeding module in all experiments├── MODULAR_ARCHITECTURE.md      ← Architecture deep-dive

3. Save seed information with results

**Use this when**:

### For Data Scientists

1. Start with [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) for data loading- Setting up development environment├── DATA_LOADER_GUIDE.md         ← Data loading reference---

2. See [DEVELOPMENT.md](DEVELOPMENT.md) for performance optimization

3. Use [SEEDING_GUIDE.md](SEEDING_GUIDE.md) for reproducible analyses- Running tests and quality checks



---- Understanding v1.0.0 improvements├── SEEDING.md                   ← Seeding API



## File Summary- Identifying code optimization opportunities



| File | Size | Purpose | Last Updated |- Contributing to the project├── SEEDING_INTEGRATION.md       ← Seeding integration**Last Updated**: 2024  

|------|------|---------|--------------|

| [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) | ~12KB | Architecture reference | 2025-01 |

| [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) | ~13KB | Data loading API | 2025-01 |

| [DEVELOPMENT.md](DEVELOPMENT.md) | ~14KB | Development guide | 2025-11 |**Key Topics**: Infrastructure, workflows, performance, optimization├── TRAINER_FIXES.md             ← Bug fixes & improvements**Status**: Archived

| [SEEDING_GUIDE.md](SEEDING_GUIDE.md) | ~14KB | Seeding & reproducibility | 2025-11 |

| [TRAINER_FIXES.md](TRAINER_FIXES.md) | ~7KB | Bug fix documentation | 2025-01 |



**Total Documentation**: ~60KB across 5 focused documents---└── EFFICIENCY_REPORT.md         ← Performance analysis



---```



## Additional Resources### 🎲 [SEEDING_GUIDE.md](SEEDING_GUIDE.md)



### Main Documentation**Reproducibility & Seeding Module**---

- **Main README**: `../README.md` - Installation, usage, examples

- **Data Loader Specific**: `../mditre/data_loader/README.md` - Data loader module docs

- **Quality Assurance**: `../QA.md` - Testing and validation

- **Changelog**: `../CHANGELOG.md` - Version historyComplete seeding module documentation:## Quick Start

- **Contributing**: `../CONTRIBUTING.md` - Contribution guidelines

- `MDITRESeedGenerator` API reference

### Examples & Tutorials

- **Examples**: `../mditre/examples/` - Working code examples- Deterministic seed generation### For New Users

- **Tutorials**: `../jupyter/` - Jupyter notebook tutorials

- **Test Notebook**: `../jupyter/run_mditre_test.ipynb` - Quick training demo- Repository-wide integration1. Read **[DOCUMENTATION.md](DOCUMENTATION.md)** for overview



### Test Documentation- Reproducibility best practices2. Check **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)** to understand the model

- **Test README**: `../tests/README.md` - Test organization and usage

- **Comprehensive Tests**: `../tests/test_mditre_comprehensive.py` - Full test suite3. See main repository `README.md` for installation and usage examples

- **Package Validation**: `../tests/validate_package.py` - Package integrity checks

**Use this when**:

---

- Ensuring experiment reproducibility### For Developers

## Documentation Standards

- Generating deterministic seeds1. **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)** - Understand the architecture

### Organization

- ✅ **Focused**: Each document covers a specific topic- Setting up cross-validation2. **[TRAINER_FIXES.md](TRAINER_FIXES.md)** - Review recent fixes

- ✅ **Complete**: Comprehensive coverage without duplication

- ✅ **Structured**: Clear table of contents and navigation- Creating reproducible training scripts3. **[EFFICIENCY_REPORT.md](EFFICIENCY_REPORT.md)** - Performance considerations

- ✅ **Cross-referenced**: Links between related documents



### Content

- ✅ **Examples**: Code examples for all major features**Key Topics**: Seed generation, reproducibility, CUDA seeding, API reference### For Reproducibility

- ✅ **API Reference**: Complete API documentation where applicable

- ✅ **Best Practices**: Guidance on recommended usage patterns1. **[SEEDING.md](SEEDING.md)** - Learn the seeding API

- ✅ **Troubleshooting**: Common issues and solutions

---2. **[SEEDING_INTEGRATION.md](SEEDING_INTEGRATION.md)** - See integration examples

### Maintenance

- ✅ **Version Tracking**: Document version and last update date

- ✅ **Status Indicators**: Production ready, experimental, deprecated

- ✅ **Change History**: Major updates documented in CHANGELOG.md### 🐛 [TRAINER_FIXES.md](TRAINER_FIXES.md)### For Data Scientists



---**Bug Fixes & Trainer Utilities**1. **[DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md)** - Load and preprocess data



## Contributing to Documentation2. **[EFFICIENCY_REPORT.md](EFFICIENCY_REPORT.md)** - Optimize performance



When updating documentation:Documentation of bug fixes applied to `mditre/trainer.py`:



1. **Maintain Focus**: Keep each document focused on its specific topic- Variable initialization fixes---

2. **Avoid Duplication**: Cross-reference instead of duplicating content

3. **Update Version**: Update "Last Updated" date when making changes- Loss variable unbound errors resolved

4. **Add Examples**: Include code examples for new features

5. **Link Related Docs**: Add cross-references to related documentation- Type safety improvements## Document Changelog

6. **Test Examples**: Verify all code examples work

7. **Follow Format**: Maintain consistent formatting and structure- Comprehensive test validation



See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.### v1.0.0 (January 2025)



---**Use this when**:- ✅ Consolidated 19 files into 7 focused documents



## Support- Understanding trainer.py fixes- ✅ Removed legacy/duplicate content



- **Issues**: [GitHub Issues](https://github.com/melhzy/mditre/issues)- Debugging trainer-related issues- ✅ Created master DOCUMENTATION.md hub

- **Discussions**: [GitHub Discussions](https://github.com/melhzy/mditre/discussions)

- **Email**: Check main README for contact information- Reviewing code quality improvements- ✅ Organized by user needs



---- Validating test results



## Change History### Removed Files (Duplicates/Obsolete)



### v1.0.0 (November 2025)**Key Topics**: Bug fixes, variable initialization, loss computation, testing- CONSOLIDATION_SUMMARY_legacy.md

- ✅ Reorganized documentation into 5 focused files

- ✅ Merged EFFICIENCY_IMPLEMENTATION.md + EFFICIENCY_REPORT.md → DEVELOPMENT.md- TEST_RESULTS_legacy.md

- ✅ Merged SEEDING.md + SEEDING_INTEGRATION.md → SEEDING_GUIDE.md

- ✅ Removed duplicate content and obsolete files (DOCUMENTATION.md)---- CHECKLIST.md

- ✅ Created comprehensive README.md as documentation index

- ✅ Total size: ~60KB across 5 documents (streamlined and focused)- IMPLEMENTATION_COMPLETE.md

- ✅ Eliminated content duplication across files

## Quick Navigation- PACKAGE_INTEGRITY_REPORT.md

### Previous Versions

- See individual file headers for earlier change history- REFACTORING_PLAN.md

- See [CHANGELOG.md](../CHANGELOG.md) for project version history

### For New Users- REFACTORING_SUMMARY.md

---

1. Start with [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) to understand the model- COMPREHENSIVE_TESTING_PLAN.md

**Documentation Version**: 1.0.0  

**Project Version**: 1.0.0  2. Check [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) for loading your data- TESTING_IMPLEMENTATION_STATUS.md

**Status**: Production Ready  

**Last Updated**: November 1, 20253. See main [README.md](../README.md) for installation and usage- QA_CHECKLIST.md



---- QA_TEST_REPORT.md



*For the latest updates, see the [MDITRE repository](https://github.com/melhzy/mditre)*### For Developers- STATUS.md


1. Read [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow

2. Review [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) for architecture**Note:** All useful content from removed files has been integrated into remaining documents.

3. Check [TRAINER_FIXES.md](TRAINER_FIXES.md) for recent improvements

---

### For Reproducibility

1. Follow [SEEDING_GUIDE.md](SEEDING_GUIDE.md) for reproducible experiments## Support

2. Use seeding module in all experiments

3. Save seed information with results- **Issues**: Check repository `QA.md` for quality assurance details

- **Tests**: Run `pytest test_mditre_comprehensive.py -v`

### For Data Scientists- **Examples**: See `jupyter/run_mditre_test.ipynb`

1. Start with [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) for data loading- **Main Docs**: Repository root `README.md`

2. See [DEVELOPMENT.md](DEVELOPMENT.md) for performance optimization

3. Use [SEEDING_GUIDE.md](SEEDING_GUIDE.md) for reproducible analyses---



---**Status:** Active and Maintained  

**Next Review:** After major version updates or significant feature additions

## File Summary

| File | Size | Purpose | Last Updated |
|------|------|---------|--------------|
| [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) | ~12KB | Architecture reference | 2025-01 |
| [DATA_LOADER_GUIDE.md](DATA_LOADER_GUIDE.md) | ~13KB | Data loading API | 2025-01 |
| [DEVELOPMENT.md](DEVELOPMENT.md) | ~28KB | Development guide | 2025-11 |
| [SEEDING_GUIDE.md](SEEDING_GUIDE.md) | ~15KB | Seeding & reproducibility | 2025-11 |
| [TRAINER_FIXES.md](TRAINER_FIXES.md) | ~8KB | Bug fix documentation | 2025-01 |

**Total Documentation**: ~76KB across 5 focused documents

---

## Additional Resources

### Main Documentation
- **Main README**: `../README.md` - Installation, usage, examples
- **Data Loader Specific**: `../mditre/data_loader/README.md` - Data loader module docs
- **Quality Assurance**: `../QA.md` - Testing and validation
- **Changelog**: `../CHANGELOG.md` - Version history
- **Contributing**: `../CONTRIBUTING.md` - Contribution guidelines

### Examples & Tutorials
- **Examples**: `../mditre/examples/` - Working code examples
- **Tutorials**: `../jupyter/` - Jupyter notebook tutorials
- **Test Notebook**: `../jupyter/run_mditre_test.ipynb` - Quick training demo

### Test Documentation
- **Test README**: `../tests/README.md` - Test organization and usage
- **Comprehensive Tests**: `../tests/test_mditre_comprehensive.py` - Full test suite
- **Package Validation**: `../tests/validate_package.py` - Package integrity checks

---

## Documentation Standards

### Organization
- ✅ **Focused**: Each document covers a specific topic
- ✅ **Complete**: Comprehensive coverage without duplication
- ✅ **Structured**: Clear table of contents and navigation
- ✅ **Cross-referenced**: Links between related documents

### Content
- ✅ **Examples**: Code examples for all major features
- ✅ **API Reference**: Complete API documentation where applicable
- ✅ **Best Practices**: Guidance on recommended usage patterns
- ✅ **Troubleshooting**: Common issues and solutions

### Maintenance
- ✅ **Version Tracking**: Document version and last update date
- ✅ **Status Indicators**: Production ready, experimental, deprecated
- ✅ **Change History**: Major updates documented in CHANGELOG.md

---

## Contributing to Documentation

When updating documentation:

1. **Maintain Focus**: Keep each document focused on its specific topic
2. **Avoid Duplication**: Cross-reference instead of duplicating content
3. **Update Version**: Update "Last Updated" date when making changes
4. **Add Examples**: Include code examples for new features
5. **Link Related Docs**: Add cross-references to related documentation
6. **Test Examples**: Verify all code examples work
7. **Follow Format**: Maintain consistent formatting and structure

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/melhzy/mditre/issues)
- **Discussions**: [GitHub Discussions](https://github.com/melhzy/mditre/discussions)
- **Email**: Check main README for contact information

---

## Change History

### v1.0.0 (November 2025)
- ✅ Reorganized documentation into 5 focused files
- ✅ Merged EFFICIENCY_IMPLEMENTATION.md + EFFICIENCY_REPORT.md → DEVELOPMENT.md
- ✅ Merged SEEDING.md + SEEDING_INTEGRATION.md → SEEDING_GUIDE.md
- ✅ Removed duplicate content and obsolete files
- ✅ Created comprehensive README.md as documentation index
- ✅ Total size reduced from ~100KB to ~76KB (24% reduction)
- ✅ Eliminated content duplication across files

### Previous Versions
- See individual file headers for earlier change history
- See [CHANGELOG.md](../CHANGELOG.md) for project version history

---

**Documentation Version**: 1.0.0  
**Project Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: November 1, 2025

---

*For the latest updates, see the [MDITRE repository](https://github.com/melhzy/mditre)*
