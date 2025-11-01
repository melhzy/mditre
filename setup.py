from setuptools import setup, find_packages
import os

# Read README with proper encoding
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='mditre',
    version='1.0.0',
    description='MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/melhzy/mditre',
    author='Venkata Suhas Maringanti',
    author_email='vsuhas.m@gmail.com',
    maintainer='melhzy',
    license='GPLv3',
    packages=find_packages(exclude=['tests*', 'docs*', 'jupyter*', 'mditre_paper_results*']),
    install_requires=[
        'numpy>=1.20.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pandas>=1.2.0',
        'scipy>=1.6.0',
        'torch>=2.0.0',  # PyTorch for neural network implementation
        'ete3>=3.1.2',  # Phylogenetic tree handling
        'dendropy>=4.5.0',  # Phylogenetic tree operations
        'seedhash @ git+https://github.com/melhzy/seedhash.git#subdirectory=Python',  # Deterministic seeding
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'jupyterlab>=3.0.0',
        ],
        'viz': [
            'PyQt5>=5.15.0',  # For advanced visualization (optional)
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    keywords='microbiome machine-learning interpretability temporal-analysis rule-extraction',
    project_urls={
        'Documentation': 'https://github.com/melhzy/mditre/blob/master/README.md',
        'Source': 'https://github.com/melhzy/mditre',
        'Bug Reports': 'https://github.com/melhzy/mditre/issues',
    },
)
