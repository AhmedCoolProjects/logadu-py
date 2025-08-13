from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
__version__ = "0.2.9"  # Update this version as needed

# Make pygraphviz optional

# In your setup.py file

install_requires = [
    'click>=7.0',
    'pandas>=1.0',
    'tqdm>=4.0',
    'regex>=2020.0',
    'numpy>=1.0',
    'torch>=2.3.1',
    'pytorch-lightning>=2.0',
    'torchmetrics>=1.0',    
    'scikit-learn>=1.0',
    'gensim>=4.0',
    'wandb>=0.15',
    'joblib>=1.0',
    'transformers>=4.0',
    'sentencepiece>=0.1'
]

extras_require = {
    'visualization': ['pygraphviz>=1.5'],
}

setup(
    name="logadu",
    version=__version__,
    author="Ahmed BARGADY",  # ADDED: Your name
    author_email="ahmed.bargady@um6p.ma", # ADDED: Your email
    description="Log Anomaly Detection Ultimate: A package for log parsing, feature representation, and model training.", # ADDED
    long_description=long_description, # ADDED
    long_description_content_type="text/markdown", # ADDED
    url="https://github.com/AhmedCoolProjects/logadu-py", # ADDED
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'logadu=logadu.cli.main:cli',
        ],
    },
    classifiers=[ # ADDED: Helps users find your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, change if needed
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Specify minimum Python version
)