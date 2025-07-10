from setuptools import setup, find_packages

# Make pygraphviz optional

install_requires = [
    'click>=7.0',
    'pandas>=1.0',
    'tqdm>=4.0',
    'regex>=2020.0',
    'numpy>=1.0',
]

extras_require = {
    'visualization': ['pygraphviz>=1.5'],
}

setup(
    name="logadu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'logadu=logadu.cli.main:cli',
        ],
    },
)