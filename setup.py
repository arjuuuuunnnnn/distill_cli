from setuptools import setup, find_packages

setup(
    name="distillation_cli",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'tensorflow>=2.12.0',
        'datasets',
        'pyyaml',
        'tqdm',
        'click',
        'pandas',
        'scikit-learn',
        'pyarrow'
    ],
    entry_points={
        'console_scripts': [
            'distill=distillation_cli.cli:cli',
        ],
    },
)
