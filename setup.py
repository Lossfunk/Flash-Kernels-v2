"""
Setup configuration for Daniel - GPU Kernel Optimization for PyTorch
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='daniel',
    version='0.1.0',
    author='Shikhar',  # Update this
    author_email='shikhar2807.ace@gmail.com',  # Update this
    description='Automatic GPU kernel optimization for PyTorch using Triton',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/thekernelcompany/Project_daniel/tree/shikhar-test-3',  # Update this
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Update if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'flake8>=4.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Could add CLI commands here later
            # 'daniel=daniel.cli:main',
        ],
    },
) 