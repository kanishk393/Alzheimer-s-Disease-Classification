"""
Setup script for Alzheimer's Disease Classification package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alzheimer-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning system for automated Alzheimer's disease classification from MRI brain images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alzheimer-disease-classification",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/alzheimer-disease-classification/issues",
        "Documentation": "https://github.com/yourusername/alzheimer-disease-classification/docs",
        "Source Code": "https://github.com/yourusername/alzheimer-disease-classification",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.5",
        ],
        "webapp": [
            "streamlit>=1.25.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alzheimer-train=scripts.train:main",
            "alzheimer-evaluate=scripts.evaluate:main",
            "alzheimer-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "alzheimer",
        "dementia",
        "deep learning",
        "medical imaging",
        "mri",
        "classification",
        "pytorch",
        "efficientnet",
        "computer vision",
        "healthcare",
        "artificial intelligence",
        "machine learning",
    ],
)