from setuptools import setup, find_packages

setup(
    name="lfads-efish",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
        "tqdm>=4.50.0",
    ],
    author="Amin Akhshi",
    author_email="amin.akhshi@gmail.com",
    description="PyTorch implementation of Latent Factor Analysis via Dynamical Systems (LFADS) for electrosensory data.",
    keywords="neuroscience, neural data, latent factors, dynamical systems, pytorch",
    url="https://github.com/aminakhshi/lfads-pytorch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
)