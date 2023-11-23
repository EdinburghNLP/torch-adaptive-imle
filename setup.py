import pathlib
from setuptools import setup, find_packages


# Get the long description from the README file
readme_path = pathlib.Path(__file__).parent.resolve() / "README.md"
long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="torch-adaptive-imle",
    version="1.0.0",
    description="Adaptive Perturbation-Based Gradient Estimation for Discrete Latent Variable Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdinburghNLP/torch-adaptive-imle/",
    author="Pasquale Minervini",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="torch, deep learning, machine learning, gradient estimation",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=["torch"],
    project_urls={
        "Bug Reports": "https://github.com/EdinburghNLP/torch-adaptive-imle/issues",
        "Source": "https://github.com/EdinburghNLP/torch-adaptive-imle/",
        "Paper": "https://arxiv.org/abs/2209.04862",
    },
)
