from setuptools import setup, find_packages

setup(
    name="animus",
    version="0.1.0",
    description="Identity emerging through relational experience in AI systems",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
        "matplotlib>=3.8.0",
    ],
)
