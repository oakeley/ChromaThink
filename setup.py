from setuptools import setup, find_packages

setup(
    name="chromathink",
    version="0.1.0",
    description="Colour-Wave Neural Architecture for Cognitive Processing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.14.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-benchmark>=4.0.0",
        ]
    }
)