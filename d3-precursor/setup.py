from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="d3-precursor",
    version="0.1.0",
    author="D3-Neuro Team",
    author_email="your.email@example.com",
    description="Knowledge extraction and processing for D3 visualizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/d3-neuro",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "d3-process=d3_precursor.cli.process:main",
        ],
    },
)
