from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name="QSparse",
    version="0.1.0",
    author="Ostix360",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    description="A PyTorch library for Q sparse neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ostix360/Q-sparse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)