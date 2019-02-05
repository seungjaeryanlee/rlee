"""setup.py for rlee."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlee",
    version="0.0.0",
    author="Seungjae Ryan Lee",
    author_email="seungjaeryanlee@gmail.com",
    description=(
        "Rlee is a research framework built on top of PyTorch 1.0 for"
        " fast prototyping of novel reinforcement learning algorithms."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/endtoendai/rlee",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "ConfigArgParse",
        "gym",
        "numpy",
        "opencv-python",
        "scipy",
        "torch",
        "wandb",
    ],
)
