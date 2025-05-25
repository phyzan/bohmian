from setuptools import setup, find_packages

setup(name = "bohmian",
      version="1.0",
      python_requires=">=3.12, <=3.13",
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib>=3.9.2",
          "numiphy>=1.0"
      ])

