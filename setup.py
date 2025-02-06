from setuptools import setup, find_packages

setup(name = "bohmian",
      version="1.0",
      python_requires=">=3.12, <=3.13",
      packages=find_packages(),
      install_requires=[
          "numpy==2.1.2",
          "scipy==1.14.1",
          "matplotlib==3.9.2",
          "pybind11==2.13.6",
          "joblib==1.4.2",
          "numiphy>=1.0"
      ])

