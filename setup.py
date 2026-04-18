from setuptools import setup, find_packages

setup(
    name="bayeswarp",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
