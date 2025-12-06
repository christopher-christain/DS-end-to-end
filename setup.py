from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements from a file and returns them as a list."""
    requirements = []
    with open(file_path, "r") as file_obj:
        for line in file_obj.readlines():
            req = line.strip()
            if req and req != "-e .":
                requirements.append(req)
    return requirements


setup(
    name="end-to-end ds project",                       # Project name
    version="0.1.0",
    author="Christopher Christain",
    author_email="christopherchristain.02@gmail.com",
    description="A Python package for data science project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/christopher-christain/DS-end-to-end",
    packages=find_packages(),                # Automatically find packages
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
