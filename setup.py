from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:

    requirements = []

    with open(file_path) as obj:
        requirements = obj.readlines()
        requirements = [r.replace("\n", "") for r in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements





setup(
    name="project_1",
    version='1.0',
    author="Pavan",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


