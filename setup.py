from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='phylotorch',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/4ment/phylotorch',
    author='Mathieu Fourment',
    author_email='mathieu.fourment@uts.edu.au',
    keywords="phylogenetics, variational, Bayes, pytorch",
    description='Phylogenetic inference with pytorch',
    license='GPL3',
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    install_requires=[
        line.strip()
        for line in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    extras_require={"libsbn": ["libsbn"], "test": ['pytest']},
    entry_points={'console_scripts': ['phylotorch=phylotorch.phylotorch:main']},
)
