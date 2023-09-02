from setuptools import setup, find_packages

setup(name = "Aptare",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy','scipy','scikit-learn', 'matplotlib', 'celerite',' emcee','corner'],
)

