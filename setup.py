from setuptools import setup


PACKAGENAME = "thechopper"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    setup_requires=["pytest-runner"],
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Python tools to subdivide cosmological simulations and tabulate synthetic observables",
    long_description="Python tools to subdivide cosmological simulations and tabulate synthetic observables",
    install_requires=["numpy"],
    url="https://github.com/ArgonneCPAC/thechopper"
)
