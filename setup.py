from setuptools import setup, find_packages


setup(
    name="feiertag",
    version="0.1.0rc1",
    author="Elijah Rippeth",
    author_email="elijah.rippeth@gmail.com",
    url="https://github.com/erip/feiertag",
    description="An open-source neural sequence tagging toolkit.",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*",)),
    zip_safe=True,
    python_requires='>=3.6,<4',
    install_requires=[
        "dataclasses",
        "hydra-core",
        "torch",
        "pytorch_lightning",
    ],
    test_requires=[
        "pytest",
        "pytest-cov",
        "hypothesis"
    ],
    entry_points={
        'console_scripts': [
            'feiertag-train=feiertag.cli.train:main'
        ],
    }
)

