import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="siren-torch",
    version="1.0",
    author="Aman Dalmia",
    author_email="amandalmia18@gmail.com",
    description="PyTorch implementation of Sinusodial Representation networks (SIREN)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dalmia/siren",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.2',
)
