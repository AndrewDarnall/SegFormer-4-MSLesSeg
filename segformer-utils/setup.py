from setuptools import setup, find_packages

setup(
    name="SegFormer3D Utilities",
    version="0.1.0",
    author="Andrew Ryan Darnall, Giovani Spadaro",
    author_email="andrew.ryan.darnall@gmail.com, giovannispada17@protonmail.com",
    description="Customized code of the original SegFormer3D repo for our own study on Multiple Sclerosis MRI Segementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/my_package",
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=["numpy", "pandas"],  # Dependencies if any
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
