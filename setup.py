from setuptools import setup, find_packages

setup(
name='Base_Stein',
packages=find_packages(where='src'),
package_dir={'': 'src'},
    install_requires=["torch>=1.11.0",
                      "pytest>=7.1.2"]
)
