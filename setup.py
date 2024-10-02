from setuptools import setup, find_packages
import os
import sys

setup(
    name='Hybrid_Base',
    version="0.1",
    install_requires=['tequila-basic', 'qulacs', 'pyscf','openfermion','pytest'],
    url='https://github.com/JdelArco98/Hybrid_Base',
    author='Javier del Arco',
    author_email='francisco.del.arco.santos@uni-a.de',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src/Hybrid_Base')]
    }
)
