import glob
from setuptools import setup, find_packages

setup(
    name='lplangid',  # Required.
    version='0.1.0',  # Required. Use major.minor.dev format.
    description='LivePerson language detection package using a Reciprocal Rank Classifier',  # Required
    packages=find_packages(),
    include_package_data=True,
    data_files=[('lplangid/freq_data', glob.glob('lplangid/freq_data/*.csv'))],
    install_requires=['flake', 'pytest'],  # List new package requirements here, but please be sure you need them!

    url='',  # Optional
    author='Dominic Widdows, Chris Brew',  # Optional
    author_email='dwiddows@gmail.com,christopher.brew@gmail.com',   # Optional
    classifiers=[  # Optional
        'Intended Audience :: NLP developers',
        'Topic :: Natural Language Processing :: Language Detection',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='language detection, language classification, reciprocal rank classifier, lplangid'
)
