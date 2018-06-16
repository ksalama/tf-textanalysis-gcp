import setuptools

requirements = ['tensorflow>=1.7.0', 'tensorflow-hub', 'nltk', 'tensorflow-transform']

setuptools.setup(
    name='df-textanalysis-tft-tfhub',
    install_requires=requirements,
    version='0.1',
    packages=setuptools.find_packages(),
    py_modules=['parameters']
)