from setuptools import setup

setup(
        name='asym_uncertainty',
        version='0.6.0',
        description='Algebra for quantities with asymmetric uncertainties using a Monte-Carlo uncertainty propagation method.',
        url='http://github.com/uga-uga/asym_uncertainty',
        author='Udo Gayer',
        author_email='gayer.udo@gmail.com',
        license='GPLv3',
        python_requires='>=3',
        packages=['asym_uncertainty'],
        install_requires=['numpy', 'mc_statistics>=0.3'],
        setup_requires=['pytest-runner'],
        tests_require=['pytest', 'pytest-cov', 'numpy', 'matplotlib'],
)
