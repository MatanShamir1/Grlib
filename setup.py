from setuptools import setup, find_packages

setup(
    name='grlib',  # Replace with your package name
    version='0.1',
    packages=find_packages(),
    python_requires="==3.11",
    install_requires=[
        'gr_libs',
        'dill',
        'opencv-python'
    ],
    include_package_data=True,
    description='Package with goal recognition frameworks baselines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MatanShamir1/Grlib/',  # Replace with your repository URL
    author='Matan Shamir',
    author_email='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
