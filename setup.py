from setuptools import setup, find_packages
import versioneer

setup(
    name='grlib',  # Replace with your package name
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.11",
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
