from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='fastfusion',
        version='0.0.1',
        author='Michael Gilbert, Tanner Andrulis',
        author_email='gilbertm@mit.edu, andrulis@mit.edu',
        install_requires=[
            'accelergy>=0.4',
            'ruamel.yaml',
            'psutil',
            'joblib',
            'argparse',
        ],
        packages=find_packages(),
        zip_safe=True,
        entry_points={
            'console_scripts': []
        }
    )
