from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='fastfusion',
        version='0.0.1',
        author='Michael Gilbert, Tanner Andrulis',
        author_email='gilbertm@mit.edu, andrulis@mit.edu',
        install_requires=[
            'ruamel.yaml',
            'joblib',
            'pandas',
            'numpy',
            'pydantic>=2.0.0',
            'pydantic-yaml>=0.11.0',
            'pyyaml>=6.0.0',
            'tqdm',
            'jinja2',
            'islpy',
            # 'combinatorics',
            'sympy',
            # 'paretoset',
            'fast-pareto',
            
        ],
        packages=find_packages(),
        zip_safe=True,
        entry_points={
            'console_scripts': []
        }
    )
