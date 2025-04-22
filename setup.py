from setuptools import find_packages, setup

install_requires = [
    'jupyterlab==4.4.0',
    'matplotlib==3.9.2',
    'networkx==3.1',
    'numpy==1.26.4',
    'PyYAML==6.0.2',
    'scikit_learn==1.5.2',
    'scipy==1.14.1',
    'seaborn==0.12.2',
    'tensorboard==2.18.0',
    'torch==2.2.2',
    'torchmetrics==1.4.1'
]

with open('README.md', 'r') as f:
    long_description = f.read()
    
setup(
    name='dnncgm',
    version='0.1.0',
    description='',
    long_description=long_description,
    context_type='text/markdown',
    url='',
    author='Jiahe Lin, Yikai Zhang, George Michailidis',
    author_email='jiahelin@umich.edu',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.10',
    zip_safe=False
)
