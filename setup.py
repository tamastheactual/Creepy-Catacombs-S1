from setuptools import setup, find_packages

setup(
    name='creepy_catacombs_s1',
    version='0.0.3',
    description='A discrete tunnel environment for RL experiments',
    author='Tamács Takács',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'pygame>=2.6.0',
        'gymnasium>=1.0.0',
        'numpy',
    ],
    python_requires='>=3.9',
)
