from setuptools import setup, find_packages

setup(
    name='MLops',          
    version='0.0.0',                   # Version number
    author='Parsa YoussefPour',                # Optional: your name
    author_email='pyoussefpour@gmail.com',     # Optional: your email
    description='Youtube comment sentiment analysis',
    packages=find_packages(),          # Automatically find all packages (i.e., folders with __init__.py)
    install_requires=[],               # You can list dependencies here or use requirements.txt separately
)