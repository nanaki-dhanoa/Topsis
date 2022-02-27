import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Topsis-Nanaki-101903195',  
    version='0.1.0',
    author="Nanaki",
    license='MIT',
    author_email="nanakidhanoa@gmail.com",
    description="Decision Making using topsis (Python Package)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = '',
    download_url = '',
    packages=setuptools.find_packages(),
    install_requires=[            
          'pandas',
          'numpy'
      ],
    classifiers=[
      'License :: OSI Approved :: MIT License', 
      'Programming Language :: Python :: 3.9',
    ],
 )