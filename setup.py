from setuptools import setup, find_packages

# Read the contents of requirements.txt to use in install_requires
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='story_reasoning',
    version='1.0',
    description='Framework to train and evaluate model on the Story Reasoning dataset',
    author='Daniel Oliveira',
    author_email='daniel.oliveira@inesc-id.pt',
    url='https://github.com/daniel3303/StoryReasoning',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8, <3.13',
    install_requires=requirements,
)
