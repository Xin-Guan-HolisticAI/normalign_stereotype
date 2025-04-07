from setuptools import setup, find_packages

setup(
    name="normalign_stereotype",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-dotenv",
        "pyyaml",
        "textblob",
        "nltk",
    ],
    author="Xin Guan",
    author_email="xin.guan@holisticai.com",
    description="A package for analyzing and normalizing stereotypes in text",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Xin-Guan-HolisticAI/normalign_stereotype",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 