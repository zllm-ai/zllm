from setuptools import setup, find_packages

setup(
    name="custom_vllm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "custom_vllm=src.main:main"
        ]
    }
)