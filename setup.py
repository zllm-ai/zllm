from setuptools import setup, find_packages

setup(
    name="ultimate_vllm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "custom_vllm=vllm.src.main:main",
            "custom_vllm_advanced=vllm.src.main_advanced:main",
            "custom_vllm_server=vllm.src.api_server:app"
        ]
    }
)