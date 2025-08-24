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
            "ultimate_vllm=vllm.src.ultimate_cli:main",
            "custom_vllm=vllm.src.main:main",
            "enhanced_vllm=vllm.src.enhanced_cli:main",
            "custom_vllm_server=vllm.src.api_server:app"
        ]
    }
)