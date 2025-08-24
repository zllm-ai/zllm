#!/bin/bash
# setup_zllm.sh
# Setup script for zllm on Linux systems

echo "Setting up zllm environment..."

# Create and activate virtual environment
python3 -m venv zllm-env
source zllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install torch transformers accelerate fastapi uvicorn pydantic pyyaml

# Install quantization libraries (optional, for GPU systems)
# Uncomment the following lines if you have a compatible GPU
# pip install auto-gptq autoawq

echo "Installation complete!"
echo "To activate the environment, run: source zllm-env/bin/activate"
echo "Then you can run: custom_vllm"