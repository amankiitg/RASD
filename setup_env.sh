#!/bin/bash

echo "Detecting platform..."

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    echo "Mac detected (${ARCH}) — using environment.yml"
    ENV_FILE="environment.yml"
    ENV_NAME="rasd"
else
    if command -v nvidia-smi &> /dev/null; then
        echo "Linux + NVIDIA GPU detected — using environment_gpu.yml"
        ENV_FILE="environment_gpu.yml"
        ENV_NAME="rasd-gpu"
    else
        echo "Linux CPU only — using environment.yml"
        ENV_FILE="environment.yml"
        ENV_NAME="rasd"
    fi
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — updating..."
    conda env update -f $ENV_FILE --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create -f $ENV_FILE
fi

echo ""
echo "Done. Now run:  conda activate ${ENV_NAME}"
