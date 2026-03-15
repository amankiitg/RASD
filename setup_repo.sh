#!/bin/bash

# Create folder structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p src/models
mkdir -p src/baselines
mkdir -p src/analysis
mkdir -p src/utils
mkdir -p configs
mkdir -p notebooks
mkdir -p results/baselines
mkdir -p results/ablations
mkdir -p results/final
mkdir -p figures
mkdir -p tables
mkdir -p manuscript
mkdir -p literature_review

# GitHub requires at least one file per folder to track it
# Create .gitkeep in all leaf directories
find . -type d -not -path './.git/*' | while read dir; do
    if [ -z "$(ls -A "$dir" 2>/dev/null | grep -v '.gitkeep')" ]; then
        touch "$dir/.gitkeep"
    fi
done

# Create README.md files where specified
touch data/README.md
touch README.md

# Create placeholder src files
touch src/run_experiment.py

# Create placeholder config/env files
touch environment.yml
touch requirements.txt

echo "✓ Folder structure created successfully"
