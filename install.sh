#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


# Create and enter Nextflow directory
mkdir -p ~/nextflow
cd ~/nextflow

# Download and install Nextflow
echo "Installing Nextflow..."
curl -s https://get.nextflow.io | bash
chmod +x nextflow

# Add Nextflow to PATH if not already added
if ! grep -Fxq 'export PATH=$HOME/nextflow:$PATH' ~/.bashrc; then
  echo 'export PATH=$HOME/nextflow:$PATH' >> ~/.bashrc
  echo "Added Nextflow to PATH in ~/.bashrc"
else
  echo "Nextflow path already present in ~/.bashrc"
fi

# Reload bash configuration
source ~/.bashrc

# Verify installation
echo "Verifying Nextflow installation..."
./nextflow -version

