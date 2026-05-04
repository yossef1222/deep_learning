#!/bin/bash
# Quick Start Script
# Automates the setup process

echo "=================================="
echo "Deep Learning Project - Quick Setup"
echo "=================================="
echo ""

# Check Python
echo "[1/5] Checking Python..."
python3 --version

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install -r requirements.txt

# Run setup check
echo "[3/5] Verifying setup..."
python3 setup_check.py

# Download datasets
echo "[4/5] Downloading datasets..."
read -p "Download datasets now? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 download_data.py
fi

# Run project
echo "[5/5] Ready to run project!"
echo ""
echo "To start training, run:"
echo "  python main.py"
echo ""
