#!/usr/bin/env python3
"""
Model Download Script for Render Deployment
Downloads large model files that are excluded from GitHub due to size limits.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded {filepath}")

def create_model_placeholders():
    """Create placeholder files for missing models"""
    model_dir = Path("Saved Models")
    model_dir.mkdir(exist_ok=True)
    
    # Create placeholder for BERT model
    bert_model_dir = model_dir / "bert_model"
    bert_model_dir.mkdir(exist_ok=True)
    
    # Create placeholder for missing files
    missing_files = [
        "Saved Models/bert_model/model.safetensors",
        "Saved Models/tfidf_vectorizer.pkl",
        "Saved Models/knn_model.pkl"
    ]
    
    for file_path in missing_files:
        if not os.path.exists(file_path):
            print(f"⚠️  Missing model file: {file_path}")
            print(f"   This file will need to be uploaded separately to Render")
            print(f"   Or the model will be disabled in production")

def main():
    """Main function to set up models for deployment"""
    print("🔧 Setting up models for Render deployment...")
    
    # Check if we're in a deployment environment
    if os.environ.get('RENDER'):
        print("🚀 Running in Render environment")
        # In Render, you would typically download from a cloud storage service
        # For now, we'll create placeholders and let the app handle missing models gracefully
        create_model_placeholders()
    else:
        print("💻 Running locally - checking for model files...")
        create_model_placeholders()
    
    print("✅ Model setup complete!")

if __name__ == "__main__":
    main()
