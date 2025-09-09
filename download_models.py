#!/usr/bin/env python3
"""
Model Download Script for Hugging Face Spaces Deployment
Downloads model files from Hugging Face Hub during build process.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face model repository
MODEL_REPO_ID = "tahmidul159/fake-news-models"

# Model files to download from Hugging Face
HUGGINGFACE_FILES = [
    "tfidf_vectorizer.pkl",
    "logistic_regression_model.pkl", 
    "random_forest_model.pkl",
    "gradient_boosting_model.pkl",
    "model_metadata.pkl",
    "bert_model/config.json",
    "bert_model/model.safetensors",
    "bert_tokenizer/special_tokens_map.json",
    "bert_tokenizer/tokenizer_config.json",
    "bert_tokenizer/vocab.txt"
]

def download_from_huggingface(filename, destination):
    """Download a file from Hugging Face Hub"""
    print(f"Downloading {filename} from Hugging Face...")
    
    try:
        # Download file from Hugging Face Hub
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            local_dir=os.path.dirname(destination),
            local_dir_use_symlinks=False
        )
        
        # Move to the correct destination if needed
        if downloaded_path != destination:
            os.rename(downloaded_path, destination)
        
        # Check file size
        file_size = os.path.getsize(destination)
        print(f"Downloaded {filename} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def download_models():
    """Download all model files from Hugging Face Hub"""
    print("Downloading models from Hugging Face Hub...")
    
    model_dir = Path("Saved Models")
    model_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_files = len(HUGGINGFACE_FILES)
    
    for filename in HUGGINGFACE_FILES:
        destination = model_dir / filename
        
        # Create parent directories if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if download_from_huggingface(filename, destination):
            success_count += 1
    
    print(f"Downloaded {success_count}/{total_files} files successfully")
    
    # Check if critical files were downloaded
    critical_files = ["tfidf_vectorizer.pkl", "logistic_regression_model.pkl", "bert_model", "bert_tokenizer"]
    critical_success = 0
    
    for critical_file in critical_files:
        if (model_dir / critical_file).exists():
            critical_success += 1
        else:
            print(f"Critical file missing: {critical_file}")
    
    print(f"Critical files downloaded: {critical_success}/{len(critical_files)}")
    
    # Return True only if all critical files are downloaded
    return critical_success == len(critical_files)

def create_model_placeholders():
    """Create placeholder files for missing models (fallback)"""
    model_dir = Path("Saved Models")
    model_dir.mkdir(exist_ok=True)
    
    # Create BERT directories
    (model_dir / "bert_model").mkdir(exist_ok=True)
    (model_dir / "bert_tokenizer").mkdir(exist_ok=True)
    
    print("Created placeholder directories for missing models")

def main():
    """Main function to set up models for deployment"""
    print("Setting up models for deployment...")
    
    # Check if models already exist locally
    model_dir = Path("Saved Models")
    if model_dir.exists() and any(model_dir.iterdir()):
        print("Local models found, skipping download")
    else:
        print("No local models found, downloading from Hugging Face Hub...")
        # Try to download from Hugging Face Hub
        if not download_models():
            print("Hugging Face Hub download failed!")
            print("Cannot continue without models. Exiting...")
            exit(1)
    
    print("Model setup complete!")

if __name__ == "__main__":
    main()