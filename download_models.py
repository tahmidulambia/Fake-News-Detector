#!/usr/bin/env python3
"""
Model Download Script for Render Deployment
Downloads model files from Google Drive during build process.
"""

import os
import requests
import zipfile
from pathlib import Path
import json

# Google Drive file IDs - you'll need to update these with your actual file IDs
GOOGLE_DRIVE_FILES = {
    # Individual model files
    "tfidf_vectorizer.pkl": "1vBO8rtnaQr4yiMQ4KVZ0FKmWQRbs-0jU",
    "gradient_boosting_model.pkl": "1NbkQ03QZnJATNQErBJ5BU_JmJSmib80d",
    "logistic_regression_model.pkl": "1oqLXlBrb3hIIVKppG9Yu9MZhPFg7rfkd", 
    "random_forest_model.pkl": "1MhqMkKVcVMrTMHM_-6cNA3t0zKY52oFC",
    "model_metadata.pkl": "1-doTMaxe_LdYfIOauEiG7LAAbb9BBpd2",
    
    # BERT model folder (zip file)
    "bert_model.zip": "1Mncmc9mSJs4bOmGc0Iu9HCHgNK92S7zN",
    "bert_tokenizer.zip": "1jzZvwBizqV7RZxe9PLT0E3E35za0uehj"
}

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive using file ID"""
    print(f"Downloading {destination} from Google Drive...")
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # First, try direct download with redirects enabled
        response = requests.get(url, stream=True, allow_redirects=True)
        
        # Check if we got a virus scan warning page
        if response.headers.get('content-type', '').startswith('text/html'):
            print(f"Large file detected, handling virus scan warning...")
            
            # Extract UUID from the HTML response
            html_content = response.text
            import re
            uuid_match = re.search(r'uuid" value="([^"]*)"', html_content)
            
            if uuid_match:
                uuid = uuid_match.group(1)
                print(f"Found UUID: {uuid}")
                
                # Use the proper download URL with UUID
                download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid}"
                response = requests.get(download_url, stream=True, allow_redirects=True)
            else:
                print(f"Could not extract UUID from virus scan warning page")
                return False
        
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check file size
        file_size = os.path.getsize(destination)
        print(f"Downloaded {destination} ({file_size:,} bytes)")
        
        # Additional check: if file is suspiciously small and we expected a large file
        if file_size < 1000 and 'bert_model' in str(destination):
            print(f"Warning: BERT model file is very small ({file_size} bytes), might be corrupted")
            return False
            
        return True
        
    except Exception as e:
        print(f"Failed to download {destination}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path}")
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        return True
        
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")
        return False

def download_models():
    """Download all model files from Google Drive"""
    print("Downloading models from Google Drive...")
    
    model_dir = Path("Saved Models")
    model_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_files = len(GOOGLE_DRIVE_FILES)
    
    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        if file_id == f"YOUR_{filename.upper().replace('.', '_').replace('/', '_')}_FILE_ID":
            print(f"Skipping {filename} - file ID not configured")
            continue
            
        destination = model_dir / filename
        
        if download_from_google_drive(file_id, destination):
            success_count += 1
            
            # Extract zip files
            if filename.endswith('.zip'):
                extract_dir = model_dir / filename.replace('.zip', '')
                if extract_zip(destination, extract_dir):
                    print(f"Successfully extracted {filename}")
                else:
                    success_count -= 1  # Count as failed if extraction failed
    
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
    
    # Check if we're in a deployment environment
    if os.environ.get('RENDER'):
        print("Running in Render environment")
        
        # Try to download from Google Drive
        if not download_models():
            print("Google Drive download failed!")
            print("Build cannot continue without models. Exiting...")
            exit(1)  # Fail the build
    else:
        print("Running locally - checking for existing model files...")
        
        # Check if models already exist locally
        model_dir = Path("Saved Models")
        if model_dir.exists() and any(model_dir.iterdir()):
            print("Local models found, skipping download")
        else:
            print("No local models found, creating placeholders...")
            create_model_placeholders()
    
    print("Model setup complete!")

if __name__ == "__main__":
    main()
