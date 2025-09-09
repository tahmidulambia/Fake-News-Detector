
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Global variables to store loaded models
models = {}
tfidf_vectorizer = None
bert_model = None
bert_tokenizer = None
bert_device = None

def load_models():
    """Load all saved models and vectorizer"""
    global models, tfidf_vectorizer, bert_model, bert_tokenizer, bert_device
    
    model_path = "Saved Models"
    
    if not os.path.exists(model_path):
        print(f"Model directory {model_path} does not exist!")
        return False
    
    try:
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(model_path, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            print("Loaded TF-IDF vectorizer")
        else:
            print("TF-IDF vectorizer not found - traditional models will be disabled")
            tfidf_vectorizer = None
        
        # Load all models
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl', 
            'gradient_boosting': 'gradient_boosting_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            file_path = os.path.join(model_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
            else:
                print(f"Skipping {model_name} model (file not found: {filename})")
        
        # Load BERT model and tokenizer
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch
            
            bert_model_path = os.path.join(model_path, "bert_model")
            bert_tokenizer_path = os.path.join(model_path, "bert_tokenizer")
            
            if os.path.exists(bert_model_path) and os.path.exists(bert_tokenizer_path):
                # Set device to CPU for Render deployment (no GPU)
                device = torch.device('cpu')
                
                global bert_tokenizer, bert_model, bert_device
                bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_tokenizer_path)
                bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
                bert_model.to(device)  # Move to CPU
                bert_model.eval()  # Set to evaluation mode
                
                # Optimize model for inference (skip if not available)
                try:
                    if hasattr(torch.jit, 'optimize_for_inference'):
                        bert_model = torch.jit.optimize_for_inference(bert_model)
                except Exception:
                    # Continue without optimization
                    pass
                
                # Store device globally for predictions
                bert_device = device
                
                print("BERT model and tokenizer loaded successfully")
            else:
                print("BERT model files not found, skipping BERT loading")
        except ImportError:
            print("Transformers library not available, skipping BERT loading")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
                
        print(f"Successfully loaded {len(models)} traditional models")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def preprocess_text(text):
    """Clean and preprocess the input text to match training preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove location prefixes from Reuters articles
    text = re.sub(r'^[A-Z\s]+\(Reuters\)\s*-\s*', '', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove numbers with units
    text = re.sub(r'\d+\s+(people|years|percent|million|billion|thousand|dollars)', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short words
    text = ' '.join([word for word in text.split() if len(word) > 2])
    
    return text.strip()

def predict_fake_news(text, model_name='random_forest'):
    """Predict if the given text is fake or real news"""
    # Handle BERT model
    if model_name == 'bert':
        if bert_model is None or bert_tokenizer is None:
            return {"error": "BERT model not available"}
        
        try:
            import torch
            
            # Tokenize text for BERT with optimized settings
            inputs = bert_tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=True,
                padding=True,  # Dynamic padding instead of max_length
                truncation=True,
                max_length=128,  # Reduced from 256 to 128 for faster processing
                return_tensors='pt'
            )
            
            # Move inputs to the same device as the model
            device = bert_device if 'bert_device' in globals() else torch.device('cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
                prediction = torch.argmax(logits, dim=1)[0].item()
            
            # Check if BERT prediction is reasonable (not overfitted)
            max_prob = max(probabilities).item()
            if max_prob > 0.98:  # Increased threshold from 95% to 98% to reduce fallbacks
                # Fall back to Logistic Regression for more reliable results
                print("⚠️  BERT showing very high confidence - may be overfitted, using Logistic Regression instead")
                fallback_result = predict_fake_news(text, 'logistic_regression')
                # Add fallback indicator
                if 'model_used' in fallback_result:
                    fallback_result['model_used'] = fallback_result['model_used'] + " (Fallback - BERT overfitting detected)"
                return fallback_result
            
            # Convert prediction (0=Real, 1=Fake)
            # Based on the training data: 0=Real, 1=Fake
            if prediction == 1:
                prediction_label = "Fake"
                fake_prob = probabilities[1].item()
                real_prob = probabilities[0].item()
            else:
                prediction_label = "Real"
                fake_prob = probabilities[1].item()  # Fixed: was probabilities[0]
                real_prob = probabilities[0].item()  # Fixed: was probabilities[1]
            
            confidence = max(probabilities).item() * 100
            
            return {
                "prediction": prediction_label,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "fake": round(fake_prob * 100, 2),
                    "real": round(real_prob * 100, 2)
                },
                "model_used": "bert"
            }
            
        except Exception as e:
            return {"error": f"BERT prediction failed: {str(e)}"}
    
    # Handle traditional models
    if model_name not in models:
        return {"error": f"Model {model_name} not found"}
    
    if tfidf_vectorizer is None:
        return {"error": "TF-IDF vectorizer not available - traditional models disabled"}
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Vectorize text
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    
    # Make prediction
    model = models[model_name]
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Get confidence score
    confidence = max(probabilities) * 100
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {
            "fake": round(probabilities[0] * 100, 2),
            "real": round(probabilities[1] * 100, 2)
        },
        "model_used": model_name
    }

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for fake news prediction"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        model_name = data.get('model', 'logistic_regression')
        
        if len(text.strip()) < 10:
            return jsonify({"error": "Text too short. Please provide at least 10 characters."}), 400
        
        result = predict_fake_news(text, model_name)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    available_models = list(models.keys())
    
    # Add BERT if available
    if bert_model is not None and bert_tokenizer is not None:
        available_models.append('bert')
    
    return jsonify({
        "available_models": available_models,
        "default_model": "logistic_regression"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": len(models),
        "vectorizer_loaded": tfidf_vectorizer is not None
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check model files and environment"""
    import os
    from pathlib import Path
    
    model_path = Path("Saved Models")
    debug_info = {
        "current_directory": os.getcwd(),
        "environment": "PRODUCTION",
        "model_directory_exists": model_path.exists(),
        "model_directory_contents": [],
        "models_loaded": list(models.keys()),
        "vectorizer_loaded": tfidf_vectorizer is not None,
        "bert_model_loaded": bert_model is not None,
        "bert_tokenizer_loaded": bert_tokenizer is not None,
        "total_models_loaded": len(models),
        "load_models_success": False
    }
    
    if model_path.exists():
        try:
            debug_info["model_directory_contents"] = [str(p) for p in model_path.iterdir()]
            debug_info["model_directory_size"] = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        except Exception as e:
            debug_info["model_directory_error"] = str(e)
    else:
        debug_info["model_directory_error"] = "Directory does not exist"
    
    # Test if we can load models
    try:
        test_load = load_models()
        debug_info["load_models_success"] = test_load
    except Exception as e:
        debug_info["load_models_error"] = str(e)
    
    return jsonify(debug_info)

# Load models when the module is imported (for production)
print("Loading models...")
if not load_models():
    print("Failed to load models!")
    exit(1)
print("Models loaded successfully!")

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    print(f"Starting Flask app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
