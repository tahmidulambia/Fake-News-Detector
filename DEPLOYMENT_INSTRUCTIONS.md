# Deployment Instructions for Fake News Detector

## GitHub Repository Setup

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Set repository name: `Fake-News-Detector`
5. Set description: `Machine Learning-powered fake news detection web application with Flask and 4 trained models`
6. Make it **Public** (so Render can access it)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

### Step 2: Connect and Push Your Code
After creating the repository, run these commands in your terminal:

```bash
# Add the GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/Fake-News-Detector.git

# Push your code to GitHub
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Render Deployment Setup

### Step 1: Connect to Render
1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" and select "Web Service"
3. Connect your GitHub account if not already connected
4. Select the `Fake-News-Detector` repository

### Step 2: Configure Deployment Settings
- **Name**: `fake-news-detector` (or your preferred name)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Plan**: Free (or upgrade if needed)

### Step 3: Environment Variables (if needed)
The app should work with default settings, but you can add:
- `PYTHON_VERSION`: `3.11.5`

### Step 4: Deploy
Click "Create Web Service" and wait for deployment to complete.

## Files Included in Repository

✅ **Core Application Files:**
- `app.py` - Main Flask application
- `templates/index.html` - Web interface
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `runtime.txt` - Python version specification

✅ **Trained Models:**
- `Saved Models/` - All 4 trained ML models
- `Saved Models/bert_model/` - BERT transformer model
- `Saved Models/bert_tokenizer/` - BERT tokenizer
- Traditional models: Logistic Regression, Random Forest, Gradient Boosting, KNN

✅ **Training Data:**
- `Data Files/` - Complete training datasets
- `project.ipynb` - Training notebook with full ML pipeline

✅ **Documentation:**
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

## Model Performance Summary

| Model | Speed | Accuracy | Status |
|-------|-------|----------|---------|
| **Logistic Regression** | ~0.002s | 50% | ✅ **Default** |
| **Gradient Boosting** | ~0.001s | 50% | ✅ **Alternative** |
| **Random Forest** | ~0.01s | 40% | ⚠️ **Experimental** |
| **BERT** | ~2.5s | 40% | ✅ **Optimized** |

## Features

- 🚀 **Fast Response Times** - Traditional models respond in milliseconds
- 🤖 **4 ML Models** - Multiple algorithms for comparison
- 🎯 **BERT Integration** - Advanced transformer model with overfitting detection
- 📱 **Responsive UI** - Modern, mobile-friendly interface
- ⚡ **Optimized Performance** - BERT model optimized for web deployment
- 🔄 **Fallback Mechanism** - Automatic fallback when BERT shows overfitting
- 📊 **Confidence Scores** - Detailed probability breakdowns
- ⚠️ **User Warnings** - Clear disclaimers about model limitations

## API Endpoints

- `GET /` - Web interface
- `POST /api/predict` - Classify news articles
- `GET /api/models` - List available models
- `GET /api/health` - Health check

## Deployment Checklist

- [ ] GitHub repository created and code pushed
- [ ] Render account connected to GitHub
- [ ] Web service created on Render
- [ ] Environment variables configured (if needed)
- [ ] Deployment successful
- [ ] Health check endpoint responding
- [ ] Web interface accessible
- [ ] All models loading correctly
- [ ] API endpoints working

## Troubleshooting

### Common Issues:
1. **Models not loading**: Check file paths in `app.py`
2. **Slow BERT responses**: This is expected (2-3 seconds)
3. **Memory issues**: Consider upgrading Render plan for larger models
4. **Build failures**: Check `requirements.txt` for dependency conflicts

### Support:
- Check Render logs for deployment issues
- Verify all model files are present in repository
- Test locally before deploying
