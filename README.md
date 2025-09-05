# Fake News Detector - Machine Learning Web Application

A production-ready web application that uses machine learning to classify news articles as real or fake. Built with Python, Flask, and Scikit-learn.

## 🚀 Features

- **Multiple ML Models**: Logistic Regression, Gradient Boosting, Random Forest, and K-Nearest Neighbors
- **High Accuracy**: Up to 82.56% accuracy on test data
- **Real-time Analysis**: Instant classification with confidence scores
- **Professional UI**: Modern, responsive web interface
- **API Endpoints**: RESTful API for integration
- **Model Selection**: Choose between different ML algorithms
- **Confidence Metrics**: Detailed probability breakdowns

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Logistic Regression | 82.56% | 0.8285 | 0.8256 | 0.8259 | ✅ Recommended |
| Gradient Boosting | 82.27% | 0.8225 | 0.8227 | 0.8225 | ✅ Alternative |
| Random Forest | 81.45% | 0.8525 | 0.8145 | 0.8123 | ✅ Alternative |
| K-Nearest Neighbors | 79.17% | 0.7923 | 0.7917 | 0.7906 | ❌ Reference |

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, TF-IDF Vectorization
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Deployment**: Render 
- **Models**: Logistic Regression, Gradient Boosting, Random Forest, KNN

## 📁 Project Structure

```
Fake News Predictor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
├── render.yaml           # Render deployment configuration
├── README.md             # Project documentation
├── templates/
│   └── index.html        # Web interface
├── Saved Models/         # Trained ML models
│   ├── logistic_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   ├── knn_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.pkl
├── Data Files/
│   ├── Fake.csv          # Fake news dataset
│   ├── True.csv          # Real news dataset
│   └── fake_news_dataset.csv  # Additional dataset
└── project.ipynb         # Training pipeline notebook
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-news-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 🔧 API Endpoints

### POST `/api/predict`
Classify news article text as fake or real.

**Request Body:**
```json
{
  "text": "Your news article content here...",
  "model": "logistic_regression"
}
```

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 95.67,
  "probabilities": {
    "fake": 95.67,
    "real": 4.33
  },
  "model_used": "logistic_regression"
}
```

### GET `/api/models`
Get list of available models.

**Response:**
```json
{
  "available_models": ["logistic_regression", "gradient_boosting", "random_forest", "knn"],
  "default_model": "logistic_regression"
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "vectorizer_loaded": true
}
```

## 📈 Model Training Details

The models were trained on a comprehensive dataset of 57,560+ news articles:
- **Fake News**: 26,429 articles from unreliable sources
- **Real News**: 31,131 articles from reliable sources (Reuters, BBC, etc.)
- **Additional Dataset**: 20,000 articles for improved training

**Preprocessing:**
- Aggressive text cleaning and normalization
- TF-IDF vectorization (10,000 features with trigrams)
- Train/test split (80/20)
- Dataset balancing and deduplication

**Training Process:**
- Cross-validation for hyperparameter tuning
- Model evaluation using precision, recall, and F1-score
- Model serialization for production deployment


## 🔒 Privacy & Security

- No user data is stored or logged
- All processing happens in memory
- HTTPS encryption on Render deployment
- Input validation and sanitization



## 📝 License

This project is open source and available under the [MIT License](LICENSE).




## 📊 Datasets

- [Fake News Detection on Kaggle](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)
- [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset)

---

**Note**: This application is for educational and demonstration purposes. Always verify information from multiple reliable sources.

