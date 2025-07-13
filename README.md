# MLOps Bad Buzz Detector 🚨

## Project: Deep Learning for Sentiment Analysis

### 📋 Project Overview
An end-to-end MLOps project implementing a sentiment analysis system to detect potential "bad buzz" on social media. Built for Air Paradis (fictional airline) to monitor and anticipate negative publicity on Twitter.

### 🎯 Objectives
- Build a binary sentiment classifier for tweets
- Implement both custom and pre-trained models
- Deploy the model with a production-ready API
- Create comprehensive documentation and presentation

### 📊 Dataset
- **Source**: Sentiment140 dataset from Stanford
- **Size**: 1.6 million tweets (8,000 used for prototype)
- **Classes**: Binary (Positive/Negative sentiment)
- **Features**: Raw tweet text with emoticons

### 🛠️ Technical Stack
- **Python 3.x**
- **Deep Learning Frameworks**:
  - TensorFlow/Keras
  - Transformers (BERT)
- **NLP Libraries**:
  - NLTK
  - contractions
  - TweetTokenizer
- **Deployment**:
  - Flask (API)
  - Docker (containerization)
- **Embeddings**:
  - GloVe
  - FastText

### 📁 Repository Structure
```
mlops-badbuzz-detector/
├── data/                                             # Data directory
├── Derraz_Sofiane_1_1_modeles_sur_mesure_avancé_052022.ipynb   # Custom models notebook
├── Derraz_Sofiane_1_2_modele_sur_etagere.ipynb     # Pre-trained models notebook
├── Derraz_Sofiane_2_scripts_deploiement_052022.ipynb           # Deployment scripts
├── score.py                                          # Scoring/inference script
├── Derraz_Sofiane_3_article_de_blog_052022.pdf     # Technical blog article
├── Derraz_Sofiane_4_presentation_052022.pptx        # Project presentation
└── README.md                                         # This file
```

### 🤖 Models Implemented

#### 1. Custom Models
- **SimpleRNN**: Baseline recurrent model
- **LSTM**: Long Short-Term Memory for better context
- **Custom architectures** with different embedding strategies

#### 2. Pre-trained Models
- **BERT**: State-of-the-art transformer model
- Fine-tuned for sentiment analysis task

### 🔧 Text Preprocessing Pipeline
```python
def clean_tweet(tweet):
    # Expand contractions
    tweet = contractions.fix(tweet)
    
    # Tokenization
    tokenizer = TweetTokenizer()
    
    # Remove URLs, mentions, hashtags
    # Convert to lowercase
    # Remove special characters
    
    return cleaned_text
```

### 📈 Key Features
- **Robust preprocessing**: Handles contractions, URLs, mentions
- **Multiple model comparison**: Custom vs pre-trained
- **Production-ready API**: Flask endpoint for real-time predictions
- **Comprehensive documentation**: Blog post and presentation included

### 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/DerrazSofiane/mlops-badbuzz-detector.git
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow transformers nltk contractions flask
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Run the notebooks in order:
   - Start with custom models notebook
   - Then pre-trained models
   - Finally deployment scripts

5. Launch the API:
   ```bash
   python score.py
   ```

### 🎯 API Usage
```python
# Example API call
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={'tweet': 'Your tweet text here'}
)
print(response.json())
```

### 📊 Performance
- Binary classification accuracy achieved
- Real-time inference capability
- Scalable deployment architecture

### 🔄 MLOps Best Practices
- Model versioning
- Automated testing
- Continuous integration ready
- Monitoring and logging capabilities

### 📝 Skills Demonstrated
- Deep Learning (RNN, LSTM, Transformers)
- Natural Language Processing
- MLOps and deployment
- API development
- Technical documentation

### 🚀 Future Improvements
- Multi-class sentiment analysis
- Real-time streaming integration
- A/B testing framework
- Model monitoring dashboard
- Multi-language support

### 📚 Documentation
- See the included blog article PDF for detailed methodology
- PowerPoint presentation available for business stakeholders

---
*Note: This educational project demonstrates production-ready ML engineering practices applicable to real-world brand monitoring and crisis prevention systems.*
