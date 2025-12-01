from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer at startup
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_for_traditional(text):
    """Same cleaning function from your assignment"""
    text = re.sub(r'<.*?>', '', text)       # Remove HTML
    text = text.lower()                     # Converts to lowercase
    text = re.sub(r'[^a-z\s]', '', text)    # Removed all non-letters
    text = ' '.join(text.split())           # Remove extra whitespace
    return text

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>IMDB Sentiment Analysis API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container {
                max-width: 700px;
                margin: 0 auto;
            }
            .card {
                background: #1f1f1f;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,0,0,0.1);
                border: 2px solid #2a2a2a;
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .subtitle {
                color: #999;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            label {
                display: block;
                font-weight: 600;
                margin-bottom: 10px;
                color: #e0e0e0;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                padding: 15px;
                font-size: 16px;
                border: 2px solid #3a3a3a;
                background: #2a2a2a;
                color: #e0e0e0;
                border-radius: 12px;
                font-family: inherit;
                transition: border-color 0.3s;
                resize: vertical;
            }
            textarea:focus {
                outline: none;
                border-color: #cc0000;
            }
            textarea::placeholder {
                color: #666;
            }
            button {
                background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                font-size: 18px;
                font-weight: 600;
                margin-top: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 4px 15px rgba(211, 47, 47, 0.4);
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(211, 47, 47, 0.6);
            }
            button:active {
                transform: translateY(0);
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
                border: 1px solid #3a3a3a;
                border-radius: 12px;
                animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .result h3 {
                margin-bottom: 15px;
                color: #e0e0e0;
            }
            .sentiment-box {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 10px 0;
                font-size: 1.2em;
                color: #e0e0e0;
            }
            .positive {
                color: #ffd700;
                font-weight: bold;
                font-size: 1.5em;
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
            }
            .negative {
                color: #ff3333;
                font-weight: bold;
                font-size: 1.5em;
                text-shadow: 0 0 10px rgba(255, 51, 51, 0.3);
            }
            .confidence {
                font-size: 1.1em;
                color: #aaa;
            }
            .loading {
                text-align: center;
                color: #cc0000;
                font-size: 1.1em;
            }
            .emoji {
                font-size: 2em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🎬 IMDB Sentiment Analysis</h1>
                <p class="subtitle">Powered by TF-IDF + Logistic Regression</p>
                
                <form id="testForm">
                    <label for="text">Enter a Movie Review:</label>
                    <textarea id="text" rows="6" placeholder="Type your movie review here...">This movie was absolutely fantastic!</textarea>
                    <button type="submit">Analyze Sentiment</button>
                </form>
                
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('testForm').onsubmit = async (e) => {
                e.preventDefault();
                const text = document.getElementById('text').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<div class="result loading">🔍 Analyzing...</div>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    const result = await response.json();
                    
const sentimentClass = result.sentiment === 'positive' ? 'positive' : 'negative';
const confidencePercent = (result.confidence * 100).toFixed(1);

resultDiv.innerHTML = `
    <div class="result">
        <h3>Result</h3>
        <div class="sentiment-box">
            <span>Sentiment: <span class="${sentimentClass}">${result.sentiment.toUpperCase()}</span></span>
        </div>
        <p class="confidence">Confidence: <strong>${confidencePercent}%</strong></p>
    </div>
`;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result"><p style="color: #ff3333;">❌ Error: ${error.message}</p></div>`;
                }
            };
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Clean text using same function from training
        cleaned_text = clean_for_traditional(text)
        
        # Transform and predict
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Map prediction to sentiment
        sentiment = 'positive' if prediction == 'positive' else 'negative'
        confidence = float(max(probabilities))
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)