from flask import Flask, request, jsonify, url_for, render_template
import pickle
import re
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

# url_for('static', filename='style.css')

model = pickle.load(open('models/model.sav', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)

    return processed_text

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['GET'])
def predict():
    text = request.args.get('text', '')
    if text == '':
        return jsonify({"output": -1})

    text = preprocess_text(text)
    text = vectorizer.transform([text])
    output = int(model.predict(text)[0])

    return jsonify({"output": output})

if __name__ == '__main__':
    app.run(debug=True)
