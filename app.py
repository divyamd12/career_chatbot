from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
import webbrowser

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)



lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def search_chatgpt(query):
    query = query.replace(" ", "+")
    url = f"https://www.chatgpt.com/?q={query}"
    try:
        webbrowser.open(url)
    except Exception as e:
        print("Error opening web browser:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    print(message)

    if message.lower() == 'exit':
        response = "Bot: Goodbye! Have a great day!"
    else:
        ints = predict_class(message)
        print(ints)
        if not ints:
            response = "Bot: I'm sorry, I couldn't understand your query"
        else:
            response = get_response(ints, intents)
    # Process the message and generate a response using your chatbot backend
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
