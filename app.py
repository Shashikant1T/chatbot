from flask import Flask, request, jsonify, render_template
import json
import random
import datetime
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load intents
with open("intents.json", encoding="utf-8") as f:
    data = json.load(f)

patterns, labels, responses = [], [], {}
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, labels)

def get_weather(city="Pune"):
    API_KEY = "8215869df4c718eb8cfa0c35c3c8c085"  # Replace with your key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url)
        data = res.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The weather in {city} is {desc} with {temp}°C 🌦️"
        else:
            return "Sorry, I couldn’t fetch the weather right now."
    except:
        return "Error fetching weather."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_input = request.json["message"]
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]

    if prediction == "time":
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        response = f"The current time is {current_time} ⏰"
    elif prediction == "date":
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        response = f"Today's date is {today} 📅"
    elif prediction == "weather":
        response = get_weather("Pune")
    else:
        response = random.choice(responses[prediction])

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
