from flask import Flask, render_template, send_file, request, make_response
from flask_cors import CORS
from ttsAPI import getTTS
import numpy as np
from io import BytesIO



app = Flask(__name__, template_folder="./templates/")
cors = CORS(app)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/tts", methods=["POST"])
def generateTTS():

    input_text = request.form["input_text"]
    batched = request.form["batched"] 

    input_text = preprocessing(input_text)

    wav_file = getTTS(input_text, batched)
    save_path = wav_file[1]
    wav_file = wav_file[0]

    return send_file(save_path, mimetype="audio/wav")

def preprocessing(input_text):
    texts = input_text.split(" ")

    result = ""
    for text in texts[:-1]:
        result += text + " "
    
    result += texts[-1]
    if texts[-1] not in [".", "?", "!"]:
        result += "."
    
    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="80", debug=True)