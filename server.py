from flask import Flask, render_template, send_file, request, jsonify, abort, json
from flask_cors import CORS
from ttsAPI import getTTS
import numpy as np
from io import BytesIO
import base64
from queue import Queue, Empty
import time
import threading

import torch
from utils import hparams as hp
from models.tacotron import Tacotron
from models.fatchord_version import WaveRNN
from utils.text.symbols import symbols
import zipfile, os


app = Flask(__name__, template_folder="./templates/")
cors = CORS(app)

###########################################################################################################

os.makedirs('quick_start/tts_weights/', exist_ok=True)
os.makedirs('quick_start/voc_weights/', exist_ok=True)

zip_ref = zipfile.ZipFile('pretrained/ljspeech.wavernn.mol.800k.zip', 'r')
zip_ref.extractall('quick_start/voc_weights/')
zip_ref.close()

zip_ref = zipfile.ZipFile('pretrained/ljspeech.tacotron.r2.180k.zip', 'r')
zip_ref.extractall('quick_start/tts_weights/')
zip_ref.close()

try:
    hp.configure('hparams.py')  # Load hparams from file
except:
    pass

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device:', device)

print('\nInitialising WaveRNN Model...\n')

# Instantiate WaveRNN Model
voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode='MOL').to(device)

voc_model.load('quick_start/voc_weights/latest_weights.pyt')

print('\nInitialising Tacotron Model...\n')

# Instantiate Tacotron Model
tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                        num_chars=len(symbols),
                        encoder_dims=hp.tts_encoder_dims,
                        decoder_dims=hp.tts_decoder_dims,
                        n_mels=hp.num_mels,
                        fft_bins=hp.num_mels,
                        postnet_dims=hp.tts_postnet_dims,
                        encoder_K=hp.tts_encoder_K,
                        lstm_dims=hp.tts_lstm_dims,
                        postnet_K=hp.tts_postnet_K,
                        num_highways=hp.tts_num_highways,
                        dropout=hp.tts_dropout,
                        stop_threshold=hp.tts_stop_threshold).to(device)


tts_model.load('quick_start/tts_weights/latest_weights.pyt')

###########################################################################################################

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
            batch_outputs = []
            for request in requests_batch:
                batch_outputs.append(run(request['input'][0], request['input'][1]))

            for request, output in zip(requests_batch, batch_outputs):
                request['output'] = output
                
threading.Thread(target=handle_requests_by_batch).start()

def run(input_text, batched):
    try:
        wav_file = getTTS(input_text, batched, voc_model, tts_model, hp)
    except Exception as e:
        print(e)
        return 500

    save_path = wav_file[1]

    with open(save_path, 'rb') as wav:
        wav_bytes = wav.read()
    
    wav_io = BytesIO(wav_bytes)
    wav_io.seek(0)

    if os.path.exists(save_path):
        os.remove(save_path)
    
    return wav_io


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/tts", methods=["POST"])
def generateTTS():
    try:
        input_text = request.form["input_text"]
        batched = request.form["batched"] 
    except: 
        return jsonify({"error": "Please check input text or selected option"}), 400

    input_text = preprocessing(input_text)

    status = input_handling(input_text)

    if status == 400:
        return jsonify({"error":"Input text error"}), 400

    if requests_queue.qsize() >= BATCH_SIZE:
        return jsonify({"error":'Too Many Request'}), 429

    req = {
        'input': [input_text, batched]
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    if req['output'] == 500:
        return jsonify({'error': 'Generate TTS error!'}), 500
    
    return send_file(req['output'], mimetype="audio/wav")


def input_handling(input_text):
    splited = input_text.split(' ')
    len_txt = ""

    for word in splited:
        len_txt += word

    if len(len_txt) == 1:
        if len_txt[-1] == '.': 
            return 400 
    
    return 200


def preprocessing(input_text):
    texts = input_text.split(" ")

    result = ""
    for text in texts[:-1]:
        result += text + " "
    
    result += texts[-1]
    if texts[-1] not in [".", "?", "!"]:
        result += "."
    
    return result

@app.route("/healthz", methods = ["GET"])
def healthCheck():
    return "ok",200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="80")