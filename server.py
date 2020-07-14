from flask import Flask, render_template, send_file, request, jsonify, abort
from flask_cors import CORS
from ttsAPI import getTTS
import numpy as np
from io import BytesIO

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

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/tts", methods=["POST"])
def generateTTS():

    try:
        input_text = request.form["input_text"]
        batched = request.form["batched"] 
    except: 
        return "", "400 Please Input text or Select option"

    input_text = preprocessing(input_text)

    status = input_handling(input_text)

    if status == 400:
        return "", "400 Input text error"

    try:
        wav_file = getTTS(input_text, batched, voc_model, tts_model, hp)
    except Exception as e:
        print(e)
        return "", "500 Generating TTS error"

    save_path = wav_file[1]
    wav_file = wav_file[0]

    return send_file(save_path, mimetype="audio/wav")

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="80", threaded=False, debug=True)