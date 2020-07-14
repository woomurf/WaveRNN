import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
import zipfile, os
import uuid


def getTTS(input_text, batched, voc_model, tts_model, hp):

    if input_text:
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else:
        with open('sentences.txt') as f:
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    r = tts_model.r

    simple_table([('WaveRNN', str(voc_k) + 'k'),
                  (f'Tacotron(r={r})', str(tts_k) + 'k'),
                  ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', 11_000 if batched else 'N/A'),
                  ('Overlap Samples', 550 if batched else 'N/A')])

    wav_list = []

    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)

        save_path = './sound/' + str(uuid.uuid4()) + '.wav'

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8

        wav_file = voc_model.generate(m, save_path, batched, 3000, 550, hp.mu_law)
        
        wav_list.append(wav_file)
        wav_list.append(save_path)

    print('\n\nDone.\n')

    return wav_list
