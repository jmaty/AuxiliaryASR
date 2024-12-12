#coding: utf-8

import logging
# import os
import os.path as osp
import random
# import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
# from g2p_en import G2p
# from torch import nn
from torch.utils.data import DataLoader

from text_utils import TextCleaner, add_spaces_around_punctuation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
# SPECT_PARAMS = {
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dict_path=DEFAULT_DICT_PATH,
                 max_text_len=256,
                ):

        # spect_params = SPECT_PARAMS
        # mel_params = MEL_PARAMS

        # _data_list = [l[:-1].split('|') for l in data_list]
        # Remove long utterances
        _data_list = [parts for l in data_list if len((parts := l[:-1].split('|'))[1]) <= max_text_len]
        print(f"Reducing input utterances from {len(data_list)} to {len(_data_list)}, ignoring utterances with >{max_text_len} phonemes")
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]

        self.text_cleaner = TextCleaner(dict_path)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
        self.mean, self.std = -4, 4


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wave, text_tensor, _ = self._load_tensor(data)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_melspec(wave_tensor)

        if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                mode='linear').squeeze(0)

        acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean)/self.std

        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, text_tensor, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, _ = sf.read(wave_path)

        # # phonemize the text
        # ps = self.g2p(text.replace('-', ' '))
        # if "'" in ps:
        #     ps.remove("'")

        # Assume phonetic sentence at the input => no need to phonemize
        # Surround sentence-internal punctuation by spaces to be compatible with
        # original g2p and PL-BERT and StyleTTS2
        text = self.text_cleaner(add_spaces_around_punctuation(text))
        # blank_index = self.text_cleaner.word_index_dictionary[" "]
        _, blank_index = self.text_cleaner.blank
        text = [blank_index] + text + [blank_index]  # Add silence (blank) at the beginning and end
        # text.insert(0, blank_index) # add a blank at the beginning (silence)
        # text.append(blank_index) # add a blank at the end (silence)

        text = torch.LongTensor(text)

        return wave, text, speaker_id


class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert text_size < (mel_size//2)

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config=None,
                     dataset_config=None):
    collate_config = collate_config or {}
    dataset_config = dataset_config or {}

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=device != 'cpu')
    return data_loader
