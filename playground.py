import torch
from dataset import LJSpeechDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

metadata_path="./LJSPEECH/train.txt"
mel_dir="./LJSPEECH/mel"
duration_dir="./LJSPEECH/duration"

def create_phonemes_dict():
   phonemes=set()
   with open(metadata_path,'r',encoding='utf-8') as file:
        for line in file:
            _phonemes_=line.strip().split("|")[2].strip('{}')
            for phoneme in _phonemes_.split():
                phonemes.add(phoneme)
   arr=sorted(phonemes)
   phoneme_map={ val:key for key,val in enumerate(arr)}
   phoneme_map["PAD"] = len(arr)
   phoneme_map["UNK"] = len(arr)+1
   return phoneme_map


#----------BATCH-------------
from torch.nn.utils.rnn import pad_sequence

phoneme_map=create_phonemes_dict()   



def collatefn(batch):
    mels = []
    durations = []
    phonemes = []
    texts = []
    for item in batch:
        mels.append(item['mel'])
        durations.append(item['duration'])
        phonemes.append(item['phonemes'])
        texts.append(item['text'])
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    durations_padded = pad_sequence(durations, batch_first=True, padding_value=0)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=phoneme_map["PAD"])
    return {
        'mel': mels_padded,
        'duration': durations_padded,
        'phonemes': phonemes_padded,
        'text': texts
    }

if __name__ == "__main__":
    
    dataset=LJSpeechDataset(metadata_path=metadata_path,mel_dir=mel_dir,duration_dir=duration_dir,phoneme_dict=phoneme_map)
    dataloader=DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collatefn
    )

    for batch in dataloader:
      print("Mel Shape:", batch['mel'].shape)
      print("Duration Shape:", batch['duration'].shape)
      print("Phonemes:", batch['phonemes'].shape)
      print("Text:", batch['text'])
      break

