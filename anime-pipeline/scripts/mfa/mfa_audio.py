
import os, soundfile, soxr
from pathlib import Path    
from glob import glob

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset  

from utils.file import read_metadata
from utils.data import rescale_temporal_data, get_temporal_torch_data

from audio.utils.audio import load_audio, audio_volume_normalize
from audio.utils.augment import random_aug



class AudioDataset(Dataset):

    def __init__(self, data_dir, sr=None, volume_norm=True, layer_norm=True, aug_config=None):
        super().__init__() 

        self.sr = sr
        self.volume_norm = volume_norm
        self.layer_norm = layer_norm
        self.aug_config = aug_config

        self.metadata = []

        if type(data_dir) is list:
            # metas directly, usually generated online meta to avoid file IO
            for meta in data_dir:
                file = meta["file"]
                if not os.path.isfile(file) or not file.endswith(".wav"):
                    continue
                self.metadata.append({ 
                        "index": meta["index"] if "index" in meta else os.path.basename(file)[:-4],
                        "file": file, #os.path.dirname(f),
                    })
        elif os.path.isfile(f"{data_dir}/total.meta"):
            metas, _ = read_metadata( f"{data_dir}/total.meta" ) 
            for meta in metas:
                file = f"{data_dir}/wavs/{meta['index']}.wav"
                if not os.path.isfile(file):
                    continue
                self.metadata.append({ 
                        "index": meta["index"],
                        "file": file,
                    })
        else:
            if os.path.isdir(f"{data_dir}/wavs"):
                files = glob(f"{data_dir}/wavs/*.wav")
            else:
                files = glob(f"{data_dir}/*.wav")
            for f in files:
                self.metadata.append({ 
                        "index": os.path.basename(f)[:-4],
                        "file": f, #os.path.dirname(f),
                    })
        print(f"total {len(self.metadata)} audio files in dataset.")


    def __len__(self):
        return len(self.metadata) 

    def __getitem__(self, idx):
        meta = self.metadata[idx] 

        audio, sr = load_audio(meta["file"], self.sr)
        if self.aug_config is not None:
            audio = random_aug(audio, sr, self.aug_config, False)
        if self.volume_norm:
            audio = audio_volume_normalize(audio, coff=0.2)    
        audio = torch.from_numpy(audio).float()
        if self.layer_norm:
            audio = F.layer_norm(audio, audio.shape)

        duration = torch.tensor(audio.shape[0]).long()

        sample = {
            "index": meta["index"], 
            "file": meta["file"],
            "wav": audio,
            "duration": duration,
            }
        return sample 


    def collate_fn(self, batch):

        collate_batch = {}

        for k in ["index", "file"]: #, "data_dir"
            collate_batch[k] = [ b[k] for b in batch ] 

        for k in ["duration"]: #(B, 1)
            v = [ b[k] for b in batch ] 
            collate_batch[k] = torch.stack(v, dim=0)
        
        for k in ["wav"]: #(B, T)
            v = [ b[k] for b in batch ] 
            collate_batch[k] = nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=0)

        return collate_batch



class HubertDataset(Dataset):
    def __init__(self, data_dir, hubert_type, resample_scale=0.6):
        super().__init__() 

        self.data_dir = data_dir 
        self.hubert_type = hubert_type
        self.resample_scale = resample_scale

        if os.path.isfile(f"{data_dir}/total.meta"):
            self.metadata, _ = read_metadata( f"{data_dir}/total.meta" ) 
            self.meta_filter()
        else:
            if os.path.isdir(f"{data_dir}/wavs_{self.hubert_type}"):
                files = glob(f"{data_dir}/wavs_{self.hubert_type}/*.pt")
            else:
                files = glob(f"{data_dir}/*.pt")
            
            self.metadata = []
            for f in files:
                meta = { 
                    "index": os.path.basename(f)[:-3],
                    "data_dir": data_dir,
                    "path": os.path.dirname(f),
                    }
                self.metadata.append(meta)
        print(f"total {len(self.metadata)} files of HubertDataset.")


    def meta_filter(self):
        metadata= []
        for meta in self.metadata:
            index = meta["index"]
            file = f"{self.data_dir}/wavs_{self.hubert_type}/{index}.pt"
            if not os.path.isfile(file):
                continue
            meta["path"] = os.path.dirname(file)
            meta["data_dir"] = self.data_dir
            metadata.append(meta)
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata) 

    def __getitem__(self, idx):
        meta = self.metadata[idx] 
        index = meta["index"]
        path = meta["path"]

        sample = {}

        file = f"{path}/{index}.pt"
        hubert = get_temporal_torch_data(file, rescale=True).float()

        #hu_scale = 30.0 / 50.0 #anim is 30fps, hubert is 50fps
        frame_num = int(hubert.shape[0] * self.resample_scale + 0.5)
        sample["frame_num"] = torch.tensor(frame_num).long()
        sample["hubert"] = rescale_temporal_data(hubert, frame_num) #(T,1024)
        
        sample["index"] = index
        sample["path"] = path
        sample["data_dir"] = meta["data_dir"]
        return sample


    def collate_fn(self, batch):

        collate_batch = {}

        for k in ["index", "data_dir", "path"]:
            collate_batch[k] = [ b[k] for b in batch ] 

        for k in ["frame_num"]: #(B, 1)
            v = [ b[k] for b in batch ] 
            collate_batch[k] = torch.stack(v, dim=0)
        
        for k in ["hubert"]: #(B, 1024, T)
            v = [ b[k] for b in batch ] 
            v = nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=0)
            collate_batch[k] = v.transpose(1, 2) # B x C x T

        return collate_batch


