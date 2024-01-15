import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = file_name.split("_")[1]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = key.split("_")[1]
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    data[key]["vertice"] = np.load(vertice_path,allow_pickle=True) # memory tweak

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'train':range(*[int(i) for i in args.train_splits.split(" ")]),
              'val':range(*[int(i) for i in args.val_splits.split(" ")]),
              'test':range(*[int(i) for i in args.test_splits.split(" ")])}
    
    def segmented_append(data_list, orig_v, seconds=15):
        audio_ticks = orig_v["audio"].shape[0]
        for i in range(math.ceil(audio_ticks / (16000 * seconds))):
            new_v = defaultdict(dict)
            new_v["name"] = orig_v["name"] + "-" + str(i)
            new_v["template"] = orig_v["template"]
            if (i+1) * 16000 * seconds <= audio_ticks:
                new_v["audio"] = orig_v["audio"][i * 16000 * seconds : (i+1) * 16000 * seconds]
                new_v["vertice"] = orig_v["vertice"][i * 30 * seconds : (i+1) * 30 * seconds]
            else:
                new_v["audio"] = orig_v["audio"][i * 16000 * seconds :]
                new_v["vertice"] = orig_v["vertice"][i * 30 * seconds :]
            data_list.append(new_v)
   
    for k, v in data.items():
        subject_id = k.split("_")[1]
        sentence_id = int(k.split(".")[0][-3:])
        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            segmented_append(train_data, v, seconds=args.segment_append_seconds)
        if subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            segmented_append(valid_data, v, seconds=args.segment_append_seconds)
        if subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            segmented_append(test_data, v, seconds=args.segment_append_seconds) 

    print("Training: " + str(len(train_data)), "Validation: " + str(len(valid_data)), "Test: " + str(len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    