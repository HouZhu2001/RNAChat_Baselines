import os
import sys
from rnachat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random
import pandas as pd
questions = ["Tell me about this RNA", 
                "What is the functionality of this RNA", 
                "Briefly summarize the functionality of this RNA",
                "Please provide a detailed description of the RNA"]

class RNADataset(BaseDataset):
    def __init__(self, seq_path, split):

        df = pd.read_csv("rna_summary_2d.csv")
        if split == 'train':
            self.names = df['name'].values.tolist()[:4200]
            self.sequence = df['Sequence'].values.tolist()[:4200]
            self.labels = df['summary_no_citation'].values.tolist()[:4200]
        else:
            self.names = df['name'].values.tolist()[4200:]
            self.sequence = df['Sequence'].values.tolist()[4200:]
            self.labels = df['summary_no_citation'].values.tolist()[4200:]
            

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):

        seq = self.sequence[index]
        name = self.names[index]
        prompt =  f"###Human: <RNA><RNAHere></RNA> {random.choice(questions)} named {name}. ###Assistant:"
        if len(seq) > 1000:
            seq = seq[:1000]
        return {
            "seq": seq,
            "text_input": self.labels[index],
            "prompt": prompt
        }




