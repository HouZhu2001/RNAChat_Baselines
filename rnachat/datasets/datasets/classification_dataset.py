import os
import sys
from rnachat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random
import pandas as pd

questions = ["What is the functionality of this RNA in the process:",
             "Tell me about the role of this RNA in the process:",
             "Describe the function of this RNA in the process:"]

class ClassificationDataset(BaseDataset):
    def __init__(self, seq_path, split):

        df = pd.read_csv("rna_go.csv")# Convert to dictionary
        sequence_dict = df.set_index("rna_id")["sequence"].to_dict()

        self.rna_dict = {}
        for index, row in df.iterrows():
            go_term_dict = {col: row[col] if pd.notna(row[col]) else None for col in df.columns}
            self.rna_dict[row['rna_id']] = go_term_dict

        
        self.go_list = [(rna_id, sequence_dict[rna_id], go_term, qualifier) for rna_id, value in self.rna_dict.items() for go_term, qualifier in value.items()]
        
        if split == 'train':
            self.go_list = self.go_list[:int(len(self.go_list)*0.8)]
        else:
            self.go_list = self.go_list[int(len(self.go_list)*0.8):]

    def __len__(self):
        return len(self.go_list)

    def __getitem__(self, index):

        rna_id, seq, go_term, qualifier = self.go_list[index]
        prompt =  f"###Human: <RNA><RNAHere></RNA> {random.choice(questions)} {go_term}. ###Assistant:"
        if len(seq) > 1000:
            seq = seq[:1000]
        return {
            "seq": seq,
            "text_input": qualifier if qualifier else "no experimental evidence",
            "prompt": prompt
        }




