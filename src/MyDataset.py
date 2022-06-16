import random

import lmdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pickle
from dataParser import parse_file_new
from pathlib import Path
import os

INPUT_FEATURES = 8

class LMDBDataset(Dataset):
    """Access LMDB data."""

    def __init__(self, data_path, lmdb_path, seq_len, transform=None, replace=True, padding='zero'):
        """
        Initializer.
        :param lmdb_path: Path to the LMDB dataset.
        :param transform: Pytorch transforms to be applied to each sample, can be None.
        :param padding:
        """
        super(LMDBDataset).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.n_points = 42  # number of points per frame
        self.padding = padding
        self.seq_len = seq_len
        self.action_label={
            "boxing":0,
            "jack":1,
            "jump":2,
            "squats":3,
            "walk":4
        }
        if replace or not Path(lmdb_path).is_dir():
            self.create_lmdb(data_path)
        else:
            self.open_lmdb()
            with self.env.begin(write=False) as txn:
                self.data_len = int.from_bytes(txn.get('data_len'.encode()), 'big')
            self.env.close()
            self.env = None

    #@staticmethod
    def create_lmdb(self, data_dir):
        db = lmdb.open(self.lmdb_path, map_size=int(1e12))
        data_dir = Path(data_dir)
        index = 0
        with db.begin(write=True) as txn:
            for subdir in data_dir.iterdir():
                if not subdir.is_dir(): continue
                print('parsing dir: %s'%subdir)
                label = self.action_label[subdir.stem]
                for f in subdir.iterdir():                    
                    print('parsing file: %s' % f.name)
                    data, (mi, ma) = parse_file_new(f)
                    self.add_data(data, label, txn, index)
                    index += len(data)
            txn.put('data_len'.encode(), index.to_bytes(4, 'big'))
        self.data_len = index
        db.close()
    
    def add_data(self, data, label, txn, index):
        for sequence in data:
            frame_sz = []
            sequence_new = []
            for frame in sequence:
                assert len(frame) <= self.n_points, 'Frame with more points than %d' % self.n_points
                frame_sz.append(len(frame))
                sequence_new.append(np.concatenate([np.array(frame), np.zeros((self.n_points-len(frame),INPUT_FEATURES))]))
            frame_sz = np.array(frame_sz)
            sequence_new = np.array(sequence_new)            
            seq_key = "{}seq".format(index).encode()
            frame_sz_key = "{}frame_sz".format(index).encode()
            label_key = "{}label".format(index).encode()
            txn.put(seq_key, sequence_new)
            txn.put(frame_sz_key, frame_sz)
            txn.put(label_key, label.to_bytes(4,'big'))
            index += 1

    def open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                                 readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.data_len

    def pad(self, sequence, frame_sz):
        if self.padding=='zero':
            return sequence
        if self.padding=='repeat':
            sequence_new = []
            for frame, sz in zip(sequence, frame_sz):
                pad_idx = np.random.choice(sz, self.n_points-sz)
                sequence_new.append(np.concatenate([frame[:sz], frame[pad_idx]]))
            return np.stack(sequence_new)
        raise NotImplementedError('Wrong padding type')
        
    def __getitem__(self, index):
        self.open_lmdb()
        seq_key = "{}seq".format(index).encode()
        frame_sz_key = "{}frame_sz".format(index).encode()
        label_key = "{}label".format(index).encode()
        #id_key = "id{}".format(index).encode()
        with self.env.begin(write=False) as txn:
            sequence = np.frombuffer(txn.get(seq_key)).reshape((self.seq_len, self.n_points, INPUT_FEATURES))
            frame_sz = np.frombuffer(txn.get(frame_sz_key), dtype=int)
            label = int.from_bytes(txn.get(label_key),'big')
        sequence = self.pad(sequence, frame_sz)
        sequence = torch.tensor(sequence).float()
        frame_sz = torch.tensor(frame_sz).long()
        label = torch.tensor(label).long()
        if self.transform is not None:
            sequence = self.transform(sequence)
        return sequence, label, frame_sz

if __name__=="__main__":
    # lmdbData=LmdbData('./lmdbData')
    # print(lmdbData.get(0)[0].shape)
    # print(lmdbData.len())
    # train_dataset=MyDataset('./Data/lmdbData_train')
    train_dataset = LMDBDataset('data/Test', 'data/lmdbdebug2', 60, replace=False, padding='repeat')
    print(train_dataset.__len__())
    train_dataset[0]
    train_dataset[1]
    # print(train_dataset.__getitem__(0))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )
    for i, data in enumerate(train_loader):
        inputs, labels = data

        print(inputs.size(), labels.size())
        break