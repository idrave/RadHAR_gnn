import random

import lmdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pickle
from pathlib import Path
import os

INPUT_FEATURES = 8


def parse_file_new(path):
    """
    Parses an input raw data text file
    :returns : 
        data: list of point cloud sequences
        min_points, max_points: minimum and maximum # of points in a frame in this file
    """
    with open(path) as f:
        lines = f.readlines()

    point_info_len = 16
    max_points = 0
    min_points = 1e10
    sequence = []
    frame = []

    for i in range(0, len(lines)-point_info_len+1, point_info_len):
        point_id    = int(lines[i+6].split()[1])
        x           = float(lines[i+7].split()[1])
        y           = float(lines[i+8].split()[1])
        z           = float(lines[i+9].split()[1])
        rang        = float(lines[i+10].split()[1])
        velocity    = float(lines[i+11].split()[1])
        doppler     = int(lines[i+12].split()[1])
        bearing     = float(lines[i+13].split()[1])
        intensity   = float(lines[i+14].split()[1])
        if point_id == 0 and i != 0:
            # append previous frame
            sequence.append(frame)
            max_points = max(max_points, len(frame))
            min_points = min(min_points, len(frame))
            frame = []
        frame.append([x, y, z, rang, velocity, doppler, bearing, intensity])
    sequence.append(frame) # append last frame
    max_points = max(max_points, len(frame))
    min_points = min(min_points, len(frame))   
    # Generate sequences of certain window length
    data=[]
    window=60 # of frames in sequence
    sliding=10
    frame_id=0
    while frame_id+window<len(sequence):
        data.append(sequence[frame_id:frame_id+window])
        frame_id+=sliding
    print(len(sequence), min_points, max_points)
    return data, (min_points, max_points)

class LMDBDataset(Dataset):
    """Create and access LMDB data."""

    def __init__(self, data_path, lmdb_path, seq_len, transform=None, replace=True, padding='zero'):
        """
        Initializer.
        :param data_path: Path to raw text files folder.
        :param lmdb_path: Path to the LMDB dataset.
        :param seq_len: Frame length of point cloud sequences in data
        :param transform: Pytorch transforms to be applied to each sample
        :param replace: Replace existing LMDB at lmdb_path
        :param padding: 'zero' or 'repeat', method to pad point clouds with less than 42 points
            'zero': pad with zeros
            'repeat': pad by repeating existing points
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
            # Read number of data sequences stored (dataset length)
            with self.env.begin(write=False) as txn:
                self.data_len = int.from_bytes(txn.get('data_len'.encode()), 'big')
            self.env.close()
            self.env = None

    def create_lmdb(self, data_dir):
        """
        Create LMDB dataset from raw data
        """
        db = lmdb.open(self.lmdb_path, map_size=int(1e12))
        data_dir = Path(data_dir)
        index = 0
        with db.begin(write=True) as txn:
            for subdir in data_dir.iterdir():
                if not subdir.is_dir(): continue
                # Parse subdirectory
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
        """
        Adds data sequences to LMDB with their corresponding ground truth labels
        """
        for sequence in data:
            frame_sz = []
            sequence_new = []
            for frame in sequence:
                assert len(frame) <= self.n_points, 'Frame with more points than %d' % self.n_points
                frame_sz.append(len(frame))
                sequence_new.append(np.concatenate([np.array(frame), np.zeros((self.n_points-len(frame),INPUT_FEATURES))])) # sequence stored with padding of zeros
            frame_sz = np.array(frame_sz)
            sequence_new = np.array(sequence_new)
            # put key value pairs in LMDB       
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
        """
        Pad point clouds with less points than n_points
        """
        if self.padding=='zero':
            return sequence # sequences are stored with padded zeros in LMDB
        if self.padding=='repeat':
            sequence_new = []
            for frame, sz in zip(sequence, frame_sz):
                # Choose remaining point indexes from existing ones and concatenate them
                pad_idx = np.random.choice(sz, self.n_points-sz)
                sequence_new.append(np.concatenate([frame[:sz], frame[pad_idx]]))
            return np.stack(sequence_new)
        raise NotImplementedError('Wrong padding type')
        
    def __getitem__(self, index):
        self.open_lmdb()
        # Define LMDB keys
        seq_key = "{}seq".format(index).encode()
        frame_sz_key = "{}frame_sz".format(index).encode()
        label_key = "{}label".format(index).encode()
        with self.env.begin(write=False) as txn:
            # Load values into np arrays
            sequence = np.frombuffer(txn.get(seq_key)).reshape((self.seq_len, self.n_points, INPUT_FEATURES))
            frame_sz = np.frombuffer(txn.get(frame_sz_key), dtype=int)
            label = int.from_bytes(txn.get(label_key),'big')
        # Pad and convert to tensors
        sequence = self.pad(sequence, frame_sz)
        sequence = torch.tensor(sequence).float()
        frame_sz = torch.tensor(frame_sz).long()
        label = torch.tensor(label).long()
        if self.transform is not None:
            sequence = self.transform(sequence)
        return sequence, label, frame_sz