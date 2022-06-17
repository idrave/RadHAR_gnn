# RadHAR dataset classification with GNN+LSTM

## Code structure

The RadHAR data must be copied into a `data` directory with the following structure:

```
.
├─  requirements.txt
├── data
│   ├── Test
│   │   ├── boxing
│   │   ├── jack
│   │   ├── jump
│   │   ├── squats
│   │   └── walk
│   └── Train
│       ├── boxing
│       ├── jack
│       ├── jump
│       ├── squats
│       └── walk
└── src
    ├── data.py
    ├── PointGNN.py
    ├── README.md
    └── train.py
```
## Setup

The required packages can be installed with:

    pip install r requirements.txt

## Running the code

To train the model with default parameters use command:
### For Point-GNN
    python src/train.py --model pointgnn
### For MM-Point-GNN
    python src/train.py --model pointgnn

Several options are available for training and can be displayed with:

    python src/train.py -h