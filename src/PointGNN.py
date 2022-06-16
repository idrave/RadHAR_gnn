from turtle import forward
import torch
import torch.nn as nn
import time

class ReshapeBatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        shape = x.shape
        x = x.reshape((-1, self.dim))
        out = self.bn(x)
        return out.reshape(shape)

class PointGNN(nn.Module):
    def __init__(self, T=3, r=0.05, state_dim = 8, dropout=0.1):
        super(PointGNN, self).__init__()
        self.T = T
        self.r = r
        self.state_dim = state_dim
        
        # input(batch_size, n_points, state_dim) output(batch_size, n_points, state_dim)
        self.MLP_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim,64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,3), nn.Dropout(dropout)
                ) 
            for i in range(self.T)
        ])
        # input(N, 42, 42, 6) output(N, 42, 42, 300)
        self.MLP_f = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim+3,64),
                nn.ReLU(),
                nn.Linear(64,128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,128), nn.Dropout(dropout),
                nn.ReLU()
                ) 
            for i in range(self.T)
        ])
        # input(N, 42, 300) output(N, 42, 3)
        self.MLP_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128,64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,32), nn.Dropout(dropout),
                nn.ReLU(), 
                nn.Linear(32,state_dim), nn.Dropout(dropout),
                nn.ReLU()
                )
            for i in range(self.T)
        ])

    def forward(self, state, frame_sz):
        """
        :param state: tensor of point cloud sequence (batch_size, n_points, state_dim)
        """
        batch_size, n_points, _ = state.shape
        x = state[:,:,:3]
        diff = x.unsqueeze(1) - x.unsqueeze(2) # x_i - x_j
        adj = torch.sum(diff*diff,dim=-1) < self.r
        rang = torch.arange(0,n_points).to(state.device)
        point_idx = torch.max(rang.unsqueeze(0),rang.unsqueeze(1))
        adj = torch.logical_and(adj, point_idx<frame_sz[:,None,None])

        for t in range(self.T):
            delta = self.MLP_h[t](state)
            x_js = state.unsqueeze(1).repeat((1,delta.shape[1],1,1))
            eij_input = torch.cat((delta.unsqueeze(2) - diff, x_js), dim=-1)
            eij_output = self.MLP_f[t](eij_input)
            eij_output = torch.where(adj.unsqueeze(-1), eij_output, torch.tensor(0.,device=x.device))
            pool = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](pool) + state

        return state


class HAR_PointGNN(nn.Module):
    def __init__(self,r = 0.5,output_dim = 5, T = 3, state_dim = 8, frame_num=60, dropout=0.1):
        super(HAR_PointGNN, self).__init__()
        self.pgnn = PointGNN(T=T, r=r, state_dim = state_dim, dropout=dropout)
        self.lstm_net = nn.LSTM(336, 16,num_layers=1, dropout=0, bidirectional=True)
        self.bn = nn.BatchNorm1d(frame_num*2*16)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(frame_num*2*16,output_dim)

    def forward(self,state,frame_sz):
        batch_size, seq_len, n_points, _ = state.shape
        state = state.reshape((batch_size*seq_len, n_points, -1))
        frame_sz = frame_sz.flatten()
        x = self.pgnn(state,frame_sz)
        x = x.reshape((batch_size, seq_len, -1)).permute(1,0,2)
        lstm_out,hn = self.lstm_net(x)
        lstm_out = lstm_out.permute(1,0,2).reshape(batch_size,-1)
        #lstm_out = self.bn(lstm_out)
        lstm_out = self.dropout(lstm_out)
        logits = self.dense(lstm_out)
        return logits

def test_point_gnn():
    s = torch.randn(60,42,8)

    model = PointGNN(state_dim = 8)
    y = model(s)
    y.sum().backward()
    print(model(s).size())

if __name__ == '__main__':
    test_point_gnn()