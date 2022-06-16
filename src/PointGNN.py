from turtle import forward
import torch
import torch.nn as nn
import time

class BasePointGNN(nn.Module):
    def __init__(self, r=0.05, T=3, state_dim = 8, dropout=0.1):
        super().__init__()
        self.r = r
        self.T = T
        self.state_dim = state_dim
        
        # input(batch_size, n_points, state_dim) output(batch_size, n_points, state_dim)
        self.MLP_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,128), nn.LayerNorm(128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,3)
                ) 
            for i in range(self.T)
        ])
        # input(N, 42, 42, 6) output(N, 42, 42, 300)
        self.MLP_f = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim+3,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,128), nn.LayerNorm(128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU()
                ) 
            for i in range(self.T)
        ])
        # input(N, 42, 300) output(N, 42, 3)
        self.MLP_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,32), nn.LayerNorm(32), nn.Dropout(dropout),
                nn.ReLU(), 
                nn.Linear(32,state_dim), nn.Dropout(dropout),
                nn.ReLU()
                )
            for i in range(self.T)
        ])

    def forward(self, state, frame_sz):
        raise NotImplementedError()

class PointGNN(BasePointGNN):
    def __init__(self, T=3, r=0.05, state_dim = 8, dropout=0.1):
        super().__init__(r=r, T=T, state_dim=state_dim, dropout=dropout)

    def forward(self, state, frame_sz):
        """
        :param state: tensor of point cloud sequence (batch_size, n_points, state_dim)
        """
        batch_size, n_points, _ = state.shape
        x = state[:,:,:3]
        diff = x.unsqueeze(1) - x.unsqueeze(2) # x_i - x_j
        adj = torch.sum(diff*diff,dim=-1) < self.r
        rang = torch.arange(0,n_points).to(state.device)
        mask = torch.max(rang.unsqueeze(0),rang.unsqueeze(1)) < frame_sz[:,None,None]
        adj = torch.logical_and(adj, mask)

        for t in range(self.T):
            delta = self.MLP_h[t](state)
            x_js = state.unsqueeze(1).repeat((1,delta.shape[1],1,1))
            eij_input = torch.cat((delta.unsqueeze(2) - diff, x_js), dim=-1)
            eij_output = self.MLP_f[t](eij_input)
            eij_output = torch.where(adj.unsqueeze(-1), eij_output, torch.tensor(0.,device=x.device))
            pool = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](pool) + state

        return state

class MMPointGNN(BasePointGNN):
    def __init__(self, r=0.05, T=3, state_dim = 8, dropout=0.1):
        super().__init__(r=r, T=T, state_dim=state_dim, dropout=dropout)
        # input(N,42,42,128) output(N, 42, 42, )
        self.MLP_r = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                # nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1)
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
        mask = torch.max(rang.unsqueeze(0),rang.unsqueeze(1)) < frame_sz[:,None,None]
        adj = torch.logical_and(adj, mask)

        for t in range(self.T):
            delta = self.MLP_h[t](state)
            x_js = state.unsqueeze(1).repeat((1,delta.shape[1],1,1))
            eij_input = torch.cat((delta.unsqueeze(2) - diff, x_js), dim=-1)
            eij_output = self.MLP_f[t](eij_input)
            delta_a = self.MLP_r[t](eij_output).squeeze(-1)
            if self.training:
                adj = nn.Hardsigmoid()(adj + delta_a)
            else:
                adj = adj + delta_a > 0.0
            adj = adj * mask
            eij_output = adj.unsqueeze(-1) * eij_output
            pool = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](pool) + state

        return state

class HAR_PointGNN(nn.Module):
    def __init__(self, gnn, output_dim = 5, frame_num=60, dropout=0.1):
        super(HAR_PointGNN, self).__init__()
        self.pgnn = gnn
        self.lstm_net = nn.LSTM(336, 16,num_layers=1, dropout=0, bidirectional=True)
        self.ln = nn.LayerNorm(2*16)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(frame_num*2*16,output_dim)

    def forward(self,state,frame_sz):
        batch_size, seq_len, n_points, _ = state.shape
        state = state.reshape((batch_size*seq_len, n_points, -1))
        frame_sz = frame_sz.flatten()
        x = self.pgnn(state,frame_sz)
        x = x.reshape((batch_size, seq_len, -1)).permute(1,0,2)
        lstm_out,hn = self.lstm_net(x)
        lstm_out = lstm_out.permute(1,0,2)
        lstm_out = self.ln(lstm_out).reshape(batch_size,-1)
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