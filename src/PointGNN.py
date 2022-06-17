import torch
import torch.nn as nn

class BasePointGNN(nn.Module):
    """
    Base class for point GNN models
    """
    def __init__(self, r=0.05, layers=3, state_dim = 8, dropout=0.1, mask=False):
        """
        :param r: Radius in which to consider points as adjacent
        :param layers: number of layers of GNN
        :param state_dim: number of dimensions of input point state
        :param dropout: percentage of dropout
        :param mask: Whether to remove adjacency to padding in input
        """
        super().__init__()
        self.radius = r
        self.layers = layers
        self.state_dim = state_dim
        self.mask = mask
        
        # input(batch_size, n_points, state_dim) output(batch_size, n_points, state_dim)
        self.MLP_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,128), nn.LayerNorm(128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,3)
                ) 
            for i in range(self.layers)
        ])
        # input(batch_size, n_points, n_points, state_dim+3) output(batch_size, n_points, n_points, 128)
        self.MLP_f = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim+3,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,128), nn.LayerNorm(128), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU()
                ) 
            for i in range(self.layers)
        ])
        # input(batch_size, n_points, 128) output(batch_size, n_points, state_dim)
        self.MLP_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128,64), nn.LayerNorm(64), nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64,32), nn.LayerNorm(32), nn.Dropout(dropout),
                nn.ReLU(), 
                nn.Linear(32,state_dim), nn.Dropout(dropout),
                nn.ReLU()
                )
            for i in range(self.layers)
        ])

    def forward(self, state, frame_sz):
        raise NotImplementedError()

class PointGNN(BasePointGNN):
    """
    Point-GNN model from:
        'Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud'
    """
    def __init__(self, layers=3, r=0.05, state_dim = 8, dropout=0.1, mask=False):
        super().__init__(r=r, layers=layers, state_dim=state_dim, dropout=dropout, mask=mask)

    def forward(self, state, frame_sz):
        """
        :param state: tensor of point cloud sequence (batch_size, n_points, state_dim)
        :param frame_sz: number of points in each point cloud before padding
        """
        batch_size, n_points, _ = state.shape
        x = state[:,:,:3] # point coordinates
        diff = x.unsqueeze(2) - x.unsqueeze(1) # x_i - x_j
        adj = torch.sum(diff*diff,dim=-1) < self.radius # get adjacency matrix
        
        if self.mask:
            # remove edges to points that are part of padding
            rang = torch.arange(0,n_points).to(state.device)
            mask = torch.max(rang.unsqueeze(0),rang.unsqueeze(1)) < frame_sz[:,None,None]
            adj = torch.logical_and(adj, mask)

        for t in range(self.layers):
            delta = self.MLP_h[t](state)
            s_js = state.unsqueeze(1).repeat((1,delta.shape[1],1,1))
            eij_input = torch.cat((delta.unsqueeze(2) - diff, s_js), dim=-1) # [x_j - x_i + delta x, s_j]
            eij_output = self.MLP_f[t](eij_input)
            # Remove entries for non-adjacent nodes
            eij_output = torch.where(adj.unsqueeze(-1), eij_output, torch.tensor(0.,device=x.device))
            pool = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](pool) + state

        return state

class ModifiedHardsigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_sigmoid = nn.Hardsigmoid()
    def forward(self, x):
        # torch hard sigmoid is increasing in domain [-3,3], modify to [-1,1]
        return self.hard_sigmoid(x*3.0) 

class MMPointGNN(BasePointGNN):
    """
    MM-Point-GNN model from
        'MMPoint-GNN: Graph Neural Network with Dynamic Edges for Human Activity Recognition
        through a Millimeter-wave Radar'
    """
    def __init__(self, r=0.05, layers=3, state_dim = 8, dropout=0.1, mask=False):
        super().__init__(r=r, layers=layers, state_dim=state_dim, dropout=dropout, mask=mask)
        # input(batch_size,n_points,n_points,128) output(batch_size,n_points,n_points,1)
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
            for i in range(self.layers)
        ])
        self.mhs = ModifiedHardsigmoid()

    def forward(self, state, frame_sz):
        """
        :param state: tensor of point cloud sequence (batch_size, n_points, state_dim)
        :param frame_sz: number of points in each point cloud before padding
        """
        batch_size, n_points, _ = state.shape
        x = state[:,:,:3]  # point coordinates
        diff = x.unsqueeze(2) - x.unsqueeze(1) # x_i - x_j
        adj = torch.sum(diff*diff,dim=-1) < self.radius # get adjacency matrix
        if self.mask:
            # remove edges to points that are part of padding
            rang = torch.arange(0,n_points).to(state.device)
            mask = torch.max(rang.unsqueeze(0),rang.unsqueeze(1)) < frame_sz[:,None,None]
            adj = torch.logical_and(adj, mask)

        for t in range(self.layers):
            delta = self.MLP_h[t](state)
            x_js = state.unsqueeze(1).repeat((1,delta.shape[1],1,1))
            eij_input = torch.cat((delta.unsqueeze(2) - diff, x_js), dim=-1) # [x_j - x_i + delta x, s_j]
            eij_output = self.MLP_f[t](eij_input)
            delta_a = self.MLP_r[t](eij_output).squeeze(-1)
            if self.training:
                adj = self.mhs(adj + delta_a)
            else:
                adj = adj + delta_a > 0.0
            if self.mask:
                adj = adj * mask
            eij_output = adj.unsqueeze(-1) * eij_output # apply adjacency matrix
            pool = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](pool) + state

        return state

class HAR_PointGNN(nn.Module):
    """
    LSTM + GNN model
    """
    def __init__(self, gnn, output_dim = 5, frame_num=60, dropout=0.1):
        """
        :param gnn: Base GNN model for point cloud
        :param output_dim: number of classes
        :param frame_num: number of frames in input sequence
        :param dropout: percentage of dropout
        """
        super(HAR_PointGNN, self).__init__()
        self.pgnn = gnn
        self.lstm_net = nn.LSTM(42*8, 16,num_layers=1, dropout=0, bidirectional=True)
        self.ln = nn.LayerNorm(2*16)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(frame_num*2*16,output_dim)

    def forward(self,state,frame_sz):
        batch_size, seq_len, n_points, _ = state.shape
        state = state.reshape((batch_size*seq_len, n_points, -1))
        frame_sz = frame_sz.flatten()
        x = self.pgnn(state,frame_sz)
        x = x.reshape((batch_size, seq_len, -1)).permute(1,0,2) # (seq_len, batch_size, 42*8)
        lstm_out,hn = self.lstm_net(x)
        lstm_out = lstm_out.permute(1,0,2) # (batch_size, seq_len, 2*16)
        lstm_out = self.ln(lstm_out).reshape(batch_size,-1)
        lstm_out = self.dropout(lstm_out)
        logits = self.dense(lstm_out)
        return logits

def test_point_gnn():
    s = torch.randn(60,42,8)

    model = PointGNN(state_dim = 8)
    y = model(s)
    print(y.size())

if __name__ == '__main__':
    test_point_gnn()