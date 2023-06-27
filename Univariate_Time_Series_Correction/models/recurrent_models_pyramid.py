import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)

        recurrent_features, _ = self.lstm0(input, (h_0, c_0))
        recurrent_features, _ = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element. 

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim, device=None):
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=100, num_layers=1, batch_first=True)
        # self.lstm0 = nn.LSTM(input_size=in_dim, hidden_size=64, num_layers=1, batch_first=True)
        # self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.linear = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.linear = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 100).to(self.device)
        c_0 = torch.zeros(1, batch_size, 100).to(self.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        # recurrent_features, _ = self.lstm0(input, (h_0, c_0))
        # recurrent_features, _ = self.lstm1(recurrent_features)
        # recurrent_features, _ = self.lstm2(recurrent_features)
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 100))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs, recurrent_features

class LSTMGenerator_1(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)

        recurrent_features, _ = self.lstm0(input, (h_0, c_0))
        recurrent_features, _ = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features
    
if __name__ == "__main__":
    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4

    gen = LSTMGenerator(noise_dim, seq_dim)
    dis = LSTMDiscriminator(seq_dim)
    noise = torch.randn(8, 16, noise_dim)
    gen_out = gen(noise)
    dis_out = dis(gen_out)
    
    print("Noise: ", noise.size())
    print("Generator output: ", gen_out.size())
    print("Discriminator output: ", dis_out.size())
