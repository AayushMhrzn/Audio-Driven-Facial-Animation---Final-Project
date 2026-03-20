import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class CNN_TCN_Seq(nn.Module):
    def __init__(self, mfcc_dim=13, lip_points=20, out_len=5, cnn_out_dim=64, tcn_channels=[128,128,128,128]):
        super().__init__()
        self.out_len = out_len
        self.out_dim_per_frame = lip_points * 2
        self.cnn = nn.Sequential(
            nn.Conv1d(mfcc_dim, cnn_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
            nn.Conv1d(cnn_out_dim, cnn_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU()
        )
        self.tcn = TCN(input_size=cnn_out_dim, num_channels=tcn_channels, kernel_size=3, dropout=0.2)

        # decoder: map TCN features (batch, channels, seq_len) -> per-frame outputs
        # we'll use a conv1d to produce out_len * out_dim_per_frame features across time, then pool
        self.decoder = nn.Sequential(
            nn.Conv1d(tcn_channels[-1], tcn_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], 256),
            nn.ReLU(),
            nn.Linear(256, out_len * self.out_dim_per_frame)  # final flattened seq output
        )

    def forward(self, x):
        # x: (batch, seq_len, mfcc_dim)
        x = x.transpose(1,2)                # -> (batch, mfcc_dim, seq_len)
        x = self.cnn(x)                     # -> (batch, cnn_out_dim, seq_len)
        x = self.tcn(x)                     # -> (batch, tcn_out, seq_len)
        out = self.decoder(x)               # -> (batch, out_len * out_dim_per_frame)
        out = out.view(-1, self.out_len, self.out_dim_per_frame)  # -> (batch, out_len, out_dim)
        return out
