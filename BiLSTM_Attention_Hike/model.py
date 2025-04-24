# model.py
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output).squeeze(-1), dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context

class DualHeadBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=2, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = Attention(hidden_size)
        self.reg_head = nn.Linear(hidden_size * 2, output_size * input_size)
        self.cls_head = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attn(lstm_out)
        reg_out = self.reg_head(context)
        cls_out = torch.sigmoid(self.cls_head(context))
        return reg_out, cls_out
