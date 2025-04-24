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
    def __init__(self, input_size, hidden_size=256, output_size=2, num_layers=3, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.attn = Attention(hidden_size)

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size * input_size)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, H*2]
        lstm_out = self.norm(lstm_out)
        context = self.attn(lstm_out)  # [B, H*2]
        context = self.dropout(context)


        reg_out = self.reg_head(context)  # [B, output_size * input_size]
        cls_out = self.cls_head(context)  # [B, input_size]
        return reg_out, cls_out
