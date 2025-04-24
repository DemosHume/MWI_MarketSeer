# train.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from model import DualHeadBiLSTM
from data import PriceDataset

def train_model(x_data, y_data, input_size, n_future, device, model=None):
    from data import PriceDataset
    from model import DualHeadBiLSTM

    dataset = PriceDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    if model is None:
        model = DualHeadBiLSTM(input_size, output_size=n_future).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in tqdm(range(100)):
        model.train()
        total_loss = 0
        for x, y, cls in loader:
            x, y, cls = x.to(device), y.to(device), cls.to(device)
            reg_out, cls_out = model(x)
            loss = mse_loss(reg_out, y) + 0.5 * bce_loss(cls_out, cls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), 'model.pth')
    return model

