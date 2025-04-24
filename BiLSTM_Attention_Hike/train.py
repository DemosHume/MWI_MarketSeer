# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(x_data, y_data, input_size, n_future, device, model=None,epoch=100,lr=0.001):
    from data import PriceDataset
    from model import DualHeadBiLSTM

    dataset = PriceDataset(x_data, y_data, device=device)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    if model is None:
        model = DualHeadBiLSTM(input_size, output_size=n_future).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in tqdm(range(epoch)):
        model.train()
        total_loss = 0
        for x, y, cls in loader:
            reg_out, cls_out = model(x)
            loss = mse_loss(reg_out, y) + 0.5 * bce_loss(cls_out, cls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'model.pth')
            print(f"Model saved at epoch {epoch+1}")
    torch.save(model.state_dict(), 'model.pth')
    return model

