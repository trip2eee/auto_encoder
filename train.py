import torch
from torch import nn
from model.auto_encoder import AutoEncoder
from model.dataset import EncoderDataSet
from torch.utils.data import DataLoader
import numpy as np
import cv2

learning_rate = 1e-3
batch_size = 1
epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EncoderDataSet('./data', transform=True)
train_dataloader = DataLoader(dataset, batch_size=batch_size)

model = AutoEncoder()
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def to_image(tensor):
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, [1, 2, 0])
    image = image * 255.0
    image = np.ascontiguousarray(image, dtype=np.uint8)

    return image

def train_loop(data_loader, model, loss_fn, optimizer):

    size = len(data_loader.dataset)
    
    if optimizer is not None:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    sum_loss = 0.0

    for batch, (x, y) in enumerate(data_loader):
        x = x.to(device)        
        y = y.to(device)

        pred = model(x)
        
        loss = loss_fn(pred, y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        loss = loss.item()
        sum_loss += loss

        if batch % 1 == 0:            
            current = batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        
        # image_org = to_image(y[0])
        # cv2.imshow('origin', image_org)

        # image_in = to_image(x[0])
        # cv2.imshow('input', image_in)

        # image = to_image(pred[0])
        # cv2.imshow('recov', image)
        # cv2.waitKey(1)

    mean_loss = sum_loss / size
    return mean_loss


min_loss = 1e10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, loss_fn, optimizer)

    if loss < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), 'model_weights_min_loss.pth')        

torch.save(model.state_dict(), 'model_weights.pth')

print("Done!")

