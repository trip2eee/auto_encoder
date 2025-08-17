import torch
from torch import nn
from model.auto_encoder import AutoEncoder
from model.dataset import EncoderDataSet
from torch.utils.data import DataLoader
import numpy as np
import cv2

learning_rate = 1e-3
batch_size = 1
epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EncoderDataSet('./data', transform=True)
train_dataloader = DataLoader(dataset, batch_size=batch_size)

model = AutoEncoder()
model.to(device)
model.load_state_dict(torch.load('model_weights.pth'))

loss_fn = nn.MSELoss()

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

    for batch, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        
        loss = loss_fn(pred, y)
        
        image_org = to_image(y[0])
        cv2.imshow('origin', image_org)

        image_in = to_image(x[0])
        cv2.imshow('input', image_in)

        image = to_image(pred[0])
        cv2.imshow('recov', image)
        cv2.waitKey(0)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if batch % 1 == 0:
            loss = loss.item()
            current = batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, None)

torch.save(model.state_dict(), 'model_weights.pth')

print("Done!")

