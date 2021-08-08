import os
import time
from glob import glob
from operator import add, sub
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    #Confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    

    #Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    

    #F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    
    return [accuracy, specificity, sensitivity,precision, jaccard_index, F1_score]

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            score = calculate_metrics(y, y_pred)
            metrics_score = list(map(add, metrics_score, score))
        epoch_loss = epoch_loss/len(loader)
        metrics_score = [sc / len(loader) for sc in metrics_score]
    return epoch_loss, metrics_score

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("/content/drive/MyDrive/Program/files")

    """ Load dataset """
    train_x = sorted(glob("/content/drive/MyDrive/Program/new_data/train/image/*"))
    train_y = sorted(glob("/content/drive/MyDrive/Program/new_data/train/mask/*"))

    valid_x = sorted(glob("/content/drive/MyDrive/Program/new_data/test/image/*"))
    valid_y = sorted(glob("/content/drive/MyDrive/Program/new_data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "/content/drive/MyDrive/Program/files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, metrics_score = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\tVal. Loss: {valid_loss:.3f}'
        print(data_str)

        accuracy = metrics_score[0]
        specificity = metrics_score[1]
        sensitivity = metrics_score[2]
        precision = metrics_score[3]
        jaccard = metrics_score[4]
        f1 = metrics_score
        print(f"\tAccuracy: {accuracy:1.4f} \n\tSpecificity: {specificity:1.4f} \n\tSensitivity: {sensitivity:1.4f} \n\tPrecision: {precision:1.4f} \n\tJaccard: {jaccard:1.4f}\n\tF1: {f1:1.4f}\n")
#accuracy, specificity, sensitivity,precision, jaccard_index, F1_score
