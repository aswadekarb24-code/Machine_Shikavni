import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import optim

from src.config import BATCH_SIZE, CHECKPOINT, RESULTS
from src.models.cnn import TrialCNN
from src.models.helper import get_criterion
from src.preproc.factory import getloader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')

def train_1(model, loader, optimizer, criterion,scaler):
    model.train()
    cumloss, ok, tot =0.0,0.0,0.0
    nums_y_, nums_y = [], []
    for x,y in loader:
        x,y = x.to(DEVICE,non_blocking=True), y.to(DEVICE,non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda',enabled=DEVICE.type=='cuda'):
            out = model.forward(x)
            loss = criterion(out,y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cumloss += loss.item()*x.size(0)
        y_ = out.argmax(dim=1)
        ok += (y_ == y).sum().item()
        tot += y.size(0)

        nums_y_.extend(y_.cpu().numpy())
        nums_y.extend(y.cpu().numpy())
    macf1 = f1_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    prec = precision_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    rec = recall_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    return cumloss/tot, ok/tot,macf1,prec,rec, nums_y_, nums_y

def evaluate(model, loader, criterion):
    cumloss, nums_y_, nums_y = 0.0,[],[]

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE,non_blocking=True), y.to(DEVICE,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=DEVICE.type=='cuda'):
                out = model.forward(x)
                loss = criterion(out,y)
    
            cumloss += loss.item()*x.size(0)
            y_ = out.argmax(dim=1)
            nums_y_.extend(y_.cpu().numpy())
            nums_y.extend(y.cpu().numpy())

    tot = len(nums_y_)
    acc = accuracy_score(y_pred=nums_y_,y_true=nums_y)
    macf1 = f1_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    prec = precision_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    rec = recall_score(y_pred=nums_y_,y_true=nums_y,average='macro',zero_division=0)
    return cumloss/tot,acc,macf1,prec,rec, nums_y_, nums_y

def train(datanm, epochs = 50, useprob=False, cprob =0.5, batch_size =BATCH_SIZE, in_channels=3):
    print(f"Training : {datanm} Dataset, noise = {useprob}")
    trainld, testld, nclasses = getloader(dataset=datanm, batch_size=BATCH_SIZE, useprob=useprob,cprob=cprob)

    model = TrialCNN(in_channels = in_channels, n_classes =nclasses).to(DEVICE)
    criterion = get_criterion(datanm)
    optimizer = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled = (DEVICE.type =='cuda'))

    suffix = 'mixed' if useprob else 'clean'
    modelpath = os.path.join(CHECKPOINT, f'{datanm.lower()}_{suffix}.pth')
    bestf1 = 0.0
    history =[]
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
        _, _, tstf1, _, _, _, _= evaluate(model, testld, criterion)
        bestf1 =tstf1

    os.makedirs(os.path.dirname(modelpath),exist_ok=True)
    for epoch in range(1,epochs+1):
        trloss, tracc, trf1, trprec, trrec, _, _ = train_1(model,trainld, optimizer,criterion,scaler)
        tstloss, tstacc, tstf1,tstprec,tstrec, _ , _ = evaluate(model,testld,criterion)
        scheduler.step()
        history.append({
            'train loss' : trloss, 'train accuracy' : tracc,
            'train f1' : trf1, 'train precision' : trprec,
            'train recall' : trrec, 'test loss' : tstloss,
            'test accuracy' : tstacc, 'test f1' : tstf1,
            'test precision' : tstprec, 'test recall' : tstrec,
        })

        # if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch: {epoch:03d}/{epochs:03d}")
        print(f"Train loss : {trloss} | Train acc : {tracc} | Train f1 : {trf1}")
        print(f"Test loss : {tstloss} | Test acc : {tstacc} | Test f1 : {tstf1}")

        if tstf1 > bestf1:
            bestf1 = tstf1
            torch.save(model.state_dict(), modelpath)
            print(f"New best checkpoint: {modelpath} | F1 = {bestf1}")

    historyfile = os.path.join(RESULTS,f"history_{datanm.lower()}_{suffix}.csv")
    os.makedirs(os.path.dirname(historyfile),exist_ok=True)
    pd.DataFrame(history).to_csv(historyfile,index=False)


if __name__ == '__main__':

    # for ds in ['GTSRB','CTSD','BTSD']:
    #     train(datanm=ds,epochs=50)

    for ds in ['GTSRB','CTSD','BTSD']:
        train(datanm=ds,epochs=200,useprob=True, cprob=0.9,batch_size=256)
    
        

        
            