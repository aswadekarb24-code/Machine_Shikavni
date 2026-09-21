import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import PROCESSED_DIR, RAW_DIR


def procCTSD(datadir,test_size=0.2, rndstate=42):
    csvFile = os.path.join(datadir, "annotations.csv")
    imgs = os.path.join(datadir, "images")

    data = pd.read_csv(csvFile)

    valid = (data['x2'] > data['x1']) & (data['y2'] > data['y1'])
    data = data[valid].dropna().reset_index(drop=True)

    train, test = train_test_split(
        data, test_size=test_size, stratify=data['category'], random_state=rndstate
    )

    def cropsave(data, isTrain):
        type = 'train' if isTrain else 'test'
        print(f"dataset: CTSD, type:{type}")
        for i,row in tqdm(data.iterrows(),total=len(data)):
            img = os.path.join(imgs, row['file_name'])
            if not os.path.exists(img):
                continue
            sav = os.path.join(PROCESSED_DIR,"CTSD", type, str(int(row['category'])))
            os.makedirs(sav,exist_ok=True)

            with Image.open(img) as im:
                sign = (row['x1'],row['y1'],row['x2'],row['y2'])
                cim = im.crop(sign)
                path = os.path.join(sav, f"{i}_{row['file_name']}")
                cim.save(path)

    cropsave(train,True)
    cropsave(test,False)

def procGTSRB(datadir):
    def cropsave(data, isTrain):
        type= 'train' if isTrain else 'test'
        # print(data.columns)
        valid = (data['Roi.X2'] > data['Roi.X1']) & (data['Roi.Y2'] > data['Roi.Y1'])
        data = data[valid].dropna().reset_index(drop=True)
        print(f"dataset: GTSRB, type:{type}")
        for i,row in tqdm(data.iterrows(),total=len(data)):
            img = os.path.join(datadir, str(row['Path']))
            if not os.path.exists(img):
                continue
            sav = os.path.join(PROCESSED_DIR,"GTSRB", type, str(row['ClassId']))
            os.makedirs(sav,exist_ok=True)
            with Image.open(img) as im:
                sign = (row['Roi.X1'],row['Roi.Y1'],row['Roi.X2'],row['Roi.Y2'])
                cim = im.crop(sign)
                fname = os.path.basename(row['Path'])
                path = os.path.join(sav, f"{i}_{fname}")
                cim.save(path)

    train, test = pd.read_csv(os.path.join(datadir,"Train.csv")), pd.read_csv(os.path.join(datadir,"Test.csv"))
    cropsave(train,True)
    cropsave(test,False)

def procBTSD(datadir):
    def cropsave(folder,isTrain):
        type = 'train' if isTrain else 'test'
        folderPath = os.path.join(datadir, folder)

        if not os.path.exists(folderPath):
            alt = os.path.join(datadir,f"BelgiumTSC_{folder}")
            if os.path.exists(alt):
                folderPath = alt
            else:
                raise NotADirectoryError(f"{folder} not found. download btsd dataset or check folder name")

        CSVs = []
        for root,_,files in os.walk(folderPath):
            for file in files:
                if file.startswith('GT-') and file.endswith('.csv'):
                    CSVs.append(os.path.join(root,file))

        print(f"dataset: BTSD, type:{type}")
        def savData(data, dir):
            for i,row in tqdm(data.iterrows(),total=len(data)):
                img = os.path.join(dir, str(row['Filename']))
                if not os.path.exists(img):
                    continue
                sav = os.path.join(PROCESSED_DIR,"BTSD", type, str(row['ClassId']))
                os.makedirs(sav,exist_ok=True)
                with Image.open(img) as im:
                    sign = (row['Roi.X1'],row['Roi.Y1'],row['Roi.X2'],row['Roi.Y2'])
                    cim = im.crop(sign)
                    fname = os.path.basename(row['Filename'])
                    fname_clean = os.path.splitext(fname)[0]
                    path = os.path.join(sav, f"{i}_{fname_clean}.png")
                    cim.convert('RGB').save(path)
        for CSV in CSVs:
            dir = os.path.dirname(CSV)
            data = pd.read_csv(CSV, sep=';')
            data.columns = data.columns.str.strip()
            valid = (data['Roi.X2'] > data['Roi.X1']) & (data['Roi.Y2'] > data['Roi.Y1'])
            data = data[valid].dropna().reset_index(drop=True)
            savData(data,dir)

    cropsave("Training",True)
    cropsave("Testing",False)

if __name__ == '__main__':
    gdir,bdir,cdir = os.path.join(RAW_DIR,"GTSRB"), os.path.join(RAW_DIR,"BTSD"),os.path.join(RAW_DIR,"CTSD")
    if os.path.exists(bdir):
        procBTSD(bdir)
    if os.path.exists(cdir):
        procCTSD(cdir)
    if os.path.exists(gdir):
       procGTSRB(gdir)
    
    
            
            
