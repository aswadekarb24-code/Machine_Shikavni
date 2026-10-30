import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import optim

from src.config import BATCH_SIZE, CHECKPOINT, RESULTS
from src.ensemble import ENSEMBLE_MEMBERS, train_member
from src.models.factory import get_model
from src.models.helper import get_criterion
from src.preproc.factory import getloader, get_fulllloader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # models_to_train = ['trialcnn', 'resnet18', 'mobilenet_v2', 'efficientnet_b0']
    models_to_train = ['TSRDClassifier']

    for model_name in models_to_train:
        for ds in ['BTSD', 'CTSD', 'GTSRB']:
            for member in ENSEMBLE_MEMBERS:
                train_member(ds, member, arch=model_name, epochs=20 if ds == 'GTSRB' else 80, batch_size=256)
