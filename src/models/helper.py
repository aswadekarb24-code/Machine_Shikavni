import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.preproc.makedat import b_traindat, c_traindat, g_traindat

DEVICE = torch.device('cuda' if torch.cuda.is_available() else'cpu')

datasetmp = {
    'GTSRB' :g_traindat,
    'BTSD': b_traindat,
    'CTSD' : c_traindat
}

def get_criterion(datanm):
    if datanm not in datasetmp:
        raise ValueError("Invalid dataset name")

    dataset = datasetmp[datanm]

    class_wts = compute_class_weight(
        classes=np.unique(dataset.targets),
        y=dataset.targets,
        class_weight='balanced'
    )
    wts_tensor = torch.tensor(class_wts,dtype=torch.float32).to(DEVICE)
    return torch.nn.CrossEntropyLoss(weight=wts_tensor)