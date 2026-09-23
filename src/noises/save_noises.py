import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.noises.corruptions import CORRUPTIONS, corruptimg
from src.config import EXTDIR

from src.preproc.makedat import (
    g_traindat, g_testdat,
    b_traindat, b_testdat,
    c_traindat,c_testdat
)

SEVERITIES = [1,2,3,4,5]
dsetmp = {
    'GTSRB' :{'train' : g_traindat, 'test':g_testdat},
    'BTSD' :{'train' : b_traindat, 'test':b_testdat},
    'CTSD' : {'train':c_traindat, 'test':c_testdat}
}

os.makedirs(EXTDIR,exist_ok=True)
for corr in CORRUPTIONS.keys():
    for sev in SEVERITIES:
        for dsnm, mp in dsetmp.items():
            print(f"Saving Dataset :{dsnm}, Corruption :{corr}, Severity :{sev}")
            for type, data in mp.items():
                dirr = os.path.join(EXTDIR, corr, str(sev), dsnm, type)
                os.makedirs(dirr, exist_ok=True)
                for i, (imgpath, label) in enumerate(tqdm(data.samples)):
                    fname = os.path.splitext(os.path.basename(imgpath))[0]
                    sname = f'{i}_{fname}.npy'
                    odirr = os.path.join(dirr,str(label))
                    os.makedirs(odirr,exist_ok=True)
                    spath = os.path.join(odirr, sname)
                    if os.path.exists(spath):
                        continue
                    
                    img = Image.open(imgpath).convert('RGB')
                    cimg = corruptimg(img,corr,sev)
                    cornp = np.array(cimg,dtype=np.uint8)
                    np.save(spath,cornp)