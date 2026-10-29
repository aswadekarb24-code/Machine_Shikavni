import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import optim

from src.config import CHECKPOINT, RESULTS
from src.models.factory import get_model
from src.models.helper import get_criterion
from src.preproc import transforms as T
from src.preproc.cache import VariantBank, apply_tform, to_input

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = {'GTSRB': 43, 'CTSD': 58, 'BTSD': 62}
ALL_SEV = (1, 2, 3, 4, 5)

ENSEMBLE_MEMBERS = {
    'noise': dict(tform=T.noise_tform, severities=ALL_SEV,
                  corruptions=['gaussian_noise', 'shot_noise', 'impulse_noise']),
    # dropped after the sweep: weakest member and removing it never hurt the ensemble (checkpoints kept on disk)
    # 'blur': dict(tform=T.blur_tform, severities=ALL_SEV,
    #              corruptions=['defocus_blur', 'motion_blur', 'zoom_blur']),
    'light': dict(tform=T.light_tform, severities=ALL_SEV,
                  corruptions=['brightness', 'contrast', 'fog']),
    'weather': dict(tform=T.weather_tform, severities=ALL_SEV,
                    corruptions=['snow', 'frost']),
    # 'distort': dict(tform=T.distort_tform, severities=ALL_SEV,
    #                 corruptions=['elastic_transform', 'pixelate', 'jpeg_compression']),
    'fog': dict(tform=T.fog_tform, severities=ALL_SEV,
                corruptions=['fog', 'contrast']),
    'defocus': dict(tform=T.defocus_tform, severities=ALL_SEV,
                    corruptions=['defocus_blur']),
    'heavyblur': dict(tform=T.heavyblur_tform, severities=ALL_SEV,
                      corruptions=['defocus_blur']),
    'motion': dict(tform=T.motion_tform, severities=ALL_SEV,
                   corruptions=['motion_blur']),
    # preprocessing identical to the previous single robust transform
    'generalist': dict(tform=T.generalist_tform, severities=ALL_SEV,
                       corruptions=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
                                    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']),
}


def ckpt_path(ds, member, arch='trialcnn'):
    return os.path.join(CHECKPOINT, f'{arch.lower()}_{ds.lower()}_ens_{member}.pth')


def metrics(y, y_):
    kw = dict(average='weighted', zero_division=0)
    return (accuracy_score(y, y_), f1_score(y, y_, **kw),
            precision_score(y, y_, **kw), recall_score(y, y_, **kw))


@torch.no_grad()
def evaluate_bank(model, bank, variant_ids, criterion, bs=2048):
    """Evaluate on a fixed variant per sample (variant_ids, shape (N,))."""
    model.eval()
    ys, ps, loss = [], [], 0.0
    for idx in torch.arange(len(bank)).split(bs):
        x = to_input(bank.data[variant_ids[idx].to(bank.device), idx.to(bank.device)].to(DEVICE))
        y = bank.labels[idx.to(DEVICE)]
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            out = model(x)
            loss += criterion(out, y).item() * len(idx)
        ys.append(y)
        ps.append(out.argmax(1))
    y, y_ = torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()
    return (loss / len(y), *metrics(y, y_))


def train_member(ds, member, arch='trialcnn', epochs=100, batch_size=256):
    """Train on CLEAN images only; the member's corruptions only determine its preprocessing design."""
    cfg = ENSEMBLE_MEMBERS[member]
    arch = arch.lower()
    print(f"Training ensemble member '{member}' ({arch}) on {ds} (clean data only), targets: {cfg['corruptions']}")
    train_bank = VariantBank(ds, 'train', cfg, DEVICE)
    test_bank = VariantBank(ds, 'test', cfg, DEVICE)
    test_v = torch.zeros(len(test_bank), dtype=torch.long)

    model = get_model(arch, in_channels=1, n_classes=N_CLASSES[ds]).to(DEVICE)
    criterion = get_criterion(ds)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    modelpath = ckpt_path(ds, member, arch)
    # One file per architecture/dataset/member; rows from later runs are appended.
    historyfile = os.path.join(RESULTS, f'history_{arch}_{ds.lower()}_ensemble_{member}.csv')
    os.makedirs(os.path.dirname(modelpath), exist_ok=True)
    os.makedirs(os.path.dirname(historyfile), exist_ok=True)

    bestf1, history = 0.0, []
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath, map_location=DEVICE))
        bestf1 = evaluate_bank(model, test_bank, test_v, criterion)[2]

    n = len(train_bank)
    for epoch in range(1, epochs + 1):
        model.train()
        cumloss = torch.zeros((), device=DEVICE)
        ys, ps = [], []
        for idx in torch.randperm(n).split(batch_size):
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            x = to_input(train_bank.draw(idx, 0.0).to(DEVICE, non_blocking=True))
            y = train_bank.labels[idx.to(DEVICE)]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cumloss += loss.detach() * len(idx)
            ys.append(y)
            ps.append(out.argmax(1))
        y, y_ = torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()
        trloss = cumloss.item() / len(y)
        tracc, trf1, trprec, trrec = metrics(y, y_)
        tstloss, tstacc, tstf1, tstprec, tstrec = evaluate_bank(model, test_bank, test_v, criterion)
        scheduler.step()

        history.append({
            'model': f'{arch}_{member}', 'architecture': arch,
            'dataset': ds, 'member': member, 'epoch': epoch,
            'train loss': trloss, 'train accuracy': tracc,
            'train f1': trf1, 'train precision': trprec, 'train recall': trrec,
            'test loss': tstloss, 'test accuracy': tstacc, 'test f1': tstf1,
            'test precision': tstprec, 'test recall': tstrec,
        })
        print(f"[{member.upper()}] Epoch: {epoch:03d}/{epochs:03d}")
        print(f"Train loss : {trloss:.4f} | Train acc : {tracc:.4f} | Train f1 : {trf1:.4f}")
        print(f"Test loss  : {tstloss:.4f} | Test acc  : {tstacc:.4f} | Test f1  : {tstf1:.4f}")

        if tstf1 > bestf1:
            bestf1 = tstf1
            torch.save(model.state_dict(), modelpath)
            print(f"New best checkpoint saved: {modelpath} | F1 = {bestf1:.4f}")

    pd.DataFrame(history).to_csv(
        historyfile,
        mode='a',
        header=not os.path.exists(historyfile),
        index=False,
    )


def load_members(ds, arch='trialcnn', names=None):
    """Load trained members for a dataset; missing checkpoints are skipped."""
    members = {}
    for name in (names or ENSEMBLE_MEMBERS):
        path = ckpt_path(ds, name, arch)
        if not os.path.exists(path):
            print(f"[SKIP] Member checkpoint not found: {path}")
            continue
        model = get_model(arch, in_channels=1, n_classes=N_CLASSES[ds]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        members[name] = model.eval()
    return members


@torch.no_grad()
def member_probs(members, raw, bs=2048):
    """Per-member softmax probabilities for raw (N, H, W, 3) uint8 images -> {name: (N, C) tensor on CPU}."""
    probs = {}
    for name, model in members.items():
        probs[name] = specialist_probs(name, {name: model}, raw, bs)[name]
    return probs


@torch.no_grad()
def specialist_probs(member, models, raw, bs=2048):
    """Evaluate several architectures sharing one specialist transform. (avoids recomputation)
    """
    x_all = torch.from_numpy(apply_tform(ENSEMBLE_MEMBERS[member]['tform'], raw))
    probs = {}
    for name, model in models.items():
        chunks = []
        for x in x_all.split(bs):
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                out = model(to_input(x.to(DEVICE)))
            chunks.append(F.softmax(out.float(), dim=1).cpu())
        probs[name] = torch.cat(chunks)
    return probs


CONF_POWER = 4


def ensemble_probs(probs):
    """Confidence-weighted average: each member's weight per sample is (its top probability) ** CONF_POWER.
    """
    P = torch.stack(list(probs.values()))
    w = P.max(2, keepdim=True).values ** CONF_POWER
    return (w * P).sum(0) / w.sum(0)
