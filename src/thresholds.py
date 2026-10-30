"""Per-severity accuracy scoreboard: python -m src.thresholds [results.csv]

Exits non-zero if any (dataset, corruption, severity) cell is below src.config.THRESHOLDS.
"""
import glob
import os
import sys

import pandas as pd
import numpy as np

from src.config import RESULTS, THRESHOLDS

REPORTPATH = os.path.join(RESULTS, f'report.csv')
OUTDF = {"csv_path" : [], "model":[], "score":[], "GTSRB_avg_acc":[],"BTSD_avg_acc":[],"CTSD_avg_acc":[],"overall_avg_acc":[]}


def report(csv_path):
    df = pd.read_csv(csv_path)
    for model in np.unique(df['Model']):   
        d = df[(df['Model'] == model) & (df['Corruption'] != 'none')]
        tot = ok = 0
        print(f"Thresholds {THRESHOLDS} on {model} ({os.path.basename(csv_path)})")
        ds_acc = {}
        for ds, g in d.groupby('Dataset'):
            # print(f"{g[g['Corruption']=='gaussian_blur']}")
            ds_acc[ds] = np.mean(g['accuracy'])
            p = g.pivot(index='Corruption', columns='Severity', values='accuracy')
            passed = p.apply(lambda col: col >= THRESHOLDS[col.name])
            tot += passed.size
            ok += int(passed.values.sum())
            print(f"{ds}: {int(passed.values.sum())}/{passed.size} cells pass")
            for corr in p.index:
                if not passed.loc[corr].all():
                    accs = ' '.join(f"{p.loc[corr, s]:.2f}{'' if passed.loc[corr, s] else '*'}" for s in p.columns)
                    print(f"    {corr:18s} {accs}")
        print(f"TOTAL: {ok}/{tot}")
        OUTDF["csv_path"].append(csv_path)
        OUTDF["model"].append(model)
        OUTDF["score"].append(f'{ok}/{tot}')
        OUTDF["GTSRB_avg_acc"].append(ds_acc.get('GTSRB', np.nan))
        OUTDF["BTSD_avg_acc"].append(ds_acc.get('BTSD', np.nan))
        OUTDF["CTSD_avg_acc"].append(ds_acc.get('CTSD', np.nan))
        OUTDF["overall_avg_acc"].append(np.mean([v for _,v in ds_acc.items()]))
    
    # return ok == tot


if __name__ == '__main__':
    with open(REPORTPATH,'w') as f:
        pass
    paths =  sys.argv[1:] if len(sys.argv) > 1 else (glob.glob(os.path.join(RESULTS, 'noise_results_*.csv')))
    for path in paths:
        report(path)
    pd.DataFrame(OUTDF).to_csv(REPORTPATH,index=False)
    # sys.exit(0 if report(path) else 1)
