"""Small hyper-parameter sweep over the paper's method.

Each run is a separate trainAUC.py process. The script records the best validation
AUC and the test AUC reported from the corresponding checkpoint, so the sweep can be
driven by validation rather than by the test set.

Usage:
    python sweep.py stage1
"""
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sweep_results.csv'

STAGES = {
    # stabilise the policy term first: learning rate x RL weight
    'stage1': dict(
        epochs=3,
        grid=dict(lr=[1e-4, 1e-3], rl_coef=[0.05, 0.5],
                  dim=[64], n_degree=[20], n_update_degree=[5], lambda2=[0.1]),
    ),
    # then the entropy weight and the candidate budget
    'stage2': dict(
        epochs=4,
        grid=dict(lr=[1e-3], rl_coef=[0.5],
                  dim=[64, 128], n_degree=[20], lambda2=[0.01, 0.1, 1.0]),
    ),
}

EPOCH_RE = re.compile(r'epoch\s+(\d+)\s+\|\s+trans\s+AUC\s+([0-9.]+)')
TEST_RE = re.compile(r'in test,.*auc = ([0-9.]+)')


def run(cfg, epochs):
    cmd = [PY, str(ROOT / 'trainAUC.py'), '--model', 'tfdga', '--selection_mode', 'mdp',
           '--n_epoch', str(epochs), '--bs', '64']
    for k, v in cfg.items():
        cmd += [f'--{k}', str(v)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    vals = [float(m.group(2)) for m in EPOCH_RE.finditer(proc.stdout)]
    tests = [float(m.group(1)) for m in TEST_RE.finditer(proc.stdout)]
    return (max(vals) / 100 if vals else float('nan'),
            tests[-1] if tests else float('nan'),
            proc.returncode)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    spec = STAGES[stage]
    keys = list(spec['grid'])
    new = not OUT.exists()
    with OUT.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(['stage'] + keys + ['best_val_auc', 'test_auc', 'rc'])
        for values in itertools.product(*(spec['grid'][k] for k in keys)):
            cfg = dict(zip(keys, values))
            best_val, test, rc = run(cfg, spec['epochs'])
            writer.writerow([stage] + list(values) + [f'{best_val:.4f}', f'{test:.4f}', rc])
            fh.flush()
            print(f'{cfg}  best_val={best_val:.4f}  test={test:.4f}  rc={rc}', flush=True)


if __name__ == '__main__':
    main()
