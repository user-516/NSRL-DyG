"""Controlled single-injection probe for the noise domino effect.

Protocol
--------
For every trial the script runs four copies of the same trained model with identical
weights and identical initial memory:

    RNS-clean / RNS-noisy        gate active
    w/o-RNS-clean / w/o-RNS-noisy  gate bypassed

Each copy first replays the same warm-up history, so all four enter the measurement
window with the same memory.  Then one edge (u, v, t0) is injected into the two noisy
copies only, and the same subsequent events are replayed through all four.  Because the
two arms see the same injection positions and the same noise edges, the comparison is
paired.

After every event k we record

    delta_i(k) = || e_i^noisy(k) - e_i^clean(k) ||_2

and report the four quantities the referee asks for:

    deviation     mean delta over the nodes reachable from {u, v}
    spread        number of nodes with delta > eps
    accumulation  mean delta as a function of k
    damage        AP on the replayed events, noisy vs clean, per block

Many trials are averaged, and the mean +/- standard deviation across trials is plotted.

Determinism
-----------
The reference model instantiates its aggregation and attention modules inside
forward(), so their weights are redrawn on every call and eval is NOT deterministic.
The probe reseeds before every forward, which makes the four copies comparable; the
same issue also means the published AP/AUC are averages over random projections.

Usage
-----
    python domino_probe.py --ckpt log/NSRLDyG/<run>/checkpoints/epochN.pth \
        -d uci --data_root E:/dataset/Ada-DyGNN-dataset \
        --start 48000 --stride 2000 --trials 10 \
        --warmup 400 --horizon 500 --out figures/domino
"""

import argparse
import collections
import copy
import os

import numpy as np
import torch

from models.NSRLDyG import NSRLDyG
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder

DET_SEED = 12345


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint written by torch.save(model, ...)')
    ap.add_argument('-d', '--data', default='uci')
    ap.add_argument('--data_root', default='./data')
    ap.add_argument('--start', type=int, default=48000,
                    help='window start of the first trial')
    ap.add_argument('--stride', type=int, default=2000,
                    help='spacing between trial windows')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=400,
                    help='events replayed identically through every copy before injection')
    ap.add_argument('--horizon', type=int, default=500,
                    help='events replayed after the injection')
    ap.add_argument('--eps', type=float, default=1e-3,
                    help='threshold used for the affected-node count')
    ap.add_argument('--block', type=int, default=50,
                    help='block size for the AP damage curve')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='figures/domino')
    return ap.parse_args()


# ---------------------------------------------------------------------------
def run_event(model, s, d, neg, e, t):
    """One event, batch size one, no grad, deterministic.

    Reseeding is required because the reference model builds its attention modules
    inside forward(); without it two identical calls differ.
    """
    torch.manual_seed(DET_SEED)
    with torch.no_grad():
        return model(np.array([s]), np.array([d]), np.array([neg]),
                     np.array([e]), np.array([t]))


def hops_from(edges, sources, n_nodes):
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = {int(s): 0 for s in sources}
    queue = collections.deque(dist)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return np.array([dist.get(i, -1) for i in range(n_nodes)])


def block_ap(pos, neg, block):
    from sklearn.metrics import average_precision_score
    pos, neg = np.asarray(pos), np.asarray(neg)
    out = []
    for i in range(0, len(pos), block):
        p, n = pos[i:i + block], neg[i:i + block]
        if len(p) < 2:
            continue
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        out.append(average_precision_score(y, np.r_[p, n]))
    return np.asarray(out)


# ---------------------------------------------------------------------------
def run_arm(base_model, finder, full, args, st, u, v, t0, noise_edge, noise_neg,
            use_rns, n_nodes):
    """Replay one arm (RNS or w/o-RNS) and return its measurements."""
    clean = copy.deepcopy(base_model)
    noisy = copy.deepcopy(base_model)
    for m in (clean, noisy):
        m.use_rns = use_rns
        m.eval()
        m.set_neighbor_finder(finder)
        m.reset_graph()

    w0 = max(0, st - args.warmup)
    for j in range(w0, st):
        s, d = int(full.src[j]), int(full.dst[j])
        e, t = int(full.edge_idxs[j]), float(full.timestamps[j])
        run_event(clean, s, d, noise_neg, e, t)
        run_event(noisy, s, d, noise_neg, e, t)

    gap = float((noisy.memory.emb - clean.memory.emb).abs().max())
    if gap > 1e-5:
        raise SystemExit(f'copies diverged during warm-up (max|diff|={gap:.2e})')

    run_event(noisy, u, v, noise_neg, noise_edge, t0)

    deltas, n_aff, pc, nc_, pn, nn_ = [], [], [], [], [], []
    for k in range(args.horizon):
        j = st + k
        s, d = int(full.src[j]), int(full.dst[j])
        e, t = int(full.edge_idxs[j]), float(full.timestamps[j])
        rc = run_event(clean, s, d, noise_neg, e, t)
        rn = run_event(noisy, s, d, noise_neg, e, t)
        dev = (noisy.memory.emb - clean.memory.emb).norm(dim=1).cpu().numpy()
        deltas.append(dev)
        n_aff.append(int((dev > args.eps).sum()))
        pc.append(float(rc[0])); nc_.append(float(rc[1]))
        pn.append(float(rn[0])); nn_.append(float(rn[1]))

    deltas = np.stack(deltas)                       # (horizon, n_nodes)
    reach = hops_from(list(zip(full.src[st:st + args.horizon],
                               full.dst[st:st + args.horizon])), [u, v], n_nodes)
    reached = reach >= 0
    mean_reach = deltas[:, reached].mean(axis=1) if reached.any() else deltas.mean(axis=1)
    ap_c, ap_n = block_ap(pc, nc_, args.block), block_ap(pn, nn_, args.block)

    return dict(
        delta=deltas, mean=mean_reach, n_aff=np.asarray(n_aff),
        reach=reach, reached=reached,
        ap_c=float(ap_c.mean()), ap_n=float(ap_n.mean()),
    )


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    node_features, edge_features, full, train_data, val_data, test_data, _, _, _ = \
        get_data(args.data, args.data_root)
    finder = get_neighbor_finder(full, False)

    base = torch.load(args.ckpt, map_location=device, weights_only=False)
    base.to(device)
    base.eval()
    base.set_neighbor_finder(finder)
    base.use_rns = True
    n_nodes = base.memory.n_nodes

    trials = {'RNS': [], 'w/o-RNS': []}
    for t in range(args.trials):
        st = args.start + t * args.stride
        if st + args.horizon >= len(full.src):
            print(f'  trial {t}: window beyond the stream, stopping at {t} trials')
            break

        win = slice(st, st + args.horizon)
        counts = collections.Counter(np.r_[full.src[win], full.dst[win]].tolist())
        u = counts.most_common(1)[0][0]                    # most active node in the window
        v = int(np.random.randint(0, n_nodes))
        while v == u:
            v = int(np.random.randint(0, n_nodes))
        t0 = float(full.timestamps[st]) - 1e-6
        noise_edge = int(np.random.choice(full.edge_idxs[win]))
        noise_neg = int(np.random.randint(0, n_nodes))

        for arm, use_rns in (('RNS', True), ('w/o-RNS', False)):
            r = run_arm(base, finder, full, args, st, u, v, t0, noise_edge,
                        noise_neg, use_rns, n_nodes)
            r['start'], r['u'], r['v'] = st, u, v
            trials[arm].append(r)
        print(f'  trial {t + 1}/{args.trials}  start={st}  injected=({u},{v})')

    if not trials['RNS']:
        raise SystemExit('no completed trial')

    # ---- aggregate --------------------------------------------------------
    def stack(arm, key):
        return np.stack([r[key] for r in trials[arm]])

    summary = {}
    for arm in ('RNS', 'w/o-RNS'):
        m = stack(arm, 'mean')                          # (trials, horizon)
        a = stack(arm, 'n_aff')
        summary[arm] = dict(
            mean=m.mean(axis=0), std=m.std(axis=0),
            aff=a.mean(axis=0), aff_std=a.std(axis=0),
            peak_aff=a.max(axis=1).mean(),
            dev_1=m[:, 0].mean(),
            dev_early=m[:, :min(50, m.shape[1])].mean(),
            dev_late=m[:, -min(50, m.shape[1]):].mean(),
            ap_c=np.mean([r['ap_c'] for r in trials[arm]]),
            ap_n=np.mean([r['ap_n'] for r in trials[arm]]),
        )

    print('\n=== summary over %d trials ===' % len(trials['RNS']))
    print(f'{"metric":<34}{"RNS":>12}{"w/o-RNS":>12}')
    for key, fmt in (('dev_1', '.4e'), ('dev_early', '.4e'), ('dev_late', '.4e'),
                     ('peak_aff', '.1f')):
        print(f'{key:<34}{format(summary["RNS"][key], fmt):>12}'
              f'{format(summary["w/o-RNS"][key], fmt):>12}')
    for arm in ('RNS', 'w/o-RNS'):
        print(f'{arm + " AP clean->noisy":<34}'
              f'{summary[arm]["ap_c"]:.4f}->{summary[arm]["ap_n"]:.4f}')

    # final-event deviation by hop, averaged over trials
    print('\nfinal deviation by hop (mean over trials):')
    for arm in ('RNS', 'w/o-RNS'):
        vals = {}
        for r in trials[arm]:
            for h in np.unique(r['reach'][r['reached']]):
                vals.setdefault(int(h), []).append(r['delta'][-1][r['reach'] == h].mean())
        row = '  '.join(f'h{h}:{np.mean(v):.2e}' for h, v in sorted(vals.items()))
        print(f'  {arm:<8} {row}')

    # ---- plots ------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    colors = {'RNS': '#2a78d6', 'w/o-RNS': '#e34948'}
    ks = np.arange(1, args.horizon + 1)

    def frame(ax, ylabel, title):
        ax.set_xlabel('subsequent events $k$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, color='#e1e0d9', lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for arm in ('RNS', 'w/o-RNS'):
        mu, sd = summary[arm]['mean'], summary[arm]['std']
        ax.plot(ks, mu, lw=2, color=colors[arm], label=arm)
        ax.fill_between(ks, mu - sd, mu + sd, color=colors[arm], alpha=0.18, lw=0)
    ax.legend(frameon=False)
    frame(ax, r'mean $\delta(k)$', 'Error accumulation')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}_accumulation.{ext}', dpi=250, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for arm in ('RNS', 'w/o-RNS'):
        mu, sd = summary[arm]['aff'], summary[arm]['aff_std']
        ax.plot(ks, mu, lw=2, color=colors[arm], label=arm)
        ax.fill_between(ks, mu - sd, mu + sd, color=colors[arm], alpha=0.18, lw=0)
    ax.legend(frameon=False)
    frame(ax, r'nodes with $\delta>\varepsilon$', 'Propagation spread')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}_spread.{ext}', dpi=250, bbox_inches='tight')

    print(f'\nsaved {args.out}_accumulation and {args.out}_spread (.png/.pdf)')


if __name__ == '__main__':
    main()
