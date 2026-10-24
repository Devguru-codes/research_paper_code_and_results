"""Shared protocol for the revision reruns (REVISION_PLAN.md, section 1).

Every run script starts with `from hsi_common import *`; build.sh pastes this file
into each Kaggle kernel so all runs use exactly the same code.
"""
import os, re, sys, json, time, copy, random, gc, urllib.request
import numpy as np, pandas as pd, scipy, scipy.io as sio, sklearn
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb

SMOKE = globals().get('SMOKE', False)  # build.sh sets this; smoke = 1 seed, 1 split, 2 epochs
SEEDS = [42] if SMOKE else [42, 43, 44, 45, 46]
SPLITS = [0.05] if SMOKE else [0.01, 0.05, 0.10, 0.20]
EPOCHS = 2 if SMOKE else 50
PATIENCE, LR, WD, BATCH = 10, 5e-4, 1e-4, 64
FREEZE_EPOCHS = 1 if SMOKE else 5
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else 'out'
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def free(*objs):
    del objs
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- data
def load_indian_pines(patch=15):
    """All labelled pixels as 15x15 patches (reflect-padded). Returns X (N,H,W,200), y in 0..15."""
    base = 'https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/'
    for f in ['Indian_pines_corrected.mat', 'Indian_pines_gt.mat']:
        if not os.path.exists(f):
            urllib.request.urlretrieve(base + f, f)
    X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected'].astype(np.float32)
    gt = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    m = patch // 2
    Xp = np.pad(X, ((m, m), (m, m), (0, 0)), mode='reflect')
    rows, cols = np.nonzero(gt)
    return np.stack([Xp[r:r + patch, c:c + patch] for r, c in zip(rows, cols)]), (gt[rows, cols] - 1).astype(np.int64)


def ip_split(y, frac, seed):
    """Train = frac of labelled pixels; val = 10% of the rest; test = remainder (all stratified)."""
    tr, rest = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=0.1, stratify=y[rest], random_state=seed)
    return tr, va, te


# Scenes for the spatially disjoint benchmark: (file, key, gt file, gt key, tile size in pixels)
SCENES = {'IndianPines': ('Indian_pines_corrected.mat', 'indian_pines_corrected', 'Indian_pines_gt.mat', 'indian_pines_gt', 24),
          'PaviaU': ('PaviaU.mat', 'paviaU', 'PaviaU_gt.mat', 'paviaU_gt', 32),
          'Salinas': ('Salinas_corrected.mat', 'salinas_corrected', 'Salinas_gt.mat', 'salinas_gt', 32)}


def load_scene(name):
    """Cube (H,W,C) float32 and ground truth (H,W) with 0 = unlabelled, classes 1..K."""
    f, k, gf, gk, _ = SCENES[name]
    for x in (f, gf):
        if not os.path.exists(x):
            urllib.request.urlretrieve('https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/' + x, x)
    return sio.loadmat(f)[k].astype(np.float32), sio.loadmat(gf)[gk].astype(np.int64)


def disjoint_tiles(gt, seed, tile, margin=7, tries=400, big=50):
    """Train/test tile map for one seed: square tiles split 50/50, keeping the best of `tries` random
    assignments. Ranking: classes (with >= `big` pixels) that have a fair share in both halves
    (>= 25% of the class in the training area and >= 10% in the test area), then such classes present
    at all (>= 10 training pixels, >= 1 test pixel), then test size. Test pixels within `margin` pixels
    (Chebyshev) of a training tile are dropped, so no 15x15 test window contains a training-area pixel.
    Returns boolean masks (train_pool, test) over labelled pixels."""
    from scipy.ndimage import binary_dilation
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    ty, tx = -(-h // tile), -(-w // tile)
    lab, best = gt > 0, None
    total = {c: (gt == c).sum() for c in np.unique(gt[lab]) if (gt == c).sum() >= big}
    for _ in range(tries):
        role = np.kron(rng.permutation(np.r_[np.zeros(ty * tx // 2), np.ones(ty * tx - ty * tx // 2)]).reshape(ty, tx),
                       np.ones((tile, tile)))[:h, :w]
        test = lab & (role == 1) & ~binary_dilation(role == 0, structure=np.ones((2 * margin + 1, 2 * margin + 1)))
        pool = lab & (role == 0)
        n_pool = {c: (pool & (gt == c)).sum() for c in total}
        n_test = {c: (test & (gt == c)).sum() for c in total}
        key = (sum(n_pool[c] >= 0.25 * t and n_test[c] >= 0.10 * t for c, t in total.items()),
               sum(n_pool[c] >= 10 and n_test[c] > 0 for c in total), test.sum())
        if best is None or key > best[0]:
            best = (key, pool, test)
    return best[1], best[2]


def disjoint_split(gt, frac, seed, tile, val_cap=3000):
    """Spatially disjoint split. Training pixels: `frac` of all labelled pixels, drawn per class in
    proportion to the training area (at least 1 per class present). Validation: up to `val_cap` of the
    remaining training-area pixels (used only for early stopping). Test: the buffered test area, fixed
    per seed so all budgets are scored on the same pixels. Returns three (rows, cols) tuples."""
    pool, test = disjoint_tiles(gt, seed, tile)
    rng = np.random.default_rng(seed + 1000 + round(frac * 1e4))
    n_total, n_pool, tr = round(frac * (gt > 0).sum()), pool.sum(), []
    for c in np.unique(gt[pool]):
        cand = np.argwhere(pool & (gt == c))
        tr.append(cand[rng.choice(len(cand), min(len(cand), max(1, round(n_total * len(cand) / n_pool))), replace=False)])
    tr = np.concatenate(tr)
    rest = pool.copy(); rest[tr[:, 0], tr[:, 1]] = False
    va = np.argwhere(rest)
    va = va[rng.choice(len(va), min(val_cap, len(va)), replace=False)]
    te = np.argwhere(test & np.isin(gt, np.unique(gt[pool])))  # score only classes the training area contains
    return (tr[:, 0], tr[:, 1]), (va[:, 0], va[:, 1]), (te[:, 0], te[:, 1])


class ScenePatches(Dataset):
    """15x15 windows centred on (rows, cols) of a reflect-padded scene, standardised per band."""
    def __init__(self, padded, gt, rc, mean, std, patch=15):
        self.p, self.gt, self.r, self.c, self.mean, self.std, self.k = padded, gt, rc[0], rc[1], mean, std, patch

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r, c = self.r[i], self.c[i]
        x = (self.p[r:r + self.k, c:c + self.k] - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]), int(self.gt[r, c]) - 1


def scene_loaders(cube, gt, parts, seed, patch=15):
    """parts = [(rows, cols)] for train, val, test. Band statistics from the training centre pixels."""
    m = patch // 2
    padded = np.pad(cube, ((m, m), (m, m), (0, 0)), mode='reflect')
    x = cube[parts[0][0], parts[0][1]]
    mean, std = x.mean(0), x.std(0) + 1e-8
    g = torch.Generator().manual_seed(seed)
    return [DataLoader(ScenePatches(padded, gt, rc, mean, std, patch), batch_size=BATCH if k == 0 else 256,
                       shuffle=k == 0, generator=g if k == 0 else None, num_workers=2, pin_memory=True)
            for k, rc in enumerate(parts)]


LOCAL_CLASSES = ['black', 'red', 'yellow']


def local_label(path):
    """Class from the top-level '<Colour> Soil' folder. Mixed folders (Yellow+Black, Mix BRY, Mix) return -1."""
    m = re.search(r'(black|red|yellow) soil', path.lower())
    return LOCAL_CLASSES.index(m.group(1)) if m else -1


def local_sample_id(path):
    """'Red Soil/Moist/3-2.bil' -> 'Red Soil/Moist/3': images 3-1 and 3-2 show the same physical sample."""
    return os.path.join(os.path.dirname(path), os.path.basename(path).split('-')[0])


def local_soil_id(path):
    """'Red Soil/Moist/3-1.bil' and 'Red Soil/11-10-22/3-2.bil' -> '.../Red Soil/3': the moist and dry Red
    folders reuse sample numbers and may hold the same soil, so they are never split apart (R11b)."""
    m = re.search(r'(black|red|yellow) soil', path, re.I)
    return path[:m.end()] + '/' + os.path.basename(path).split('-')[0]


def local_files(root='/kaggle/input'):
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(root, followlinks=True)
                   for f in fs if f.endswith('.bil') and local_label(os.path.join(d, f)) >= 0)
    assert files, f'no labelled .bil files under {root}'
    return files


def local_samples(root='/kaggle/input', key=local_sample_id):
    """{sample_id: (class, [image files])} for the three labelled soil folders."""
    out = {}
    for f in local_files(root):
        out.setdefault(key(f), (local_label(f), []))[1].append(f)
    return out


def local_file_split(root='/kaggle/input', seed=0):
    """Per class, split physical samples 60/20/20 into train/val/test, so no sample is in two sets."""
    files = local_files(root)
    rng = np.random.default_rng(seed)
    split = {'train': [], 'val': [], 'test': []}
    for c in range(len(LOCAL_CLASSES)):
        samples = sorted({local_sample_id(f) for f in files if local_label(f) == c})
        rng.shuffle(samples)
        n = max(1, round(0.2 * len(samples)))
        parts = {'test': samples[:n], 'val': samples[n:2 * n], 'train': samples[2 * n:]}
        for k, ss in parts.items():
            split[k] += [f for f in files if local_label(f) == c and local_sample_id(f) in ss]
    json.dump(split, open(f'{OUT}/local_files.json', 'w'), indent=1)
    return split


def read_bil(path, samples=320, bands=168):
    raw = np.fromfile(path, dtype=np.uint16)
    lines = raw.size / (samples * bands)
    return raw.reshape(int(lines), bands, samples).transpose(0, 2, 1) if lines.is_integer() else None


def soil_mask(img):
    """Soil pixels of one image: the band-mean image divided by each column's 90th percentile (the bright
    tray; removes the left-right vignetting) is below 0.6. Checked visually on 24 images."""
    b = img.mean(2)
    return b / np.percentile(b, 90, axis=0, keepdims=True) < 0.6


def load_local_files(files, patch=15, soil=None):
    """Non-overlapping 15x15 patches from each image. Returns X (N,H,W,168) float32, y.
    soil=(lo, hi): keep only patches whose soil fraction (soil_mask) lies in [lo, hi]."""
    X, y = [], []
    for f in files:
        img = read_bil(f)
        if img is None:
            print('skip (bad size):', f)
            continue
        h, w, _ = img.shape
        keep = soil_mask(img) if soil else None
        p = [img[i:i + patch, j:j + patch] for i in range(0, h - patch + 1, patch) for j in range(0, w - patch + 1, patch)
             if not soil or soil[0] <= keep[i:i + patch, j:j + patch].mean() <= soil[1]]
        X += p
        y += [local_label(f)] * len(p)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def band_stats(X, idx, chunk=2000):
    """Per-band mean/std over the given rows (chunked: no full copy of large training sets)."""
    s = sq = 0.0
    n = 0
    for k in range(0, len(idx), chunk):
        x = X[idx[k:k + chunk]].astype(np.float64)
        s = s + x.sum(axis=(0, 1, 2)); sq = sq + (x ** 2).sum(axis=(0, 1, 2)); n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s / n
    return mean.astype(np.float32), (np.sqrt(np.maximum(sq / n - mean ** 2, 0)) + 1e-8).astype(np.float32)


class Patches(Dataset):
    """Rows `idx` of X (N,H,W,C) -> (1,C,H,W), standardised per band with train-set statistics."""
    def __init__(self, X, y, idx, mean, std):
        self.X, self.y, self.idx, self.mean, self.std = X, y, idx, mean, std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (self.X[j] - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]), int(self.y[j])


def make_loaders(parts, seed):
    """parts = [(X, y, idx)] for train, val, test; idx=None means every row. Stats from train only."""
    parts = [(X, y, np.arange(len(y)) if i is None else i) for X, y, i in parts]
    mean, std = band_stats(parts[0][0], parts[0][2])
    g = torch.Generator().manual_seed(seed)
    return [DataLoader(Patches(X, y, i, mean, std), batch_size=BATCH if k == 0 else 256, shuffle=k == 0,
                       generator=g if k == 0 else None, num_workers=2, pin_memory=True)
            for k, (X, y, i) in enumerate(parts)]


def centre(X):
    return X[:, X.shape[1] // 2, X.shape[2] // 2]


# ---------------------------------------------------------------- models
# Every model takes (B,1,C,H,W), has .features(x) and a final linear head .fc.
def mid(x):
    return x[:, 0, :, x.shape[3] // 2, x.shape[4] // 2]  # centre-pixel spectrum (B,C)


def encoder(dim, depth, heads):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, batch_first=True), depth, enable_nested_tensor=False)


class ViT(nn.Module):
    """Spectral transformer on the centre pixel: one token per band."""
    def __init__(self, bands, classes, dim=128, depth=4, heads=4):
        super().__init__()
        self.patch_embed = nn.Linear(1, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, bands, dim))
        self.transformer = encoder(dim, depth, heads)
        self.fc = nn.Linear(dim, classes)

    def features(self, x):
        return self.transformer(self.patch_embed(mid(x).unsqueeze(-1)) + self.pos_embed).mean(1)

    def forward(self, x):
        return self.fc(self.features(x))


class SpatialViT(ViT):
    """One token per band; each token is that band's whole 15x15 plane, densely projected (no convolution)."""
    def __init__(self, bands, classes, dim=128, depth=4, heads=4, patch=15):
        super().__init__(bands, classes, dim, depth, heads)
        self.patch_embed = nn.Sequential(nn.Linear(patch * patch, 16), nn.ReLU(), nn.Linear(16, dim))

    def features(self, x):
        return self.transformer(self.patch_embed(x[:, 0].flatten(2)) + self.pos_embed).mean(1)


class ConvStemViT(ViT):
    """Same transformer, but tokens come from a small 3-D convolution stem (local inductive bias)."""
    def __init__(self, bands, classes, dim=128, depth=4, heads=4):
        super().__init__(bands, classes, dim, depth, heads)
        self.stem = nn.Sequential(nn.Conv3d(1, 8, (7, 3, 3)), nn.ReLU(), nn.Conv3d(8, 16, (5, 3, 3)), nn.ReLU(),
                                  nn.AdaptiveAvgPool3d((bands, 1, 1)))
        self.patch_embed = nn.Linear(16, dim)

    def features(self, x):
        return self.transformer(self.patch_embed(self.stem(x).flatten(2).transpose(1, 2)) + self.pos_embed).mean(1)


class SFBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        return x + self.mlp(self.norm2(x))


class SpectralFormer(nn.Module):
    """SpectralFormer-inspired: group-wise spectral tokens (4 bands each) + cross-layer adaptive fusion."""
    def __init__(self, bands, classes, group=4, dim=64, depth=4, heads=4):
        super().__init__()
        self.group, self.tokens = group, bands // group
        self.group_embed = nn.Linear(group, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.tokens, dim) * 0.02)
        self.blocks = nn.ModuleList(SFBlock(dim, heads) for _ in range(depth))
        self.alpha = nn.Parameter(torch.zeros(depth - 1))  # sigmoid(0) = 0.5, as in the original runs
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, classes)

    def features(self, x):
        s = mid(x)[:, :self.tokens * self.group]
        x = self.group_embed(s.view(len(s), self.tokens, self.group)) + self.pos_embed
        for i, block in enumerate(self.blocks):
            out = block(x)
            a = torch.sigmoid(self.alpha[i - 1]) if i else None
            x = a * out + (1 - a) * x if i else out
        return self.norm(x).mean(1)

    def forward(self, x):
        return self.fc(self.features(x))


class HybridSN(nn.Module):
    """HybridSN (Roy et al. 2020): three 3-D convs, one 2-D conv, two dense layers."""
    def __init__(self, bands, classes, patch=15):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = nn.Conv3d(1, 8, (7, 3, 3)), nn.Conv3d(8, 16, (5, 3, 3)), nn.Conv3d(16, 32, (3, 3, 3))
        d, s = bands - 12, patch - 6
        self.conv4 = nn.Conv2d(32 * d, 64, 3)
        self.fc1, self.fc2, self.fc = nn.Linear(64 * (s - 2) ** 2, 256), nn.Linear(256, 128), nn.Linear(128, classes)
        self.drop = nn.Dropout(0.4)

    def conv3d(self, x):
        return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))

    def probe_features(self, x):
        """Last layer that transfers between 168 and 200 bands: conv3, averaged over space."""
        return self.conv3d(x).mean(dim=(3, 4)).flatten(1)

    def features(self, x):
        x = self.conv3d(x)
        x = F.relu(self.conv4(x.flatten(1, 2))).flatten(1)
        return self.drop(F.relu(self.fc2(self.drop(F.relu(self.fc1(x))))))

    def forward(self, x):
        return self.fc(self.features(x))


MODELS = dict(ViT=ViT, SpatialViT=SpatialViT, ConvStemViT=ConvStemViT, SpectralFormer=SpectralFormer, HybridSN=HybridSN)


# ---------------------------------------------------------------- train / evaluate
def evaluate(model, loader):
    model.eval()
    loss, ys, ps = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE, non_blocking=True)); y = y.to(DEVICE)
            loss += F.cross_entropy(out, y, reduction='sum').item()
            ys.append(y.cpu()); ps.append(out.argmax(1).cpu())
    return loss / len(loader.dataset), torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metrics(y, p):
    labels = np.unique(y)  # AA and F1 over the classes present in the test set
    return dict(OA=accuracy_score(y, p) * 100, AA=recall_score(y, p, labels=labels, average='macro', zero_division=0) * 100,
                Kappa=float(cohen_kappa_score(y, p)), F1=f1_score(y, p, labels=labels, average='macro', zero_division=0))


def train(model, tr, va, frozen=(), freeze_epochs=0):
    """AdamW + cosine, early stop on validation loss, restore best weights.
    With freeze_epochs: params named in `frozen` stay fixed for that many epochs, then all train at LR/5."""
    model.to(DEVICE)
    def opt_for(params, lr, steps):
        o = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
        return o, torch.optim.lr_scheduler.CosineAnnealingLR(o, max(steps, 1))
    if freeze_epochs:
        opt, sched = opt_for([p for n, p in model.named_parameters() if n not in set(frozen)], LR, freeze_epochs)
    else:
        opt, sched = opt_for(model.parameters(), LR, EPOCHS)
    best, best_state, bad, t0 = float('inf'), copy.deepcopy(model.state_dict()), 0, time.time()
    for ep in range(EPOCHS):
        if freeze_epochs and ep == freeze_epochs:
            opt, sched = opt_for(model.parameters(), LR / 5, EPOCHS - freeze_epochs)
        model.train()
        for x, y in tr:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(DEVICE, non_blocking=True)), y.to(DEVICE)).backward()
            opt.step()
        sched.step()
        vl = evaluate(model, va)[0]
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, dict(epochs=ep + 1, train_time=time.time() - t0, val_loss=best)


def test(model, loader):
    _, y, p = evaluate(model, loader)
    return metrics(y, p)


# Default LightGBM diverged on one Salinas split (raw scores ~8e4, OA 24%); capping each tree's leaf
# output fixes it without changing results elsewhere (Pavia U mean OA 76.8 -> 76.9).
LGBM_STABLE = dict(max_delta_step=1)


def classic(kind, seed):
    return (lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1, **LGBM_STABLE) if kind == 'LightGBM'
            else make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=seed)))


def run_classic(kind, Xtr, ytr, Xte, yte, seed):
    """LightGBM or RBF-SVM on the centre-pixel spectrum."""
    clf = classic(kind, seed)
    t0 = time.time()
    clf.fit(centre(Xtr), ytr)
    return dict(model=kind, train_time=time.time() - t0, **metrics(yte, clf.predict(centre(Xte))))


def shap_bands(Xtr, ytr, Xte, yte, name, log_name):
    """LightGBM exact TreeSHAP (pred_contrib) on the test set: mean |SHAP| per band and class -> shap_<name>.csv."""
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, **LGBM_STABLE).fit(centre(Xtr), ytr)
    k, b = len(clf.classes_), Xte.shape[-1]
    c = clf.predict(centre(Xte), pred_contrib=True).reshape(len(Xte), k, b + 1)[:, :, :b]  # drop bias column
    imp = np.abs(c).mean(0)
    pd.DataFrame(imp.T, columns=[f'class_{i}' for i in range(k)]).assign(band=range(b), total=imp.sum(0)) \
        .to_csv(f'{OUT}/shap_{name}.csv', index=False)
    log(log_name, dict(dataset=name, n_test=len(yte), top_bands=np.argsort(-imp.sum(0))[:10].tolist(),
                       **metrics(yte, clf.predict(centre(Xte)))))


def transfer(src, dst):
    """Copy every source tensor whose name and shape match, except the head `fc`.
    Positional embeddings are linearly resampled along the band/token axis.
    Returns the copied names (kept frozen during the first fine-tuning epochs)."""
    s, d, copied = src.state_dict(), dst.state_dict(), []
    for k, v in s.items():
        if k.startswith('fc.') or k not in d:
            continue
        if k == 'pos_embed' and v.shape != d[k].shape:
            v = F.interpolate(v.transpose(1, 2), size=d[k].shape[1], mode='linear', align_corners=False).transpose(1, 2)
        if v.shape == d[k].shape:
            d[k] = v.detach().clone()
            copied.append(k)
    dst.load_state_dict(d)
    return copied


# ---------------------------------------------------------------- logging
def log(name, row):
    """Append one result row (JSON lines: rows may have different keys) and print it."""
    with open(f'{OUT}/{name}.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)


def finish(name, **extra):
    """Write the CSV for this run and a manifest with versions and protocol."""
    pd.read_json(f'{OUT}/{name}.jsonl', lines=True).to_csv(f'{OUT}/{name}.csv', index=False)
    json.dump(dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                   scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lgb.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', smoke=SMOKE,
                   seeds=SEEDS, splits=SPLITS, epochs=EPOCHS, patience=PATIENCE, lr=LR, wd=WD, batch=BATCH,
                   freeze_epochs=FREEZE_EPOCHS, **extra),
              open(f'{OUT}/manifest_{name}.json', 'w'), indent=1)
    print(f'== {name} done', flush=True)
