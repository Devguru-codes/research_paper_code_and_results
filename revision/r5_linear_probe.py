from hsi_common import *
# R5: linear probe. Pretrain on Local soil, freeze, train one linear layer on Indian Pines features.
# Fixes the old run: the ViT checkpoint now actually loads, and HybridSN is probed at conv3 (a real linear probe).

split = local_file_split()
Ltr, ytr = load_local_files(split['train'])
Lva, yva = load_local_files(split['val'])
X, y = load_indian_pines()
NETS = ['HybridSN', 'SpectralFormer', 'SpatialViT']

pre = {}
src = make_loaders([(Ltr, ytr, None), (Lva, yva, None), (Lva, yva, None)], 0)
for name in NETS:
    set_seed(0)
    pre[name], info = train(MODELS[name](168, 3), *src[:2])
    log('r5', dict(exp='R5', stage='pretrain_local', model=name, **info, **test(pre[name], src[1])))
del src, Ltr, Lva


def feats(model, loader):
    f = getattr(model, 'probe_features', model.features)
    model.to(DEVICE).eval()
    with torch.no_grad():  # one pass: the train loader reshuffles on every pass
        fs, ys = zip(*[(f(x.to(DEVICE)), yb) for x, yb in loader])
    return torch.cat(fs), torch.cat(ys).to(DEVICE)


def linear_probe(tr, va, te, seed, steps=2000, patience=50):
    """Full-batch logistic regression on frozen, standardised features; early stop on val loss."""
    (Ftr, ytr_), (Fva, yva_), (Fte, yte_) = tr, va, te
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    torch.manual_seed(seed)
    head = nn.Linear(Ftr.shape[1], 16).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WD)
    best, state, bad = float('inf'), None, 0
    for _ in range(20 if SMOKE else steps):
        opt.zero_grad()
        F.cross_entropy(head((Ftr - mu) / sd), ytr_).backward()
        opt.step()
        with torch.no_grad():
            vl = F.cross_entropy(head((Fva - mu) / sd), yva_).item()
        if vl < best:
            best, state, bad = vl, copy.deepcopy(head.state_dict()), 0
        elif (bad := bad + 1) >= patience:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        return metrics(yte_.cpu().numpy(), head((Fte - mu) / sd).argmax(1).cpu().numpy())


for split_ in SPLITS[:3]:
    for seed in SEEDS:
        tr, va, te = ip_split(y, split_, seed)
        loaders = make_loaders([(X, y, tr), (X, y, va), (X, y, te)], seed)
        for name in NETS:
            for init in ['scratch', 'transfer']:
                set_seed(seed)
                m = MODELS[name](200, 16)
                copied = transfer(pre[name], m) if init == 'transfer' else []
                res = linear_probe(*(feats(m, l) for l in loaders), seed)
                log('r5', dict(exp='R5', dataset='IndianPines', split=split_, seed=seed, model=name, init=init,
                               n_copied=len(copied), **res))
                free(m)
finish('r5')
