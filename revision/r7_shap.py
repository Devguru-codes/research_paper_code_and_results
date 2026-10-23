from hsi_common import *
# R7: LightGBM SHAP (exact TreeSHAP via pred_contrib) on the held-out test sets, per band and class.
# Superseded by the SHAP output of R8 (Indian Pines, disjoint split) and R11 (Local, sample-level splits).

split = local_file_split()
Ltr, ytr = load_local_files(split['train'])
Lte, yte = load_local_files(split['test'])
shap_bands(Ltr, ytr, Lte, yte, 'local', 'r7')
X, y = load_indian_pines()
tr, va, te = ip_split(y, 0.05, 42)
shap_bands(X[tr], y[tr], X[te], y[te], 'indianpines_5pct', 'r7')
finish('r7')
