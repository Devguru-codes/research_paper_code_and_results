from r8_disjoint import *
# R15: longer training on the disjoint splits. 58-85% of the 50-epoch deep runs stopped at the epoch cap, so every
# deep model is rerun with up to 100 epochs (early stopping and splits unchanged). Entry scripts r15_<scene>_<part>.py.
EPOCHS = 2 if SMOKE else 100
