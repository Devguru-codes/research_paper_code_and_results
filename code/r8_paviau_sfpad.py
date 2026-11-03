from r8_disjoint import *
# R18 (CPU): grouped-band transformer on Pavia U with the 103 bands zero-padded to 104 (the earlier run dropped 3 bands).
run_disjoint('PaviaU', nets=['SpectralFormer'], lightgbm=False, part='_sfpad')
