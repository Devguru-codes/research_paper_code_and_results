from r15_long import *
for sc in ['IndianPines', 'PaviaU']:  # grouped-band transformer is small: CPU kernel
    run_disjoint(sc, nets=['SpectralFormer'], lightgbm=False, part='_long_sf')
