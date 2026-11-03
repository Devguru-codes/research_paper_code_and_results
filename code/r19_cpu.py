from r19_random import *
for sc in ['PaviaU', 'Salinas']:  # LightGBM and the small grouped-band transformer on CPU
    run_random(sc, ['SpectralFormer'], lightgbm=True, part='_cpu')
