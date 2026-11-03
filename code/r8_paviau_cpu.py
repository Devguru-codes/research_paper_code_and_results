from r8_disjoint import *
# CPU part: LightGBM + grouped-band transformer (cheap on CPU); GPU part runs the other three.
run_disjoint('PaviaU', nets=['SpectralFormer'], part='_cpu')
