from r8_disjoint import *
run_disjoint('PaviaU', nets=['ViT', 'ConvStemViT', 'HybridSN'], lightgbm=False, part='_gpu')
