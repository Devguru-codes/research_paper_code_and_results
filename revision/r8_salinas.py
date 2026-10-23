from r8_disjoint import *
run_disjoint('Salinas', nets=['ViT', 'ConvStemViT', 'HybridSN'], lightgbm=False, part='_gpu')
