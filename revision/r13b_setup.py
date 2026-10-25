# R13b: rerun of R13's water-band-removed variant for the grouped-band transformer only, now that its last band
# group is zero-padded (R13 dropped the last 3 of 119 bands). Classical models are unaffected and not rerun.
R13_NETS, R13_CLASSIC, R13_VARIANTS, R13_LOG = ['SpectralFormer'], [], ['nowater'], 'r13b'
