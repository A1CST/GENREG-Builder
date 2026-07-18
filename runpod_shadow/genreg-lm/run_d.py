import radial_kid
radial_kid.make_next_d(n_train=50000, n_test=10000)
radial_kid.stage_d()                          # warm=kid_modelC3.json
print('D DONE', flush=True)
