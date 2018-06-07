""" yluo - 05/01/2016 creation
Call cnn_preprocess functions to generate data files ready to used by Seg-CNN
"""

import cnn_preprocess as cp

img_w = 200
pad = 7
fnwem = '/nas/corpora/accumulate/clicr/embeddings/b1654752-6f92-11e7-ac2f-901b0e5592c8/embeddings' # 200d
fndata='../data/semrel_pp%s_pad%s.p' % (img_w, pad)
scale_fac=100

mem, hwoov, hwid = cp.embed_train_test_dev(fnwem, fndata=fndata, padlen=pad, scale_fac=scale_fac, selftrain=True)

