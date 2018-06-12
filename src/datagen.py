""" yluo - 05/01/2016 creation
Call cnn_preprocess functions to generate data files ready to used by Seg-CNN
"""
import re
import sys

import cnn_preprocess as cp

img_w = 200
pad = 7
#fnwem = '/nas/corpora/accumulate/clicr/embeddings/b1654752-6f92-11e7-ac2f-901b0e5592c8/embeddings' # 200d
fnwem = "/nfshome/corpora/accumulate/clinicalembs/mimic-emb-cbow-200.txt"
fndata='../data/semrel_pp%s_pad%s.p' % (img_w, pad)
scale_fac=100

selftrain = sys.argv[1]
mo = re.search('-selftrain(\w+)', selftrain)
if mo:
    selftrain = eval(mo.group(1))
else:
    print('example: -selftrainTrue')
    sys.exit(1)
assert selftrain in {True, False}

include_selftrain = sys.argv[2]
mo = re.search('-include_selftrain(\w+)', include_selftrain)
if mo:
    include_selftrain = eval(mo.group(1))
else:
    print('example: -include_selftrainTrue')
    sys.exit(1)
assert include_selftrain in {True, False}

mem, hwoov, hwid = cp.embed_train_test_dev(fnwem, fndata=fndata, padlen=pad, scale_fac=scale_fac, selftrain=selftrain, include_selftrain=include_selftrain)

