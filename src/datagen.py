""" yluo - 05/01/2016 creation
Call cnn_preprocess functions to generate data files ready to used by Seg-CNN
"""
import sys
import cnn_preprocess as cp

#print("Setting scale factor to 0: ignoring all external features.")
#scale_fac=0

scale_fac=100  # will use only semantic classes

if sys.argv[1] == "trp":
    print("Creating data with TRP optimal params.")
    img_w = 200
    pad = 7
    fnwem = '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/clinical_embs/mimic_pubmed/mimic_pubmed_lower_shuf200.txt' # 200d
elif sys.argv[1] == "tep":
    print("Creating data with TEP optimal params.")
    img_w = 500
    pad = 4
    fnwem = '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/clinical_embs/mimic_pubmed/mimic_pubmed_lower_shuf500.txt' # 200d
elif sys.argv[1] == "pp":
    print("Creating data with PP optimal params.")
    img_w = 400
    pad = 10
    fnwem = '/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/clinical_embs/mimic_pubmed/mimic_pubmed_lower_shuf400.txt' # 200d

fndata='../data_autoconcepts/semrel_pp%s_pad%s.p' % (img_w, pad)

mem, hwoov, hwid = cp.embed_train_test_dev(fnwem, fndata=fndata, padlen=pad, scale_fac=scale_fac)

