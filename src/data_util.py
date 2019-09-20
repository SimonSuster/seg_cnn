""" yluo - 09/25/2014 creation
common data reading, writing, and transforming utilities. 
"""
import zipfile

from cnn_semrel import convert_write_cm_R, htrp_rel, htep_rel, hpp_rel

__author__= """Yuan Luo (yuan.hypnos.luo@gmail.com)"""
__revision__="0.5"

import numpy as np
from operator import itemgetter
from datetime import datetime
from itertools import chain
import os
import re
import csv
import copy
import sys

def removeNonAscii(s):
    """
    s should be a utf-8 encoded string
    """
    return "".join(i for i in s if ord(i)<128)

def load_bin_vec(fname, fnwid, var = 0.25, rand_oov=False):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    np.random.seed(1289)
    fwid = open(fnwid, 'r')
    hwid = {}; cnt = 1; # word hash id starts from 1
    while 1:
        ln = fwid.readline()
        if not ln:
            break
        ln = ln.rstrip(" \n")
        wd = ln
        hwid[wd] = cnt
        cnt += 1
    fwid.close()
    hwoov = copy.deepcopy(hwid)
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        mem = np.zeros((cnt, layer1_size))
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if hwid.has_key(word):
                wid = hwid[word]
                mem[wid] = np.fromstring(f.read(binary_len), dtype='float32')
                hwoov.pop(word, None)
            else:
                f.read(binary_len)
    if rand_oov:
        for wd in hwoov:
            wid = hwid[wd]
            mem[wid,:] = np.random.uniform(-var,var,vsize)
    return (mem, hwoov, hwid);

def indexEmbedding(fnem, fnwid, var = 0.25, rand_oov=False):
    np.random.seed(1289)
    fwid = open(fnwid, 'r')
    hwid = {}; cnt = 1; # word hash id starts from 1
    while 1:
        ln = fwid.readline()
        if not ln:
            break
        ln = ln.rstrip(" \n")
        wd = ln
        hwid[wd] = cnt
        cnt += 1
    fwid.close()
    fem = open(fnem, 'r')
    femr = csv.reader(fem, delimiter=' ', quoting=csv.QUOTE_NONE)
    lc = 0; vsize = 0
    hwoov = copy.deepcopy(hwid)
    for row in femr:
        if lc == 0:
            vsize = int(row[1])
            mem = np.zeros((cnt, vsize)) # mem[0,:] should be 0
        else:
            wd = row[0]
            if hwid.has_key(wd):
                wid = hwid[wd]
                mem[wid,:] = map(float, row[1:-1])
                hwoov.pop(wd, None)
        lc += 1
    fem.close()
    # generate random vector for out of vocabulary words ? treat as 0
    if rand_oov:
        for wd in hwoov:
            wid = hwid[wd]
            mem[wid,:] = np.random.uniform(-var,var,vsize)
    return (mem, hwoov, hwid);

def get_contrast_CMs():
    # segcnn
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_trp", h_rel=htrp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_trp_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_tep",
                       h_rel=htep_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_tep_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_pp",
                       h_rel=hpp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/segcnn_pp_")

    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_trp",
                       h_rel=htrp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_trp_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_tep",
                       h_rel=htep_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_tep_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_pp",
                       h_rel=hpp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/compa_pp_")

    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/semclass_trp", h_rel=htrp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/semclass_trp_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/semclass_tep",
                       h_rel=htep_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/semclass_tep_")

    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_trp",
                       h_rel=htrp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_trp_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_tep",
                       h_rel=htep_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_tep_")
    convert_write_cm_R(result_f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_pp",
                       h_rel=hpp_rel, dn="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/result/pmi_pp_")


if __name__== "__main__":
    get_contrast_CMs()

