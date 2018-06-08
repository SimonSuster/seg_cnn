""" yluo - 05/01/2016 creation
Preprocess i2b2/VA relations to generate data files ready to used by Seg-CNN
"""
from file_util import get_file_list

import numpy as np
import cPickle
from collections import defaultdict
import re, os
from nltk.stem import WordNetLemmatizer

import data_util as du
from background_knowledge import compatibility, read_drugbank, semclass
from lexica import lexica


trp_rel = ['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP']
tep_rel = ['TeRP', 'TeCP']
pp_rel = ['PIP']
dosages = ['mg', 'bid', 'prn', 'qd', 'po', 'tid', 'qhs', 'qid', 'qod']

drug_to_id, id_to_indication, id_to_adr = read_drugbank()


def load_stoplist(fn):
    ## in case you want to use stopword list, not really needed
    h = {}
    f = open(fn)
    for ln in f:
        swd = ln.rstrip(' \n')
        if not h.has_key(swd):
            h[swd] = 0
        h[swd] += 1
    return h;

def include_wd(wd):
    ans = re.search(r'[A-Za-z]', wd) != None or wd in '/:;()[]{}-+?'
    ans &= not 'year-old' in wd and 'y/o' != wd and 'yo' != wd and 'y.o.' != wd
    ans &= not '&' in wd and not '**' in wd
    ans &= re.search(r'[0-9]', wd) == None 
    ans &= re.search(r'^[^A-Za-z].+', wd) == None # cannot start with nonletter
    return ans;

def clean_wds(wdsin, hstop={}, strict=True):
    wdsout = []
    for wd in wdsin:
        if not strict or include_wd(wd):
            wd = re.sub('"', '', wd)
            wd = du.removeNonAscii(wd)
            if not hstop.has_key(wd):
                if strict and ('-' in wd and '-' != wd):
                    for swd in wd.split('-'):
                        if len(swd)>1:
                            wdsout.append(swd)
                else: 
                    if strict and len(wd)>1:
                        wd = re.sub('[^A-Za-z]*$', '', wd)
                    if len(wd)>0:
                        wdsout.append(wd)
    return wdsout;

def load_mask(words, iid):
    hmask = {}
    wid = 0
    st = -1; end = -1
    for word in words:
        if re.search(r'^\*\*.*\[', word):
            st = wid
        if st != -1 and re.search(r'\]', word): # can start and end on the same word
            end = wid
            hmask[(st, end+1)] = 1
            if st == -1:
                print('mask err in %s' % (iid))
            st = -1
        wid += 1
    if st != -1:
        print('mask unfinished in %s' % (iid))
        hmask[(st, wid+1)] = 1
    return hmask;


def inmask(i, hmask):
    res = False
    for mask in hmask:
        mst = mask[0]
        mend = mask[1]
        if mst <= i and i < mend:
            res = True
            break
    return res;

def mask_concept(words, st, end, hmask, hproblem, htreatment, htest):
    mwds = [words[i] for i in range(st, end) if not inmask(i, hproblem) and not inmask(i, htreatment) and not inmask(i, htest) and not inmask(i, hmask)]
    return mwds;

def fmask(words, st, end, hmask={}, hproblem={}, htreatment={}, htest={}, mask=False, skip_concept=False):
    mwds = []; out = True
    for i in range(st, end):
        if not (mask and inmask(i, hmask)):
            if skip_concept and inmask(i, hproblem):
                if out:
                    mwds += ['problem']
                out = False
            elif skip_concept and inmask(i, htreatment):
                if out:
                    mwds += ['treatment']
                out = False
            elif skip_concept and inmask(i, htest):
                if out:
                    mwds += ['test']
                out = False
            else:
                mwds += [words[i]]
                out = True
    return mwds;

def concept_type(cs, ce, iid, hproblem, htreatment, htest):
    if (cs, ce) in hproblem:
        ct = 'problem'
    elif (cs, ce) in htreatment:
        ct = 'treatment'
    elif (cs, ce) in htest:
        ct = 'test'
    else:
        print('should not be here %s %s in %s' % (cs, ce, iid))
    return ct;

def markup_sen(words, c1s, c1e, c2s, c2e, iid, hproblem, htreatment, htest):
    c1t = concept_type(c1s, c1e, iid, hproblem, htreatment, htest)
    c2t = concept_type(c2s, c2e, iid, hproblem, htreatment, htest)
    if c1s < c2s:
        mwords = words[:c1s] + ['['] + words[c1s:c1e] + [']%s' % (c1t)] + words[c1e:c2s] + ['['] + words[c2s:c2e] + [']%s' % (c2t)] + words[c2e:]
    else:
        mwords = words[:c2s] + ['['] + words[c2s:c2e] + [']%s' % (c2t)] + words[c2e:c1s] + ['['] + words[c1s:c1e] + [']%s' % (c1t)] + words[c1e:]
    return mwords;
        
def build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, rel='None', padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, mask=False, skip_concept=False, lemmatizer=None, scale_fac=1):
    words = sen.split()
    hmask = load_mask(words, iid)
    c1 = clean_wds(fmask(words, c1s, c1e, hmask), hstop, strict=False)
    c2 = clean_wds(fmask(words, c2s, c2e, hmask), hstop, strict=False)
    c1t = concept_type(c1s, c1e, iid, hproblem, htreatment, htest)
    c2t = concept_type(c2s, c2e, iid, hproblem, htreatment, htest)
    cts = '_'.join(sorted((c1t, c2t)))
    prec_end = min(c1s, c2s); succ_start = max(c1e,c2e)
    prec = clean_wds(fmask(words, 0, prec_end, hmask, hproblem, htreatment, htest, mask, skip_concept), hstop) 
    succ = clean_wds(fmask(words, succ_start, len(words), hmask, hproblem, htreatment, htest, mask, skip_concept), hstop) 
    mid = clean_wds(fmask(words, min(c1e,c2e), max(c1s,c2s), hmask, hproblem, htreatment, htest, mask, skip_concept), hstop)
    hlen[cts]['c1'] = max(hlen[cts]['c1'], len(c1))
    hlen[cts]['c2'] = max(hlen[cts]['c2'], len(c2))
    hlen[cts]['mid'] = max(hlen[cts]['mid'], len(mid))

    compa1 = compatibility(c1, c2, c1t, c2t, rel, drug_to_id, id_to_indication, scale_fac*0)  # *0: ignore feature
    compa2 = compatibility(c1, c2, c1t, c2t, rel, drug_to_id, id_to_adr, scale_fac*0)  # *0: ignore feature

    if rel.lower().startswith("tr"):
        lexicon = lexica["trp"]
        txt = mid
    elif rel.lower().startswith("te"):
        lexicon = lexica["tep"]
        txt = prec + mid + succ
    else:
        lexicon = None
        txt = None

    semclass1, semclass2, semclass3, semclass4, semclass5 = semclass(txt, lexicon, rel, lemmatizer, scale_fac)
    #semclass1, semclass2, semclass3, semclass4, semclass5 = [scale_fac]*5

    if c1s < c2s:
        c1 = prec[-padlen:] + c1 + mid[:padlen]
        c2 = mid[-padlen:] + c2 + succ[:padlen]
    else:
        c1 = mid[-padlen:] + c1 + succ[:padlen]
        c2 = prec[-padlen:] + c2 + mid[:padlen]

    prec = prec[-padlen:]
    succ = succ[:padlen]
    hlen[cts]['prec'] = max(hlen[cts]['prec'], len(prec))
    hlen[cts]['succ'] = max(hlen[cts]['succ'], len(succ))

    mwords = markup_sen(words, c1s, c1e, c2s, c2e, iid, hproblem, htreatment, htest)
    datum  = {'iid':iid,
              'rel':rel, 
              'c1': c1,
              'c2': c2,
              'prec': prec,
              'succ': succ,
              'mid': mid,
              'sen': ' '.join(mwords),
              'compa1': compa1,
              'compa2': compa2,
              'semclass1': semclass1,
              'semclass2': semclass2,
              'semclass3': semclass3,
              'semclass4': semclass4,
              'semclass5': semclass5}
    return datum;

def add_none_rel(fn, hpair, sens, rels, vocab, hlen, mask=False, mid_lmax=None, padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, skip_concept=False, scale_fac=1, lemmatizer=None):
    for senid in hpair:
        for con_pair in hpair[senid]:
            c1s = con_pair[0][0]
            c1e = con_pair[0][1]
            c2s = con_pair[1][0]
            c2e = con_pair[1][1]
            sen = sens[senid].lower()
            iid = '%s:%s (%d,%d) (%d,%d)' % (fn, senid, c1s, c1e, c2s, c2e)
            midlen = max(c1s,c2s) - min(c1e,c2e)
            if mid_lmax != None and midlen > mid_lmax:
                continue

            datum = build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept, scale_fac=scale_fac, lemmatizer=lemmatizer)

            if datum != None:
                rels.append(datum)

            
def load_con(fncon, htrp, htep, hpp, selftrain):
    hproblem = defaultdict(list)
    htreatment = defaultdict(list)
    htest = defaultdict(list)
    lc = 0
    if selftrain:
        sent_cid_to_str = {}
    with open(fncon, 'r') as f:
        lc += 1
        for ln in f:
            ln = ln.rstrip(' \n')
            mo = re.search(r'c="(.*?)" (\d+):(\d+) \d+:(\d+)\|\|t="(.*?)"', ln)
            if mo:
                senid = int(mo.group(2))-1 # start with 1 in annotation
                cs = int(mo.group(3))
                ce = int(mo.group(4))+1
                ctype = mo.group(5)
                if ctype == "problem":
                    hproblem[senid].append((cs, ce))
                elif ctype == "treatment":
                    htreatment[senid].append((cs, ce))
                elif ctype == "test":
                    htest[senid].append((cs,ce))
                else:
                    print('unrecognized ctype %s at %d in %s' % (ctype, lc, fncon))
                if selftrain:
                    sent_cid_to_str["{}_{}_{}".format(senid, cs, ce)] = mo.group(1)

        # sort the concepts according to positions
        for senid in htreatment:
            htreatment[senid] = sorted(htreatment[senid], key=lambda x: x[0])
        for senid in htest:
            htest[senid] = sorted(htest[senid], key=lambda x: x[0])
        for senid in hproblem:
            hproblem[senid] = sorted(hproblem[senid], key=lambda x: x[0])

        # genereate all possible pairs for treatment problem
        for senid in htreatment:
            for tr in htreatment[senid]:
                for p in hproblem[senid]:
                    htrp[senid][(tr, p)] = 1
                    
        # test problem pair
        for senid in htest:
            for te in htest[senid]:
                for p in hproblem[senid]:
                    htep[senid][(te, p)] = 1
                    
        # problem pair
        for senid in hproblem:
            for i in range(len(hproblem[senid])-1):
                for j in range(i+1,len(hproblem[senid])):
                    p1 = hproblem[senid][i]
                    p2 = hproblem[senid][j]
                    if p1[0] < p2[0]:
                        hpp[senid][(p1, p2)] = 1
                    else:
                        print('collapsed %s and %s in %d in %s' % (p1, p2, senid, fncon))

        # if self-training, create fake rel files based on concept pairs
        if selftrain:
            senids = set(htrp.keys()) | set(htep.keys()) | set(hpp.keys())
            fnrel = re.sub("/concept/", "/rel/", fncon)
            fnrel = re.sub("\.con", ".rel", fnrel)
            with open(fnrel, "w") as fnrel_out:
                for senid in senids:
                    if senid in htrp:
                        (c1s, c1e), (c2s, c2e) = htrp[senid].keys()[0]  # taking one concept pair is enough, rest will become None
                        # c="bipap" 71:13 71:13||r="TrAP"||c="copd" 71:8 71:8
                        # relation name does not matter
                        inst = "c=\"{e1_str}\" {s_num}:{c1s} {s_num}:{c1e}||r=\"TrAP\"||c=\"{e2_str}\" {s_num}:{c2s} {s_num}:{c2e}".format(
                            e1_str=sent_cid_to_str["{}_{}_{}".format(senid, c1s, c1e)],
                            e2_str=sent_cid_to_str["{}_{}_{}".format(senid, c2s, c2e)],
                            c1s=c1s,
                            c1e=c1e-1,
                            c2s=c2s,
                            c2e=c2e-1,
                            s_num=senid+1)
                        fnrel_out.write(inst+"\n")
                    if senid in htep:
                        (c1s, c1e), (c2s, c2e) = htep[senid].keys()[0]  # taking one concept pair is enough, rest will become None
                        # c="bipap" 71:13 71:13||r="TrAP"||c="copd" 71:8 71:8
                        # relation name does not matter
                        inst = "c=\"{e1_str}\" {s_num}:{c1s} {s_num}:{c1e}||r=\"TeRP\"||c=\"{e2_str}\" {s_num}:{c2s} {s_num}:{c2e}".format(
                            e1_str=sent_cid_to_str["{}_{}_{}".format(senid, c1s, c1e)],
                            e2_str=sent_cid_to_str["{}_{}_{}".format(senid, c2s, c2e)],
                            c1s=c1s,
                            c1e=c1e-1,
                            c2s=c2s,
                            c2e=c2e-1,
                            s_num=senid+1)
                        fnrel_out.write(inst+"\n")
                    if senid in hpp:
                        (c1s, c1e), (c2s, c2e) = hpp[senid].keys()[0]  # taking one concept pair is enough, rest will become None
                        # c="bipap" 71:13 71:13||r="TrAP"||c="copd" 71:8 71:8
                        # relation name does not matter
                        inst = "c=\"{e1_str}\" {s_num}:{c1s} {s_num}:{c1e}||r=\"PIP\"||c=\"{e2_str}\" {s_num}:{c2s} {s_num}:{c2e}".format(
                            e1_str=sent_cid_to_str["{}_{}_{}".format(senid, c1s, c1e)],
                            e2_str=sent_cid_to_str["{}_{}_{}".format(senid, c2s, c2e)],
                            c1s=c1s,
                            c1e=c1e-1,
                            c2s=c2s,
                            c2e=c2e-1,
                            s_num=senid+1)
                        fnrel_out.write(inst+"\n")

    return (hproblem, htreatment, htest)

def load_rel(fnrel, sens, htrp, htep, hpp, vocab, hlen, trp_data, tep_data, pp_data, mask=False, padlen=0, hstop={}, hproblem={}, htreatment={}, htest={}, skip_concept=False, pip_reorder=False, scale_fac=1):
    sen_seen = {}
    fnroot = re.sub(r'^.*/', '', fnrel)


    wordnet_lemmatizer = WordNetLemmatizer()  # semclass
    with open(fnrel, 'r') as f:
        lc = 0; trp_mid_lmax = 0; tep_mid_lmax = 0; pp_mid_lmax = 0
        for ln in f:
            ln = ln.rstrip(' \n')
            lc += 1
            mo = re.search(r'c=".*?" (\d+):(\d+) \d+:(\d+)\|\|r="(.*?)"\|\|c=".*?" \d+:(\d+) \d+:(\d+)', ln)
            if mo:
                senid = int(mo.group(1))-1 # start with 1 in annotation # line number
                c1s = int(mo.group(2)) # start c1
                c1e = int(mo.group(3))+1 # end c1
                rel = mo.group(4)  # relation name
                c2s = int(mo.group(5))
                c2e = int(mo.group(6))+1
                sen = sens[senid].lower()
                if not sen_seen.has_key(senid):
                    words = sen.split()
                    for word in set(clean_wds(words, hstop)):
                        vocab[word] += 1
                    sen_seen[senid] = 1
                iid = '%s:%s (%d,%d) (%d,%d)' % (fnroot, senid, c1s, c1e, c2s, c2e)
                if pip_reorder and rel == 'PIP':
                    datum = build_inst(iid, min(c1s,c2s), min(c1e,c2e), max(c1s,c2s), max(c1e,c2e), sen, vocab, hlen, rel, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept, lemmatizer=wordnet_lemmatizer, scale_fac=scale_fac)
                else:
                    datum = build_inst(iid, c1s, c1e, c2s, c2e, sen, vocab, hlen, rel, padlen=padlen, hstop=hstop, hproblem=hproblem[senid], htreatment=htreatment[senid], htest=htest[senid], mask=mask, skip_concept=skip_concept, lemmatizer=wordnet_lemmatizer, scale_fac=scale_fac)
                midlen = max(c1s,c2s) - min(c1e,c2e)
                con_pair = ((c1s, c1e), (c2s, c2e))
                con_pair2 = ((c2s, c2e), (c1s, c1e))
                if rel in trp_rel:
                    trp_data.append(datum)
                    if not htrp[senid].has_key(con_pair):
                        print('no trp pair %s in %s' % (con_pair,iid))  # so what?
                    htrp[senid].pop(con_pair, None)  # to make easier adding none rels later
                    trp_mid_lmax = max(trp_mid_lmax, midlen)
                elif rel in tep_rel:
                    tep_data.append(datum)
                    if not htep[senid].has_key(con_pair):
                        print('no tep pair %s in %s' % (con_pair,iid))
                    htep[senid].pop(con_pair, None)
                    tep_mid_lmax = max(tep_mid_lmax, midlen)
                elif rel in pp_rel:
                    pp_data.append(datum)
                    if not hpp[senid].has_key(con_pair) and not hpp[senid].has_key(con_pair2):
                        print('no pp pair %s in %s' % (con_pair,iid))
                    hpp[senid].pop(con_pair, None)
                    hpp[senid].pop(con_pair2, None)
                    pp_mid_lmax = max(pp_mid_lmax, midlen)
                else:
                    print('unrecognized rel %s' % (rel))
            else:
                print('non-matching line %d in %s' % (lc, fnrel))
        # updates trp_data as a side effect with new None data instances
        add_none_rel(fnroot, htrp, sens, trp_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept, scale_fac=scale_fac, lemmatizer=wordnet_lemmatizer)
        add_none_rel(fnroot, htep, sens, tep_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept, scale_fac=scale_fac, lemmatizer=wordnet_lemmatizer)
        add_none_rel(fnroot, hpp, sens, pp_data, vocab, hlen, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept, scale_fac=scale_fac, lemmatizer=wordnet_lemmatizer)
    return;

                            
def build_data(dn, vocab, hlen, mask=False, padlen=0, hstop={}, skip_concept=False, pip_reorder=False, scale_fac=1, selftrain=False):
    """
    Loads data 
    """
    trp_data = [] # problem treatment
    tep_data = [] # problem test
    pp_data = [] # problem problem
    dntxt = '%s/txt' % (dn)
    dnrel = '%s/rel' % (dn)
    dncon = '%s/concept' % (dn)
    fc = 0

    for fntxt in os.listdir(dntxt):
        htrp = defaultdict(dict)
        htep = defaultdict(dict)
        hpp = defaultdict(dict)
        
        if not(re.search(r'.txt$', fntxt)):
            continue
        fc += 1
        fnrel = re.sub('.txt', '.rel', fntxt)
        fncon = re.sub('.txt', '.con', fntxt)
        sens = [] # sentences to load
        with open('%s/%s' % (dntxt, fntxt), "r") as f:
            for ln in f:
                ln = ln.rstrip(' \n')
                sens.append(ln)

        # hproblem: contains all gold problem concepts for a file/doc
        # htrp: contains all possible pairs for a file/doc, updated in load_con as a side effect
        (hproblem, htreatment, htest) = load_con('%s/%s' % (dncon, fncon), htrp, htep, hpp, selftrain)

        # side effect: updates trp/tep/pp_data; adds None rels:
        load_rel('%s/%s' % (dnrel, fnrel), sens, htrp, htep, hpp, vocab, hlen, trp_data, tep_data, pp_data, mask=mask, padlen=padlen, hstop=hstop, hproblem=hproblem, htreatment=htreatment, htest=htest, skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac)

    print(fc)

    return trp_data, tep_data, pp_data

def build_train_test_dev(cdn='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/', hlen = defaultdict(lambda:defaultdict(float)), padlen=0, fnstop=None, skip_concept=False, pip_reorder=False, scale_fac=1, selftrain=False):
    hstop = {}; vocab = defaultdict(float)
    if fnstop != None:
        hstop=load_stoplist(fnstop)

    # each var from build_data() is [datum, ...], where
    # datum={'iid':iid,
    #        'rel':rel,
    #        'c1': c1,
    #        'c2': c2,
    #        'prec': prec,
    #        'succ': succ,
    #        'mid': mid,
    #        'sen': ' '.join(mwords),
    #        'compa': compa_c1c2}

    trp_beth_tr, tep_beth_tr, pp_beth_tr = build_data('%s/concept_assertion_relation_training_data/beth' % (cdn), vocab, hlen, mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac)
    print('beth_tr %d' % (len(trp_beth_tr)))
    print('beth_te %d' % (len(tep_beth_tr)))
    print('beth_p %d' % (len(pp_beth_tr)))

    trp_partners_tr, tep_partners_tr, pp_partners_tr = build_data('%s/concept_assertion_relation_training_data/partners' % (cdn), vocab, hlen, padlen=padlen, hstop=hstop, skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac)
    print('partners_tr %d' % (len(trp_partners_tr)))
    print('partners_te %d' % (len(tep_partners_tr)))
    print('partners_p %d' % (len(pp_partners_tr)))

    trp_fromtest_tr, tep_fromtest_tr, pp_fromtest_tr = build_data('%s/concept_assertion_relation_training_data/from_test' % (cdn), vocab,
                                                      hlen, mask=True, padlen=padlen, hstop=hstop,
                                                      skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac)
    print('fromtest_tr %d' % (len(trp_fromtest_tr)))
    print('fromtest_te %d' % (len(tep_fromtest_tr)))
    print('fromtest_p %d' % (len(pp_fromtest_tr)))

    trp_rel_te, tep_rel_te, pp_rel_te = build_data('%s/reference_standard_for_test_data' % (cdn), vocab, hlen,
                                                   mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept,
                                                   pip_reorder=pip_reorder, scale_fac=scale_fac)
    print('test_tr %d' % (len(trp_rel_te)))
    print('test_te %d' % (len(tep_rel_te)))
    print('test_p %d' % (len(pp_rel_te)))

    trp_rel_de, tep_rel_de, pp_rel_de = build_data('%s/concept_assertion_relation_dev_data' % (cdn), vocab, hlen,
                                                   mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept,
                                                   pip_reorder=pip_reorder, scale_fac=scale_fac)
    print('dev_tr %d' % (len(trp_rel_de)))
    print('dev_te %d' % (len(tep_rel_de)))
    print('dev_p %d' % (len(pp_rel_de)))

    trp_rel_tr = trp_beth_tr + trp_partners_tr + trp_fromtest_tr
    tep_rel_tr = tep_beth_tr + tep_partners_tr + tep_fromtest_tr
    pp_rel_tr = pp_beth_tr + pp_partners_tr + pp_fromtest_tr

    if selftrain == True:
        # selftraining on unannotated data
        trp_rel_st, tep_rel_st, pp_rel_st = build_data(
            '%s/concept_assertion_relation_training_data/partners/unannotated/' % (cdn), vocab, hlen,
            mask=True, padlen=padlen, hstop=hstop, skip_concept=skip_concept,
            pip_reorder=pip_reorder, scale_fac=scale_fac, selftrain=True)
        print('st_tr %d' % (len(trp_rel_st)))
        print('st_te %d' % (len(tep_rel_st)))
        print('st_p %d' % (len(pp_rel_st)))

        print("including selftrain to train")
        trp_rel_tr += trp_rel_st
        tep_rel_tr += tep_rel_st
        pp_rel_tr += pp_rel_st

        return trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, hlen, trp_rel_st, tep_rel_st, pp_rel_st
    else:
        return trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, hlen


def sample_fn_test(cdn='/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/', n_to_train=102, n_to_dev=68):
    """
    Sample n_to_* filenames from test and move them to train/dev.
    :param cdn: test data directory
    :param n_to_train: n of filenames to sample from test and move to train
    :param n_to_dev: n of filenames to sample from test and move to dev
    """
    np.random.seed(1234)
    fns = sorted(get_file_list(cdn + "reference_standard_for_test_data/txt/"))
    assert len(fns) == 256

    new_train_dir = cdn + "concept_assertion_relation_training_data/from_test/"
    if not os.path.exists(new_train_dir):
        os.makedirs(new_train_dir + "txt/")
        os.makedirs(new_train_dir + "ast/")
        os.makedirs(new_train_dir + "concept/")
        os.makedirs(new_train_dir + "rel/")

    new_dev_dir = cdn + "concept_assertion_relation_dev_data/"
    if not os.path.exists(new_dev_dir):
        os.makedirs(new_dev_dir + "txt/")
        os.makedirs(new_dev_dir + "ast/")
        os.makedirs(new_dev_dir + "concept/")
        os.makedirs(new_dev_dir + "rel/")

    sample_fns = np.random.choice(fns, n_to_train + n_to_dev, replace=False)
    print("moving to train")
    for fn in sample_fns[:n_to_train]:
        base_fn = os.path.basename(fn)
        print("moving {}".format(base_fn))
        os.rename(cdn + "reference_standard_for_test_data/txt/{}".format(base_fn), new_train_dir + "txt/" + base_fn)
        os.rename(cdn + "reference_standard_for_test_data/ast/{}".format(os.path.splitext(base_fn)[0] + ".ast"), new_train_dir + "ast/" + os.path.splitext(base_fn)[0] + ".ast")
        os.rename(cdn + "reference_standard_for_test_data/concept/{}".format(os.path.splitext(base_fn)[0] + ".con"), new_train_dir + "concept/" + os.path.splitext(base_fn)[0] + ".con")
        os.rename(cdn + "reference_standard_for_test_data/rel/{}".format(os.path.splitext(base_fn)[0] + ".rel"), new_train_dir + "rel/" + os.path.splitext(base_fn)[0] + ".rel")

    print("moving to dev")
    for fn in sample_fns[n_to_train:]:
        base_fn = os.path.basename(fn)
        print("moving {}".format(base_fn))
        os.rename(cdn + "reference_standard_for_test_data/txt/{}".format(base_fn), new_dev_dir + "txt/" + base_fn)
        os.rename(cdn + "reference_standard_for_test_data/ast/{}".format(os.path.splitext(base_fn)[0] + ".ast"), new_dev_dir + "ast/" + os.path.splitext(base_fn)[0] + ".ast")
        os.rename(cdn + "reference_standard_for_test_data/concept/{}".format(os.path.splitext(base_fn)[0] + ".con"), new_dev_dir + "concept/" + os.path.splitext(base_fn)[0] + ".con")
        os.rename(cdn + "reference_standard_for_test_data/rel/{}".format(os.path.splitext(base_fn)[0] + ".rel"), new_dev_dir + "rel/" + os.path.splitext(base_fn)[0] + ".rel")


def embed_train_test_dev(fnem, fnwid='../data/vocab.txt', fndata='../data/semrel.p', padlen=0, fnstop=None, skip_concept=True, pip_reorder=False, binEmb=False, scale_fac=1, selftrain=False):
    if selftrain:
        trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, hlen, trp_rel_st, tep_rel_st, pp_rel_st = build_train_test_dev(padlen=padlen, fnstop=fnstop, skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac, selftrain=selftrain)
    else:
        trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, hlen = build_train_test_dev(padlen=padlen, fnstop=fnstop, skip_concept=skip_concept, pip_reorder=pip_reorder, scale_fac=scale_fac, selftrain=selftrain)
    fwid = open(fnwid, 'w')
    for wd in sorted(vocab.keys()):
        if vocab[wd] >= 1:
            fwid.write('%s\n' % (wd))
        else:
            vocab.pop(wd, None)
    fwid.close()
    if binEmb:
        mem, hwoov, hwid = du.load_bin_vec(fnem, fnwid)
    else:
        mem, hwoov, hwid = du.indexEmbedding(fnem, fnwid)
    mem = mem.astype('float32')
    # the saved data are lists of relation dicts, with keys c1, c2 ,etc.
    if selftrain:
        cPickle.dump([trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, dict(hlen), mem, hwoov, hwid, trp_rel_st, tep_rel_st, pp_rel_st], open(fndata, "wb"))
    else:
        cPickle.dump([trp_rel_tr, tep_rel_tr, pp_rel_tr, trp_rel_te, tep_rel_te, pp_rel_te, trp_rel_de, tep_rel_de, pp_rel_de, vocab, dict(hlen), mem, hwoov, hwid], open(fndata, "wb"))
    print "dataset created!"
    return mem, hwoov, hwid

def clamp_to_con(d_clamp, d_out):
    """
    Convert clamp output that we obtained from unannotated txt files for self-training to the 'con' format of i2b2-2010.

    """
    for f_clamp in get_file_list(d_clamp):
        sent_ids = []
        tok_ids = []
        insts = []
        with open(f_clamp) as in_f:
            ls = in_f.readlines()

        for l in ls:
            l = l.strip().split("\t")
            if l[0].startswith("Sentence"):
                sent_ids.append((int(l[1]), int(l[2])))
            elif l[0].startswith("Token"):
                tok_ids.append((int(l[1]), int(l[2])))

        for l in ls:
            l = l.strip().split("\t")
            if l[0] == "NamedEntity":
                e_s_i, e_e_i = l[1:3]
                e_s_i = int(e_s_i)  # start char idx of the entity
                e_e_i = int(e_e_i)  # end char idx of the entity
                vals = {i.split("=")[0]: i.split("=")[1] for i in l[3:]}  # semantic, assertion, cui, ne; val can also be missing
                for c, (s_s_i, s_e_i) in enumerate(sent_ids, 1):
                    if s_s_i <= e_s_i <= s_e_i:
                        assert e_e_i <= s_e_i
                        s_num = c
                        break
                n_prev_toks = 0
                for c, (t_s_i, t_e_i) in enumerate(tok_ids):
                    if t_s_i >= s_s_i:
                        n_prev_toks += 1
                        if t_s_i <= e_s_i <= t_e_i:
                            t_s_num = n_prev_toks-1
                        if t_s_i <= e_e_i <= t_e_i:
                            t_e_num = n_prev_toks-1
                inst = "c=\"{e_str}\" {s_num}:{t_s_num} {s_num}:{t_e_num}||t=\"{sem_cat}\"".format(
                    e_str=vals["ne"],
                    s_num=s_num,
                    t_s_num=t_s_num,
                    t_e_num=t_e_num,
                    sem_cat=vals["semantic"]
                )
                insts.append(inst)
            else:
                break
        f_con = d_out + os.path.basename(os.path.splitext(f_clamp)[0]) + ".con"
        with open(f_con, "w") as f_out:
            for i in insts:
                f_out.write(i + "\n")

def empty_rel(ref, d_out):
    """
    Generate empty rel files into d_out for all txt files (ref)
    """
    if not os.path.exists(d_out):
        os.makedirs(d_out)
    for f in get_file_list(ref):
        f_rel = d_out + os.path.basename(os.path.splitext(f)[0]) + ".rel"
        with open(f_rel, "w") as f_out:
            f_out.write("")


if __name__ == "__main__":
    #clamp_to_con(d_clamp="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated_clamp/",
    #             d_out = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/concept/")
    empty_rel(ref="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/txt/",
              d_out = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/")