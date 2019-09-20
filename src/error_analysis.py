import pickle


def compare(idx, field_n, preds_file, data_file='data_autoconcepts/semrel_pp200_pad7.p'):
    preds_h = open(preds_file, 'rb')
    golds, preds = pickle.load(preds_h)
    data_h = open(data_file, 'rb')
    data = pickle.load(data_h)

    trp_test = data[field_n]

    with open(preds_file + ".error_analysis.txt", "w") as f_out:
        for c, (gold, pred) in enumerate(zip(golds, preds)):
            if gold != pred:
                f_out.write("gold: {}".format(idx[gold]) + "\n")
                f_out.write("pred: {}".format(idx[pred]) + "\n")
                f_out.write(trp_test[c]["sen"] + "\n")
                f_out.write("\n")


idx_trp = {1: 'tr IMPROVES p', 2: 'tr WORSENS p', 3: 'tr CAUSES p', 4: 'tr ADMINISTERED FOR p',
           5: 'tr AVOIDED BECAUSE OF p', 0: 'None'}
compare(idx_trp, field_n=3, preds_file="result/trp_img200_nhu100_pad7.p")

idx_tep = {1:'te REVEALS p', 2:'te CARRIED OUT FOR p', 0:'None'}
compare(idx_tep, field_n=4, preds_file="result/tep_img200_nhu100_pad7.p")

idx_pp = {1: 'p RELATED TO p', 0: 'None'}
compare(idx_pp, field_n=5, preds_file="result/pp_img200_nhu100_pad7.p")