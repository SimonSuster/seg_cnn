# Segment CNN for relation classification with external features
This is the code for the paper [Revisiting neural relation classification in clinical notes with external information](http://aclweb.org/anthology/W18-5603), by Simon Šuster, Madhumita Sushil and Walter Daelemans, published at the Workshop on Health Text Mining and Information Analysis (LOUHI), EMNLP, in 2018.

### Requirements
Code is written in Python (2.7) and requires Theano (0.9).


### Data Preprocessing
To process the raw data for TrP (treatment-problem) relations, run
```
python datagen.py trp
```
To process the data for other relation types, use `tep` and `pp`.

This is a wrapper code calling `cnn_preprocess.embed_train_test_dev()` with arguments specifying word embedding dimensions (e.g., 200), padding length (e.g., 7), the path to pretrained word2vec vectors and the scaling factor for the external features. This version of the code will use only the semantic classes as the external features. If you'd like to use Drugbank compatibility features, remove the multiplication by zero in `cnn_preprocess.build_inst()`. The addition of PMI features is currently available as a separate branch, `pmi_feats`.
This will create a pickle object (e.g., `semrel_pp200_pad7.p`) in the directory 'data/', which contains the dataset
with the right components to be used by `cnn_semrel.py`.



### Training
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -trp -n_runs5
```
By using `-static`, we keep the word vectors fixed throughout the training; `-word2vec` uses the pretrained word vectors specified in `datagen.py`, with a dimensionality of 200 (`-img_w`);
 `-l1_nhu` specifies the number of hidden units in the first layer; `-pad` specifies the padding length; `trp` the type of relations to train the classifier for; and `-n_runs` specifies the number of runs (i.e. classifiers) to perform (in the end, the results are averaged and the confidence intervals are reported).

### Example output
`mif` is micro-averaged f-measure, reported after each epoch for the development set (`mif_de`). Whenever an improvement is observed on the development set, the system is evaluated on the test set as well (`mif`), and the confusion matrix is printed. At the end of training, the average scores are reported across different runs.
```
...
epoch: 30, training time: 7.27 secs, train perf: 99.81 %, dev_mipre: 68.96 %, dev_mirec: 78.50 %, dev_mif: 73.42 %
msg: trp img_w: 200, l1_nhu: 100, pad: 7, mipre: 0.6913907284768211, mirec: 0.6170212765957447, mif: 0.6520924422236102, mipre_de: 0.7495741056218058, mirec_de: 0.7333333333333333, mif_de: 0.7413647851727043
Avg test confusion matrix:
	None	TrIP	TrWP	TrCP	TrAP	TrNAP
None	979.7	3.4	0.95	17.65	82.25	7.05
TrIP	10.7	16.6	0.05	1.2	11.45	0.0
TrWP	8.55	3.2	5.2	6.2	18.45	0.4
TrCP	43.9	0.1	0.0	70.8	29.9	0.3
TrAP	134.75	2.9	0.3	5.55	427.0	3.5
TrNAP	11.9	0.0	0.0	2.95	24.8	5.35
Avg mipre: 0.70313033187; CI95: (0.67312518752191, 0.733135476217524)
Avg mirec: 0.620508274232; CI95: (0.5714029545365837, 0.6696135939267733)
Avg mif: 0.658660254418; CI95: (0.6407514877872911, 0.6765690210485313)
Avg mipre_de: 0.754513942631; CI95: (0.721579013154559, 0.7874488721072679)
Avg mirec_de: 0.726329312904; CI95: (0.6915740972615753, 0.7610845285457187)
Avg mif_de: 0.739791034524; CI95: (0.7291962605701604, 0.7503858084787663)
```


