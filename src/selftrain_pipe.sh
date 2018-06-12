#!/usr/bin/env bash
# thresholding value is provided as input $1

rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/{rel,preds}/*
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainFalse
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.9
rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/*
cp -r /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/preds/* /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainTrue
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.9

rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/{rel,preds}/*
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainFalse
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs1 -n_train10000000 -selftrainTrue -st_threshold0.8
rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/*
cp -r /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/preds/* /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainTrue
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.8

rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/{rel,preds}/*
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainFalse
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs1 -n_train10000000 -selftrainTrue -st_threshold0.7
rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/*
cp -r /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/preds/* /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainTrue
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.7

rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/{rel,preds}/*
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainFalse
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs1 -n_train10000000 -selftrainTrue -st_threshold0.6
rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/*
cp -r /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/preds/* /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainTrue
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.6

rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/{rel,preds}/*
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainFalse
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs1 -n_train10000000 -selftrainTrue -st_threshold0.5
rm /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/*
cp -r /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/preds/* /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/i2b2-2010/concept_assertion_relation_training_data/partners/unannotated/rel/
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py -selftrainTrue -include_selftrainTrue
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -tep -n_runs10 -n_train10000000 -selftrainTrue -st_threshold0.5


