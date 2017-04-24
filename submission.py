import cPickle
import pandas as pd

import os, sys
DL_UTILS_PATH=os.environ.get('DL_UTILS_PATH')
sys.path.append(DL_UTILS_PATH)

CSN_PATH='/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/my_proj/studies/practicals/cs231n_1'
sys.path.append(CSN_PATH)
#from dlc_utils import *
from cs231n.features import *
from glob import glob
from cv2 import imread, cvtColor
import cv2

# where there is my train folder (from original datasets downloaded from Kaggle)
# base folder in which i save fish data folder
KAGGLE_CONP_DATA=os.environ.get('KAGGLE_PATH')

# some utils
sys.path.append(KAGGLE_CONP_DATA+'/dl_utils')
import data_utils as du
import dlc_utils

PATH_FISH_ORIG=KAGGLE_CONP_DATA+'/fish'
TRAIN_ORIG_PATH=PATH_FISH_ORIG+'/train'

import extract_predictions
import faster_rcnn_utils
import data_struct_utils
from data_struct_utils import *
from extract_predictions import *
from io_utils import *

path_to_save = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/data_for_classifier/test_stg1'
back_f = [
    'backgrounds_dets_stg1_ens_vgg_2_19_21_y4.pkl'
]

prepath_clf_pred = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/clf_results/'
prepath_dl_results = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/dl_results/'

saved_dfs = [
            prepath_dl_results+'subm_sequential_4_ftvgg_trainorig.gz',
            prepath_dl_results+'subm_sequential_4_conv_512_on_cropped_contain.gz'
    ]

clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
lbl_clss = ('NoF', 'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
class_to_ind = du.array_to_ind(clss)
lbl_clss_to_ind = du.array_to_ind(lbl_clss)
nclasses = len(clss)+1
out_path = '/home/sergio/Scrivania/new_subm/dl_on_cropped'
out_path_clf = '/home/sergio/Scrivania/new_subm/clf_on_cropped'

from collections import Counter

def merge_and_create_submission(
        df_name,
        prepath_df=prepath_dl_results,
        idx_back=0,
    ):

    backgrounds = read(path_to_save+back_f[idx_back])
    
    df_p = prepath_df+df_name

    df = pd.DataFrame.from_csv(pd_p)

    prefix_save = df_name

    dl_rows = {}
    trace_max_cls = []
    for i,v in enumerate(pd_oncropped_contain.values):

        r1=df.values[i]
        img = df.index[i]
        true_cls = np.argmax(r1)
        max_conf = r1[true_cls]
        trace_max_cls.append(true_cls)
        #print str(true_cls)+' '+str(max_conf)
        r= (create_class_row(nclasses, max_clas_pos=true_cls+1, max_confidence=0.85, \
                         append_true_class=False, true_class=None))
        dl_rows[img]=np.squeeze(r)

    print Counter(trace_max_cls)

    dl_back = {}
    for img_b in backgrounds:
        r = create_background_row(nclasses, append_true_class=False, true_class=None)
        dl_back[img_b+'.jpg'] = np.squeeze(r)


    dict_final = dict(dl_back, **dl_rows);
    print len(dict_final)


    create_submission(dict_final, lbl_clss, out_path, prefix=prefix_save, doclip=True)


def merge_and_create_clf_submission(
    pred_f,
    prefix,
    idx_back=0
    ):


    preds_dict = read(prepath_clf_pred+pred_f)

    #{'pred':pred, 'pred_p':pred_p, 'clf':best_clf, 'img_keys':img_keys}
    preds = preds_dict['pred']
    img_keys = preds_dict['imgs']

    print Counter(preds)


    backgrounds = read(path_to_save+back_f[idx_back])
    prefix_save = pred_f

    dl_rows = {}
    trace_max_cls = []
    for i,v in enumerate(preds):

        img = img_keys[i]
        true_cls = v
        trace_max_cls.append(true_cls)
        #print str(true_cls)+' '+str(max_conf)
        r = (create_class_row(nclasses, max_clas_pos=true_cls+1, max_confidence=0.85, \
                         append_true_class=False, true_class=None))
        dl_rows[img]=np.squeeze(r)

    print Counter(trace_max_cls)

    dl_back = {}
    for img_b in backgrounds:
        r = create_background_row(nclasses, append_true_class=False, true_class=None)
        dl_back[img_b+'.jpg'] = np.squeeze(r)

    dict_final = dict(dl_back, **dl_rows);
    print len(dict_final)


    create_submission(dict_final, lbl_clss, out_path, prefix=prefix_save, doclip=True)

    return preds, img_keys, dict_final



