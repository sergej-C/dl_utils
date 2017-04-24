

from sklearn.metrics import log_loss
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn import ensemble
from sklearn.model_selection import StratifiedKFold as KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import extract_predictions
from extract_predictions import do_clip, do_clip_min_max

def evaluate_pred(pred_prob, y_,  doclip=0,  clip_min=.15, clip_max=.95):
    
    pred_prob = np.nan_to_num(pred_prob)

    if doclip:
        if clip_min is not None and clip_max is not None:
            pred_prob = do_clip_min_max(pred_prob, clip_min, clip_max)
        else:
            pred_prob = do_clip(pred_prob, 0.85)

    if y_ is not None:
        le = LabelBinarizer() 
        y_prob = le.fit_transform(y_)

        print log_loss(y_prob, pred_prob)
        cm = confusion_matrix(y_, np.argmax(pred_prob, axis=1))
        print cm
    stat_pred(pred_prob)
    return pred_prob

def evaluate(clf_,x_,y_, doclip=0,  clip_min=.15, clip_max=.95):
    
    pred_prob = clf_.predict_proba(x_)
    pred_prob = np.nan_to_num(pred_prob)
    
    if doclip:
        if clip_min is not None and clip_max is not None:
            pred_prob = do_clip_min_max(pred_prob, clip_min, clip_max)
        else:
            pred_prob = do_clip(pred_prob, 0.85)


    if y_ is not None:
        le = LabelBinarizer()    
        y_prob = le.fit_transform(y_)
        print log_loss(y_prob, pred_prob)
        cm = confusion_matrix(y_, np.argmax(pred_prob, axis=1))
        print cm
    stat_pred(pred_prob)
    return pred_prob

def stat_pred(pred):
    print '='*60
    print 'shape: {}'.format(pred.shape)
    print 'max: {}'.format(np.max(pred))
    print 'min: {}'.format(np.min(pred))
    print 'mean: {}'.format(np.mean(pred))
    max_per_row = np.argmax(pred, axis=1)
    from collections import Counter
    print Counter(max_per_row)
    

def evaluate_ens_on_groups(clfs, grouped_feats, grouped_labels=None, all_together=True, scale_=True, doclip=0,  clip_min=.15, clip_max=.95, minus_one_label=False, clss_num=-1):

    """
    evaluate  the classifier on each group
    and compute stats on stacked preds

    the model must have predict_proba method
    if hasn't .classes_ property set clss_num with the number of classes
    """ 
   
    test=False 
    if grouped_labels is None:
        test=True
    all_size=0
    for g in grouped_feats:
        all_size+= len(grouped_feats[g])

    minus = 1 if minus_one_label is True else 0
    all_y = np.empty((all_size, ), dtype=np.int)

    if clss_num != -1:
        all_prob = np.empty((all_size, clss_num))
    else:
        all_prob = np.empty((all_size, clfs[0].classes_.shape[0]))
    last_index = 0
    for g in grouped_feats:
        
        X_ = grouped_feats[g] if scale_ is False else scale(grouped_feats[g])
        len_f = len(grouped_feats[g])
        if test is False:
            y_ = np.asarray(grouped_labels[g]).astype(int)-minus
            all_y[last_index:last_index+len_f] = y_
        else:
            y_ = None
        all_prob[last_index:last_index+len_f, ...] = evaluate(clfs[g], X_, y_,  doclip=doclip,  clip_min=clip_min, clip_max=clip_max)
        last_index = len_f

    if len(grouped_feats)>1:
        if test is True:
            all_y = None
        all_prob = evaluate_pred(all_prob, all_y, doclip=doclip,  clip_min=clip_min, clip_max=clip_max)

    return all_prob


def get_random_pred_by_groups(grouped_labels, nb_class,  doclip=0,  clip_min=.15, clip_max=.95, test=True):
        
    all_size=0
    for g in grouped_labels:
        all_size+= len(grouped_labels[g])

    random_prob = np.random.randint(150,999, (all_size,nb_class))/1000.
    if test:
        return random_prob

    all_y = np.empty((all_size, ), dtype=np.int)
    last_index = 0
    for g in grouped_labels:
        y_ = np.asarray(grouped_labels[g]).astype(np.int)-1
        all_y[last_index:last_index+len(y_)] = y_
        last_index = len(y_)

    return evaluate_pred(random_prob, all_y.astype(int), doclip=doclip,  clip_min=clip_min, clip_max=clip_max)


def merge_test_pred(clfs, hogs_by_groups, scale=True):
    pass

def train_cls_onfeat_by_group(clfs, hogs_by_groups, seed_=1):

    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=test_ratio,
                                                    random_state=seed_)

       

def cv_estimate_logloss(clf, X_train, y_train,n_splits=3):
    cv = KFold(n_splits=n_splits)
    
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train, y_train):            
        evaluate_ens_on_groups({0: clf}, {0: X_train[test]}, {0: y_train[test]})

def train_cls_kfold(nfold, clf_, X, Y, clss, save=True, save_prefix=None, out_p=None, seed=1):
    skf = KFold(n_splits=nfold, random_state=seed)
    y_pred = np.zeros_like(Y)
    y_pred_prob = np.empty((len(Y), len(clss)))
    clfs = []
    le = LabelBinarizer()
    clf=clf_
    
    for train, test in skf.split(X, Y):
        
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]        
        #clf=deepcopy(clf_)
        
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)    
        y_pred_prob[test,...] = clf.predict_proba(X_test)[:]
        evaluate_ens_on_groups({0: clf}, {0: X_test}, {0:y_test})
        clfs.append(clf)
    
    print ('='*50)
    print classification_report(Y, y_pred, target_names=clss)
    y_true = le.fit_transform(Y)

    print("logloss",log_loss(y_true, y_pred_prob))

    scores_in_fold = cross_val_score(clf, X, Y, cv=skf, n_jobs=2)
    best_clf_idx = np.argmax(scores_in_fold)
    best_clf = clfs[best_clf_idx]
    #clfs = None
    pred = best_clf.predict(X_test)
    pred_p = best_clf.predict_proba(X_test)
    
    # todo - ensemble 

    if save:
        save_clf ='cls_pred.pkl'
        
        if save_prefix:
            save_clf = save_prefix + '_' + save_clf

        if out_p:
            save_clf = out_p + '/' + save_clf

        dump({'pred':pred, 'pred_p':pred_p, 'clf':best_clf, 'img_keys':img_keys}, save_clf) 

    return best_clf, clfs, pred, pred_p
 
