path_tars = '/home/sergio/Scrivania/all_pkl/'

import tarfile
import cPickle
from glob import glob
import data_utils as mu
from io_utils import *
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from xml_utils import *
from faster_rcnn_utils import *
from data_struct_utils import *

import cv2
import pandas as pd

import itertools
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

from easydict import EasyDict as edict


FASTER_RCNN_PATH=os.environ.get('PYFASTER_PATH')
FASTER_RCNN_TOOLS_PATH=FASTER_RCNN_PATH+'tools'
FASTER_RCNN_LIB_PATH=FASTER_RCNN_PATH+'lib'
CAFFE_PATH=FASTER_RCNN_PATH+'/caffe-fast-rcnn'
DL_UTILS_PATH=os.environ.get('DL_UTILS_PATH')

sys.path.append(FASTER_RCNN_PATH)
sys.path.append(FASTER_RCNN_TOOLS_PATH)
sys.path.append(FASTER_RCNN_LIB_PATH)
sys.path.append(DL_UTILS_PATH)

import _init_paths
from datasets.factory import get_imdb

from test_imdetect_2 import create_tar_info

fu = faster_rcnn_utils()
TYPE_IMDB='imdb'
TYPE_NO_IMDB='no_imdb'
# same info e detections structure but no gt
TYPE_IMDB_LIKE='imdb_like'
DEBUG=False
def dbg_print(txt):
    if DEBUG:
        print(txt)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def extract_from_tar_in_folder(path_tars, uid_only=None):

    path_tars = ensure_last_slash(path_tars)

    if uid_only is not None:
        path_tars += '*'+uid_only
    glob_tars = glob(path_tars + '*.tar.gz')
    dbg_print("search tar in {}".format(path_tars))

    if len(glob_tars)==0:
        print "error, no tar found"
    # error in saving... name
    info_pkl_name = 'info_evalutation'
    detec_pkl_name = 'detections'
    all_detections = {}
    inserted = 0
    for t in glob_tars:

        is_imdb = False
        info_found = False
        tar = tarfile.open(t)
        for m, member in enumerate(tar.getmembers()):
            if m == 0:
                base_name = member.name
                k_name = base_name + '_' + str(inserted)

            if member.name[-4:] == '.pkl':
                if is_imdb == False and \
                                member.name.split('/')[-1][:-4] == detec_pkl_name:
                    dbg_print("==>START[imdb] getting detections for {}".format(k_name))
                    # ok, is imdb eval, read detections
                    is_imdb = True
                    f = tar.extractfile(member)

                    if not all_detections.has_key(k_name):
                        all_detections[k_name] = {}
                    all_detections[k_name]['dets'] = (cPickle.load(f))
                    all_detections[k_name]['type'] = TYPE_IMDB

                # is imdb, gets its infos
                if info_found == False and is_imdb == True and \
                                member.name.split('/')[-1][:-4] == info_pkl_name:
                    dbg_print( "<==END[imdb]getting infos for {}".format(k_name) )
                    f = tar.extractfile(member)
                    all_detections[k_name]['infos'] = (cPickle.load(f))['info']
                    inserted += 1
                    info_found = True

                # not imdb, get its info_dets
                if info_found == False and is_imdb == False and \
                                member.name.split('/')[-1][:-4] == info_pkl_name:
                    dbg_print( "==>[NO_imdb]<== getting infos for {}".format(k_name))
                    f = tar.extractfile(member)
                    if not all_detections.has_key(k_name):
                        all_detections[k_name] = {}
                    det_info = (cPickle.load(f))
                    all_detections[k_name]['dets'] = det_info['dets']
                    all_detections[k_name]['infos'] = det_info['info']
                    all_detections[k_name]['type'] = TYPE_NO_IMDB

                    inserted += 1
                    info_found = True

        tar.close()

    return all_detections


def check_detections_imbd(detections, info, classes, project_name, type_,
                          imset_file=None, save_tp_fp_imgs=False, save_only_uid=None, thresh_bgr=.5, cached_prod=False, p_max=.85):

    class_to_ind = du.array_to_ind(classes)
    nof_tup = ('NoF',)
    lbl_class = nof_tup + classes

    # use for test also
    with_gt = False
    if type_=='imdb':
        with_gt=True

    if with_gt is False and imset_file is None:
        print "For detections without gt, imset_file must be specified"
        exit(1)

    if with_gt:
        imdb_name = info['imdb_name']
        uid = info['uid']
        ann_p, imset_file, path_imgs = get_paths_from_imdb(imdb_name, project_name)

    rows = {}
    bboxes = {}

    tracer_max_for_cls = {}
    for idx, im_name in enumerate(imset_file):

        tracer_det = []
        tracer_cls_det = []
        tracer_max_for_cls[im_name]={}
        tracer_true_cls = None

        if with_gt:

            # load im annotation for true class and bbox
            anno_xml, objs, tree = get_objs_from_anno_path(ann_p, class_to_ind, classes, im_name)

            mism, img_name, img_with_ext = get_and_check_img_name(tree, anno_xml)

            c_bbox, true_class = get_bbox_and_true_class(objs)

            if c_bbox is None:
                #pdb.set_trace()
                print "warning im {} has annotation without gt class, skip it".format(im_name)
                continue
            tracer_true_cls = []

        dbg_print("#######check img {}".format(im_name))
        cache_max_for_class = np.empty([len(classes)+1])
        for idc, c in enumerate(classes):

            idx_ifgt=idc+1
            
            #enumerate begin from 0...
            if with_gt:
                # in this case detections 0 is for background... skip it all empty
                dets = detections[idx_ifgt][idx]
            else:
                dets = detections[im_name][c]

            # indexes where confidence is > thresh
            inds = check_is_background_for_class(dets, c, thresh=thresh_bgr)

            # adjust classes with 0 background
            if with_gt:
                true_class_adj = true_class + 1

            # no detections have confidence > thresh, set detection as background
            if len(inds) == 0:
                pred_class=0
                dbg_print("is back for {}".format(c))
                #ma tanto inds ignorato, da rimuovere
                if len(dets)>0:
                    max_dets, max_conf = get_max_for_class(dets, inds)
                else:
                    max_conf = 0
                max_dets = []

            else:
                # prediction > thresh for current class
                # class_to_ind start from 0 but not include blackground class, so shift by one
                pred_class = class_to_ind[c]+1
            
                # get the best from the survivors [based on confidence value]
                max_dets, max_conf = get_max_for_class(dets, inds)

            if cached_prod:
                cache_max_for_class[idx_ifgt] = max_conf

            #print cache_max_for_class

            # cache it for choosing the best class detection between the remaining max detections over all classes
            tracer_det.append(max_dets)

            # index of the corresponding class
            tracer_cls_det.append(pred_class)

            # cache all max detections for this class for later analysis or use
            tracer_max_for_cls[im_name][class_to_ind[c]+1]=max_dets

            # add at the end of class prediction row
            if with_gt:
                tracer_true_cls.append(true_class_adj)

        # creating the row
        if not rows.has_key(im_name):
            rows[im_name] = []

        if cached_prod is False:
            cache_max_for_class=None
        else:
            #todo - bug to fix
            if(cache_max_for_class[0]>1):
                cache_max_for_class[0]=0

        # at the end get the max over all classes for current image
        nclasses = len(classes) + 1
        rows[im_name], bboxes[im_name] = (get_max_for_all_classes(tracer_det, tracer_cls_det, nclasses, tracer_true_cls, cache_conf=cache_max_for_class, p_max=p_max))

        # todo
        if with_gt:
            true_cls = rows[im_name][-1]
        else:
            true_class=-1
        if save_tp_fp_imgs:
            save_dets = True
            if save_only_uid is not None:
                if uid != save_only_uid:
                    save_dets = False
            if save_dets:
                save_detections(img_path=path_imgs + img_with_ext,
                                lbl_class=lbl_class,
                                true_class=true_cls,
                                pred_classes=tracer_cls_det,
                                max_dets=tracer_max_for_cls,
                                gt_bbox=c_bbox,
                                out_folder_name=info['uid'] + '_tp_tf_vis'
                                )

    return rows, tracer_max_for_cls, bboxes


def get_bbox_and_true_class(objs):
    # in pyfaster_rcnn imdb dataset,
    # for fish dataset
    # instances of the same class in each image [even if there are more than one bbox, get the first class]
    # background excluded, 0 is first non background class

    if len(objs['gt_classes'])==0:
        print "warning object with no gt class!"
        return None, None
        
    true_class = objs['gt_classes'][0]
    c_bbox = objs['boxes']
    return c_bbox, true_class


def get_objs_from_anno_path(ann_p, class_to_ind, classes, im_name):
    anno_xml = ann_p + im_name + '.xml'
    tree = parse_file(anno_xml)
    # if there are more than one object get the first and associate to its class
    objs = fu.parse_obj(tree, len(classes), \
                        class_to_ind, use_diff=True, minus_one=False)
    return anno_xml, objs, tree


def get_paths_from_imdb(imdb_name, project_name):
    # imdb_dets_valid[n_class][img]
    # get list of images from imdb
    imdb_root = fu.get_data_folder(project_name) + '/' + imdb_name
    imdb_imageset_file_path = imdb_root + '/ImageSets/Main/' + imdb_name + '.txt'
    path_imgs = imdb_root + '/JPEGImages/'
    ann_p = os.path.join(
        imdb_root,
        'Annotations/'
    )
    with open(imdb_imageset_file_path, 'r') as f:
        imset_file = f.readlines()

    imset_file = [x.strip() for x in imset_file]
    return ann_p, imset_file, path_imgs


def save_detections_old(img_path, true_class, pred_class,
                    dets, gt_bbox=None, thresh=0.5,
                    out_dir='/tmp', out_folder_name='tp_tf_visualizations',
                    save=True, figsize=(8,5)
                    ):
    """
    save images with bbox if exists, in separate folders

    -one folder for class
    --one folder for true positive
    --one folder for false positive
    :param img_path:
    :param true_class:
    :param pred_class:
    :param dets:
    :param gt_bbox:
    :return:
    """

    im = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]

    has_dets = True
    if dets is None or len(dets) == 0:
        dbg_print("len dets == 0 for class {}".format(true_class))
        has_dets = False

    if has_dets:
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            dbg_print("len inds == 0")
            has_dets = False

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, aspect='equal')

    if has_dets:
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )

            if gt_bbox is not None:
                for bb in gt_bbox:
                    plt.gca().add_patch(
                        plt.Rectangle((bb[0], bb[1]),
                                      bb[2] - bb[0],
                                      bb[3] - bb[1], fill=False,
                                      edgecolor='green', linewidth=2)
                    )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(pred_class, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}, true cls:{}').format(score, pred_class,
                                                                   thresh, true_class),
                    fontsize=13)
    else:
        ax.set_title(('p({} | box) >= {:.1f}, true cls:{}').format(pred_class,
                                                                   thresh, true_class),
                     fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    is_tp = pred_class == true_class
    is_fp = pred_class != true_class

    #mk base dir
    save_dir = out_dir+'/'+out_folder_name
    tp_dir = save_dir+'/'+str(true_class)+'/tp'

    if is_tp:
        path = tp_dir
        mu.mkdirs(tp_dir)
    else:
        fp_dir = save_dir + '/' + str(true_class) + '/fp/'+str(pred_class)
        mu.mkdirs(fp_dir)
        path = fp_dir

    if save:
        file_n = path + '/' + img_name
        plt.savefig(file_n)
        plt.clf()
        plt.close(fig)
    else:
        plt.show()


def save_detections(img_path, true_class, pred_classes, max_dets,
                    lbl_class=None, gt_bbox=None, thresh=0.5,
                    out_dir='/tmp', out_folder_name='tp_tf_visualizations',
                    save=True, figsize=(8,5)
                    ):
    """
    save images with bbox if exists, in separate folders

    -one folder for class
    --one folder for true positive
    --one folder for false positive
    :param img_path:
    :param true_class:
    :param pred_class:
    :param dets:
    :param gt_bbox:
    :return:
    """

    im = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]

    has_dets = True
    if dets is None or len(dets) == 0:
        dbg_print("len dets == 0 for class {}".format(true_class))
        has_dets = False

    if has_dets:
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            dbg_print("len inds == 0")
            has_dets = False

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, aspect='equal')

    if has_dets:
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )

            if gt_bbox is not None:
                for bb in gt_bbox:
                    plt.gca().add_patch(
                        plt.Rectangle((bb[0], bb[1]),
                                      bb[2] - bb[0],
                                      bb[3] - bb[1], fill=False,
                                      edgecolor='green', linewidth=2)
                    )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(pred_class, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}, true cls:{}').format(score, pred_class,
                                                                   thresh, true_class),
                    fontsize=13)
    else:
        ax.set_title(('p({} | box) >= {:.1f}, true cls:{}').format(pred_class,
                                                                   thresh, true_class),
                     fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    is_tp = pred_class == true_class
    is_fp = pred_class != true_class

    #mk base dir
    save_dir = out_dir+'/'+out_folder_name
    tp_dir = save_dir+'/'+str(true_class)+'/tp'

    if is_tp:
        path = tp_dir
        mu.mkdirs(tp_dir)
    else:
        fp_dir = save_dir + '/' + str(true_class) + '/fp/'+str(pred_class)
        mu.mkdirs(fp_dir)
        path = fp_dir

    if save:
        file_n = path + '/' + img_name
        plt.savefig(file_n)
        plt.clf()
        plt.close(fig)
    else:
        plt.show()


def get_k_by_uid(all_dets, uid):
    for k in all_dets.keys():
        info = all_dets[k]['infos']
        if info['uid'] != uid:
            continue
        else:
            return k
    print "k not found!"

def check_is_background_for_class(dets, cls, thresh=0.3):
    if dets is None or len(dets) == 0:
        dbg_print( "len dets == 0 for class {}".format(cls) )
        return []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        dbg_print("len inds == 0 for class {}".format(cls))
        return []
    return inds

def check_detections(detections, classes, path_back_imgs=None, thresh_bkg=.5):

    # for background
    true_class = 0
    rows = {}
    bboxes = {}
    for img_name in detections.keys():

        tracer_det = []
        tracer_cls_det = []
        tracer_true_cls = []

        dbg_print("#######check img {}".format(img_name))
        for idc, c in enumerate(classes):
            dets = detections[img_name][c]

            #if path_back_imgs is not None:
            #im = cv2.imread(path_back_imgs+'/'+img_name)
            #vis_back_detections(im,c, dets)
            inds=check_is_background_for_class(dets, c, thresh=thresh_bkg)

            if len(inds) == 0:
                dbg_print("is back for {}".format(c))
                tracer_det.append([])
                tracer_cls_det.append(0)
                tracer_true_cls.append(true_class)
            else:
                max_dets = get_max_for_class(dets, inds)
                tracer_det.append(max_dets)
                tracer_cls_det.append(idc+1)
                tracer_true_cls.append(true_class)
            if not rows.has_key(img_name):
                rows[img_name] = []

            # at the end get the max over all classes for current image
            nclasses = len(classes) + 1
            rows[img_name], bboxes[img_name] = (get_max_for_all_classes(tracer_det, tracer_cls_det, nclasses, tracer_true_cls))

    return rows, bboxes


def background_loss(rows_noimdb):
    """
    compute number of TP / num rows
    :param rows:
    :return:
    """
    correct = 0
    for preds in rows_noimdb:
        argmax = np.argmax(preds)
        if argmax == 0: correct += 1
    back_loss = float(correct / len(rows_noimdb))
    return 1-back_loss

def plot_detections_error_correct_for_cls(det_type, dets, cls, clss, class_to_ind, project_name, imdb_name=None):
    """
    det_type can be: correct, error_bkg, error_cls
    """
    if imdb_name is None:
        imdb_name = dets['infos']['imdb_name']
    ann_p, imset_file, path_imgs = get_paths_from_imdb(imdb_name, project_name)
    im2ind = du.array_to_ind(imset_file)

    c = cls
    if det_type=='error_cls':
      for cc in clss:
        det_by_type = dets['infos']['cls_error_tracer'][c][det_type][cc]
        print "view errors for {}".format(cc)
        for im in det_by_type:
            anno_xml, objs, tree = get_objs_from_anno_path(ann_p, class_to_ind, clss, im)
            c_bbox, true_class = get_bbox_and_true_class(objs)
            dets_im = dets['dets'][class_to_ind[c]+1][im2ind[im]]
            img = cv2.imread(path_imgs+im+'.jpg')
            vis_detections(img, c, dets_im, gt_bbox=c_bbox, true_class=c, block=True)

    else:
        det_by_type = dets['infos']['cls_error_tracer'][c][det_type]
        for im in det_by_type:
            anno_xml, objs, tree = get_objs_from_anno_path(ann_p, class_to_ind, clss, im)
            c_bbox, true_class = get_bbox_and_true_class(objs)
            img = cv2.imread(path_imgs+im+'.jpg')
            #pdb.set_trace()
            if det_type=='error_bkg':
                cv2.imshow('det background but was {}'.format(c), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                dets_im = dets['dets'][class_to_ind[c]+1][im2ind[im]]
                vis_detections(img, c, dets_im, gt_bbox=c_bbox, true_class=c, block=True)
        
    
def vis_detections(im, cls, dets, thresh=0.5,
                        gt_bbox=None, out_dir=None, prefix='',
                        true_class='?', save_to_folder=False, block=False):
    """Draw detected bounding boxes."""
    if dets is None or len(dets) == 0:
        dbg_print ("len dets == 0 for class {}".format(cls))
        return []

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        dbg_print ("len inds == 0")
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )

        if gt_bbox is not None:
            for bb in gt_bbox:
                plt.gca().add_patch(
                    plt.Rectangle((bb[0], bb[1]),
                                  bb[2] - bb[0],
                                  bb[3] - bb[1], fill=False,
                                  edgecolor='green', linewidth=2)
                )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}, true cls:{}').format(cls, cls,
                                                               thresh, true_class),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show(block=block)

    if save_to_folder:

        if out_dir is not None:
            save_dir = out_dir
        else:
            save_dir = './detections_plots'

        mu.mkdirs(save_dir)
        file_n = '/{}_{}_{:.3f}.jpg'.format(prefix, cls, score)
        plt.savefig(save_dir + file_n)

def get_max_for_class(dets, indkeep):
    """
    get best prediction by confidence for a class
    """
    confidences = (dets[:, -1])
    argmax_for_class = np.argmax(confidences)
    return dets[argmax_for_class, :], dets[argmax_for_class, -1]


def create_background_row(nclasses, def_pos=0, p_max=.85, append_true_class=False, true_class=None, cache_conf=None):
    """
    create the row for background detection
    p_max for def_pos and others 1-p_max/(nclasses-1)
    nclasses include background in count
    """

    if cache_conf is None:
        scores = np.ones((nclasses, 1)) * 0.015 #((1 - p_max) / (nclasses - 1))
        scores[def_pos] = p_max
    else:
        scores = cache_conf
        scores[def_pos] = p_max

        if append_true_class:
            scores = np.hstack((np.squeeze(scores), [true_class]))

    return scores


def create_class_row(nclasses, max_clas_pos, max_confidence, append_true_class=False, true_class=None, cache_conf=None):
    """
    create row for class prediction
    max p is confidence, the other class have 1-p/(nclasses-1)
    nclasses include background in count
    """
    p_max = max_confidence
    if cache_conf is None:
        scores = np.ones((nclasses, 1)) * 0.015 #((1 - p_max) / (nclasses - 1))
    else:
        scores = cache_conf
    scores[max_clas_pos] = p_max
    #pdb.set_trace()

    if append_true_class:
        scores = np.hstack((np.squeeze(scores), [true_class]))
    return scores


def get_max_for_all_classes(tracer_dets, tracer_classes, nclasses, tracer_true_cls=None, cache_conf=None, p_max=.85):
    """
    tracer_dets contains one row for each class that has a > thresh detection on the current image
    with the format [x1 y1 x2 y2 confidence]
    we want the row with max confidence value
    get the index of the row with the max confidence
    get the corresponding class position in tracer_classes and create
    a row with the max value at that position and the true class at last index
    if the tracer_dets is empty, there aren't detection for the current image,
    so return a row with max at 0 index [background]
    :param tracer_dets:
    :param tracer_classes:
    :param nclasses:
    :param tracer_true_cls:
    :return:
    """
    if tracer_true_cls is not None:
        append_true_class = True
        # true classes in tracer_true_cls are all equal
        tc = tracer_true_cls[0]
    else:
        append_true_class = False
        tc=None


    # is a background prediction
    if list_is_empty(tracer_dets):
        return create_background_row(nclasses, append_true_class=append_true_class, true_class=tc, cache_conf=cache_conf, p_max=p_max), []

    # get max prediction by confidence
    # remove empty rows. gets all confidences and traces indexes
    tracer_arr = np.empty(shape=(1,))
    tracer_det_idx = []
    tracer_bboxes = np.empty(shape=(1,4))
    inserted = False
    for ide, el in enumerate(tracer_dets):
        if not len(el) == 0:
            if inserted == False:
                tracer_arr[0] = el[-1]
                tracer_bboxes[0] = el[:4]
                inserted = True
            else:
                tracer_arr = np.vstack((tracer_arr, el[-1]))
                tracer_bboxes = np.vstack((tracer_bboxes, el[:4]))
            tracer_det_idx.append(ide)

    # max index by confidence
    argmax_conf = np.argmax(tracer_arr)

    # index of max value in tracer
    maxdets_index = tracer_det_idx[argmax_conf]

    # adjust to new classes position (0 - background)
    max_clas_pos = tracer_classes[maxdets_index]
    max_confidence = tracer_arr[argmax_conf]
    max_bbox = tracer_bboxes[argmax_conf,:]

    append_true_class = False
    true_cls = None
    if tracer_true_cls is not None:
        true_cls = tracer_true_cls[maxdets_index]
        append_true_class=True
    #print("create row max_clas_pos {} true_cls {} conf {}".format(max_clas_pos, true_cls, max_confidence))
    row = create_class_row(nclasses, max_clas_pos, max_confidence, append_true_class=append_true_class, true_class=true_cls, cache_conf=cache_conf)
    #print row
    return row, max_bbox

def my_multi_log_loss(y_pred,y_true, eps=1e-15):
    # bound predicted by max value
    y_pred = np.maximum(np.minimum(y_pred, 1-eps), eps)

    return -np.sum(y_true*np.log(y_pred))/len(y_pred)

def get_loss(rows, info, classes, compensate_0clss=True, compensate_0clss_max_conf=.85):

    # background class excluded, in classes there isn't
    pred = np.empty((len(rows.keys()), len(classes)+1 ))
    plus = 1 if compensate_0clss else 0
    y = np.empty((len(rows.keys())+plus, 1))

    for idm, im in enumerate(rows.keys()):
        row_img = rows[im]

        # eliminate last index == true class
        if len(row_img[:-1].shape)>1:
            pred[idm, ...] = np.squeeze(row_img[:-1], axis=1)
        else:
            pred[idm, ...] = np.squeeze(row_img[:-1])

        # get true class
        y[idm] = row_img[-1]

    if compensate_0clss:
        # y doesn't have background, add a row with worst case
        backpred = np.zeros((1, len(classes) + 1))+((1-compensate_0clss_max_conf)/len(classes))
        backpred[0][-1] = compensate_0clss_max_conf
        pred=np.vstack((pred, backpred))
        y[idm+1]=0

    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y.squeeze())

    dbg_print("true classes {}".format(lb.classes_))
    y_true = lb.transform(y)
    #print pred.shape, y.shape
    #loss = log_loss(y_pred=pred, y_true=y)
    loss = my_multi_log_loss(y_pred=pred, y_true=y_true)
    return pred, y, loss


def get_loss_nokey(rows, info, classes, compensate_0clss=True, compensate_0clss_max_conf=.85):

    # background class excluded, in classes there isn't
    pred = np.empty((rows.shape[0], len(classes)+1 ))
    plus = 1 if compensate_0clss else 0
    y = np.empty((rows.shape[0]+plus, 1))

    for idm,row_img in enumerate(rows):

        # eliminate last index == true class
        if len(row_img[:-1].shape)>1:
            pred[idm, ...] = np.squeeze(row_img[:-1], axis=1)
        else:
            pred[idm, ...] = np.squeeze(row_img[:-1])

        # get true class
        y[idm] = row_img[-1]

    if compensate_0clss:
        # y doesn't have background, add a row with worst case
        backpred = np.zeros((1, len(classes) + 1))+((1-compensate_0clss_max_conf)/len(classes))
        backpred[0][-1] = compensate_0clss_max_conf
        pred=np.vstack((pred, backpred))
        y[idm+1]=0

    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y.squeeze())

    dbg_print("true classes {}".format(lb.classes_))
    y_true = lb.transform(y)
    #print pred.shape, y.shape
    #loss = log_loss(y_pred=pred, y_true=y)
    loss = my_multi_log_loss(y_pred=pred, y_true=y_true)
    return pred, y, loss


def add_ap_record(aps, classes, record=None, mean=True):
    """
    ap values in aps
    and there must be the same length of classes
    and same order
    """
    if record is None:
        record = {}
    sum_ap = 0
    for cid, c in enumerate(classes):
        ap = aps[cid]
        name_ap = 'ap_' + c.lower()
        record[name_ap] = ap

        if mean:
            sum_ap += ap

    if mean:
        record['ap_mean'] = float(sum_ap / len(aps))

    return record


def add_record_info(info, record=None):
    if record is None:
        record = {}

    for k in info.keys():
        if k != 'aps' and k != 'aps_mean':
            record[k] = info[k]
    return record

def confusions_counts_with_img(rows, clsses):
    """
    for each class
    count how many errors as background 
          how many errors as other class
    and keep track of image name

    clsses background excluded
    """

    #
    # rows contains N rows and num classes + 1 (gt) elements
    # each key is the name of image
    N = len(rows.keys())

    #
    # each row has num classes prob and an extra column for gt (index of the correct class)
    # add the background column
    num_cols = len(clsses) + 2

    #
    # tracer
    tracer = {}
    for c in clsses:
        tracer[c] = {}
        tracer[c]['correct'] = []
        tracer[c]['error_bkg'] = []
        tracer[c]['error_cls'] = {}

        for cc in clsses:
            tracer[c]['error_cls'][cc] = []

    
    #
    # iterate over each image and assign to corrispondent tracer index
    for im in rows.keys():

        r = rows[im]
        pred_cls = np.argmax(r[:-1])
        gt_cls = r[-1]

        if  isinstance(gt_cls, np.float64):
            gt_cls = int(gt_cls)
        else:
            gt_cls = int(gt_cls[0])

        indx_cls = clsses[gt_cls-1]
        if pred_cls==gt_cls:
            tracer[indx_cls]['correct'].append(im)
        elif pred_cls==0:
            tracer[indx_cls]['error_bkg'].append(im)
        else:
            tracer[indx_cls]['error_cls'][clsses[pred_cls-1]].append(im)

    return tracer

import pdb
def create_row_for_df(all_detections, classes,
                      project_name, append=False,
                      df2append=None, get_uid_only=None,
                      save_vis=False, save_fig_uid_only=None,
                        thresh_bgr=.5, cached_prod=False, p_max=.85
                      ):

    """
    if get_uid_only is a valid uid then only dets for that case are taken
    and save_fig_uid_only is the same uid is save_vis is True

    if get_uid only is None and save_vis is True
    if save_fig_uid_only is a valid uid then only plots
     for that case are saved

    :param all_detections:
    :param classes:
    :param project_name:
    :param append:
    :param df2append:
    :param get_uid_only:
    :param save_vis:
    :param save_fig_uid_only:
    :return:
    """

    nof_tup = ('NoF',)
    lbl_clss = nof_tup + classes
    lbl_clss_to_ind = du.array_to_ind(lbl_clss)


    all_records = []
    for ide, k in enumerate(all_detections.keys()):

        dets = all_detections[k]['dets']
        info = all_detections[k]['infos']
        type_ = all_detections[k]['type']
        uid = info['uid']
        if not all_detections[k].has_key('rows'):
            all_detections[k]['rows'] = []

        if get_uid_only is not None:
            if save_vis is True:
                save_fig_uid_only=uid
            if uid != get_uid_only:
                continue
        rec = {}
        if type_ == TYPE_IMDB:
            rows, tracer_max_for_cls, max_bboxes = check_detections_imbd(dets, info, classes, project_name, type_,
                                         save_tp_fp_imgs=save_vis, save_only_uid=uid, thresh_bgr=thresh_bgr,
                                                                         cached_prod=cached_prod,p_max=p_max)
            pred, y, loss = get_loss(rows, info, classes)

            y = y.astype(int).squeeze()
            pred_for_cm = np.argmax(pred, axis=1)
            cm = confusion_matrix(y_true=y, y_pred=pred_for_cm)
            tracer = confusions_counts_with_img(rows, classes)
            # add also a counter for all prediction (not max only)
            mx = get_max_for_all_clsses(class_to_ind=lbl_clss_to_ind, info=info, lbl_clss=lbl_clss,
                                        max_for_clss=tracer_max_for_cls, rows=rows)


            add_ap_record(all_detections[k]['infos']['aps'], classes, rec)
            counter_items = np.sum(cm.T, axis=1)
            all_detections[k]['infos']['tot_pred'] = np.sum(cm)
            all_detections[k]['infos']['cls_counter'] = counter_items
            all_detections[k]['infos']['cls_error_tracer'] = tracer
            #all_detections[k]['cm'] = cm
            new_row = {'rows':rows,'max_bboxes':max_bboxes,'pred':pred, 'y':y, 'loss':loss, 'cm':cm, 'mx':mx, 'counter_items':counter_items}
            all_detections[k]['rows'].append(new_row)

        elif type_ == TYPE_NO_IMDB:
            #rows = check_detections(dets, classes)
            imset = all_detections[k]['dets'].keys()
            rows, tracer_max_for_cls, max_bboxes = check_detections_imbd(dets, info, classes, project_name, type_,
                                                             imset_file=imset,
                                                             save_tp_fp_imgs=save_vis, save_only_uid=uid,
                                                                         thresh_bgr=thresh_bgr, cached_prod=cached_prod,
                                                                                        p_max = p_max
                                                                         )

            # how many correct / total
            loss = background_loss(rows)

            new_row = {'rows':rows, loss:'loss', 'max_bboxes':max_bboxes}
            all_detections[k]['rows'].append(new_row)

            pred = get_ypred_from_all_dets(all_detections, ide)
            pred_counter, tot = count_dets4clss_for_noimdb_dets(pred, lbl_clss)
            all_detections[k]['infos']['cls_counter'] = pred_counter
            all_detections[k]['infos']['tot_pred'] = tot


        all_detections[k]['infos']['type'] = type_
        all_detections[k]['infos']['loss'] = loss

        rec = add_record_info(all_detections[k]['infos'], rec)
        #dbg_print("loss for {} = {}".format(info['imdb_name'] + '_' + info['iteration'] + '_' + info['model'] + info['uid'],
        #                                loss))

        all_records.append(rec)
    return all_detections, all_records

def test_extract(tar_folder, clss, project_name = 'fish', get_uid_only=None,
                 save_vis=False, save_fig_only_uid=None, thresh_bgr=.5, cached_prod=False, p_max=.85):

    # class without background

    # get detections from folder
    all_dets = extract_from_tar_in_folder(tar_folder, uid_only=get_uid_only)


    # alld = create_df(all_dets, clss, project_name)
    all_dets, all_recs = create_row_for_df(all_dets, clss, project_name,
                                           get_uid_only=get_uid_only,
                                           save_fig_uid_only=save_fig_only_uid,
                                           save_vis=save_vis, thresh_bgr=thresh_bgr, cached_prod=cached_prod,p_max=p_max)

    df1 = pd.DataFrame(all_recs)
    return df1, all_dets, all_recs

def get_cms(all_dets, only_k=None):
    """
    get confusion matrices fot all iterations
    :param all_dets:
    :return:
    """
    cms = {}


    for k in all_dets.keys():
        info = all_dets[k]['infos']
        if info['type'] != 'imdb':
            continue
        iteration = info['iteration']
        net, model_name = get_net_and_model_name(all_dets, k)
        model_name = model_name.split('/')[-1]
        cms[k] = {}
        cms[k]['net'] = net
        cms[k]['model'] = model_name
        cms[k]['cm'] = all_dets[k]['rows'][0]['cm']
        if only_k is not None:
            if only_k == k:
                return cms[k]

    return  cms

def plot_all_cms(all_dets, clss, only_net=None, k=None):
    cms = get_cms(all_dets, only_k=k)

    if k is not None and cms is not None:
        plot_confusion_matrix(cms['cm'], classes=clss, title="cm it: {} net: {}".format(k, only_net))
        return

    for cmk in cms.keys():
        net = cms[cmk]['net']
        model_name = cms[cmk]['model'].split('/')[-1]
        if only_net is not None:
            if net != only_net:
                continue

        plot_confusion_matrix(cms[cmk]['cm'], classes=clss, title="cm it: {} net: {}".format(cmk, net))

def get_net_and_model_name(all_dets, k):
    model = all_dets[k]['infos']['model']
    net = model.split('/')[-1].split('_')[0]
    return net, model

def get_nets_and_model_names(df_full):
    nets = []
    model_names = []
    for model in df_full['model'].values.tolist():
        net = model.split('/')[-1].split('_')[0]
        nets.append(net)
        model_names.append(model.split('/')[-1].split('.')[0][len(net):])
    return nets, model_names

def add_nets_and_model_names_columns(df_full):
    nets, model_names = get_nets_and_model_names(df_full)
    df_full['nets'] = nets
    df_full['model_names'] = model_names
    return df_full


def get_all_imdb_tests(df_full):
    return df_full.loc[df_full['type'] == 'imdb']

def get_all_noimdb_tests(df_full):
    return df_full.loc[df_full['type'] == 'no_imdb']

def plot_loss_and_AP_mean_for_imdb_tests(df_full):
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.set_title('loss and  AP mean for all imdb tests')
    df_full[['loss', 'ap_mean']].loc[df_full['type'] == 'imdb'].plot(ax=ax, rot=45)
    #todo - plot average lines

def plot_AP_for_imdb_tests(df_full):
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title('AP for all imdb tests')
    df_full.loc[df_full['type'] == 'imdb'].drop(['loss', 'ap_mean'], axis=1, inplace=False).plot(ax=ax, kind='barh',
                                                                                                 stacked=True, rot=45)


def plot_ap_bygroups(df_full):

    df_full = ensure_net_key(df_full)

    df_imdb = df_full.drop(['loss'], axis=1, inplace=False).loc[df_full['type'] == 'imdb'].groupby(['nets', 'iteration', 'imdb_name'])
    df_imdb.sum().plot(rot=45)

def plot_ap_mean_and_loss_bygroups(df_full):

    df_full = ensure_net_key(df_full)

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title('AP Mean and loss by groups')
    df_imdb = df_full[['nets', 'iteration', 'imdb_name','loss', 'ap_mean']].loc[df_full['type'] == 'imdb'].groupby(['nets', 'iteration', 'imdb_name'])
    df_imdb.sum().hist(rot=45, ax=ax)

def plot_back_loss_bygroups(df_full):

    df_full = ensure_net_key(df_full)

    df_imdb = df_full[['nets', 'iteration', 'imdb_name','loss']].loc[df_full['type'] == 'no_imdb'].groupby(['nets', 'iteration', 'imdb_name'])
    df_imdb.sum().plot(rot=45)

def plots_all(df_full):
    plot_loss_and_AP_mean_for_imdb_tests(df_full)
    plot_AP_for_imdb_tests(df_full)
    plot_ap_bygroups(df_full)
    plot_ap_mean_and_loss_bygroups(df_full)
    plot_back_loss_bygroups(df_full)


def box_plots(df_full):
    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.set_title('AP Mean and loss by groups')
    df_imdb = df_full[['nets', 'iteration', 'imdb_name', 'loss', 'ap_mean']].loc[df_full['type'] == 'imdb']
    # .groupby(['nets', 'iteration', 'imdb_name'])
    df_imdb.boxplot(ax=ax, by=['nets'])

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.set_title('AP Mean and loss by iteration')
    df_imdb.boxplot(ax=ax, by=['iteration'], rot=45)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.set_title('AP Mean and loss by imdb_name')
    df_imdb.boxplot(ax=ax, by=['imdb_name'], rot=45)

    for n in ('nets','iteration','imdb_name'):
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.set_title('AP by '+n)
        df_imdb = df_full.loc[df_full['type'] == 'imdb'].drop(['loss', 'ap_mean'], axis=1, inplace=False)
        # .groupby(['nets', 'iteration', 'imdb_name'])
        df_imdb.boxplot(ax=ax, by=[n], rot=45)


    for n in ('nets','iteration','imdb_name'):
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.set_title('AP by '+n)
        df_imdb = df_full[['nets', 'iteration', 'imdb_name','loss']].loc[df_full['type'] == 'no_imdb']
        # .groupby(['nets', 'iteration', 'imdb_name'])
        df_imdb.boxplot(ax=ax, by=[n], rot=45)



def box_ap_loss_by_iterations_by_imdb_name(df_full, net='zf'):

    keys = np.unique(df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net)]['imdb_name'])

    for k in keys:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # df["A"][(df["B"] > 50) & (df["C"] == 900)]
        df_imdb = df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net)& (df_full['imdb_name'] == k)][
            ['nets', 'int_iteration', 'imdb_name', 'loss', 'ap_mean']]
        # .groupby(['nets', 'iteration', 'imdb_name'])
        axes = df_imdb.boxplot(ax=ax, by=['int_iteration'], rot=-45, fontsize=7)
        ax[1].set_ylim(0, 4)
        ax[0].set_ylim(0, 1)
        fig.suptitle('AP by iteration - for zf {} imdb {}'.format(net, k))

def box_ap_loss_by_iterations(df_full, net='zf'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # df["A"][(df["B"] > 50) & (df["C"] == 900)]
    df_imdb = df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net)][
        ['nets', 'int_iteration', 'imdb_name', 'loss', 'ap_mean']]
    # .groupby(['nets', 'iteration', 'imdb_name'])
    axes = df_imdb.boxplot(ax=ax, by=['int_iteration'], rot=-45, fontsize=7)
    ax[1].set_ylim(0, 4)
    ax[0].set_ylim(0, 1)
    fig.suptitle('AP by iteration - for zf {}'.format(net))

def box_back_loss_by_iterations(df_full, net='zf'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # df["A"][(df["B"] > 50) & (df["C"] == 900)]
    df_imdb = df_full[(df_full['type'] == 'no_imdb') & (df_full['nets'] == net)][
        ['nets', 'iteration', 'imdb_name', 'loss']]
    # .groupby(['nets', 'iteration', 'imdb_name'])
    axes = df_imdb.boxplot(ax=ax, by=['int_iteration'], rot=-45, fontsize=7)
    fig.suptitle('Back loss by iteration - for zf {}'.format(net))

def box_aps_by_iterations(df_full, net='zf'):
    #fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # df["A"][(df["B"] > 50) & (df["C"] == 900)]
    df_imdb = df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net)].drop(['loss', 'ap_mean'], axis=1, inplace=False)
    # .groupby(['nets', 'iteration', 'imdb_name'])
    axes = df_imdb.boxplot(by=['int_iteration'], figsize=(12, 6), rot=-45, fontsize=7)
    #ax[1].set_ylim(0, 4)
    #ax[0].set_ylim(0, 1)
    #fig.suptitle('AP by iteration - for zf {}'.format(net))


def plot_APmean_loss_for_net(df_full, net='zf', ax=None):
    df_imdb = df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net)][['nets', 'int_iteration', 'imdb_name', 'loss', 'ap_mean']]
    if ax is not None:
        return df_imdb.groupby(['int_iteration']).mean().plot(ax=ax, sort_columns=True, secondary_y='ap_mean')
    return df_imdb.groupby(['int_iteration']).mean().plot(sort_columns=True, secondary_y='ap_mean')


def search_by_string_in_column(df, column_name, key):
    return df[df[column_name].str.contains(key)]

def get_uid_by_net_iteration_and_testimdb(df_full, net, it, testimdb):
    df =  df_full[(df_full['nets'] == net)  &
            (df_full['iteration'] == it) &
            (df_full['training_imdb'].str.contains(testimdb))]

    if(len(df))>0:
        return df['uid']

def get_APmean_loss_for_uid_testcase(df_full, net, uid):
    df_imdb = df_full[(df_full['type'] == 'imdb') & (df_full['nets'] == net) & (df_full['uid']==uid) ] \
        [['nets', 'int_iteration', 'imdb_name', 'loss', 'ap_mean']]
    return df_imdb

def get_APmean_loss_for_testcase_by_key_and_net(df_full, all_dets, test_case_key, net):
    uid = all_dets[test_case_key]['infos']['uid']
    return get_APmean_loss_for_uid_testcase(df_full, net=net, uid=uid)

def plot_APmean_loss_for_nets(df_full):
    ax = plot_APmean_loss_for_net(df_full, 'vgg16')
    ax = plot_APmean_loss_for_net(df_full, 'zf', ax=ax)

def iteration_to_int(df_full):
    int_iterations = [int(l) for l in df_full['iteration']]
    df_full['int_iteration'] = int_iterations
    return df_full

def ensure_net_key(df_full):
    if not 'nets' in df_full.columns:
        df_full = add_nets_and_model_names_columns(df_full)
    return df_full

def prepare_df_for_analysis(df_full):
    df_full = ensure_net_key(df_full)
    df_full = iteration_to_int(df_full)
    return df_full

def get_sorted_by_ap_mean(df):
    return df.sort(['ap_mean'], ascending=[False])


def max_for_class(max_for_clss_arr, rows, cls_k):
    #cls_k included background class
    im_k_bet={}
    for imk in max_for_clss_arr:

        if isinstance(rows[imk][-1], np.float):
            true_cls = rows[imk][-1].astype(int)
        else:
            true_cls = rows[imk][-1][0].astype(int)
        if true_cls == cls_k:
            im_k_bet[imk]=max_for_clss_arr[imk]
    return im_k_bet



def get_ypred_from_all_dets(all_dets, key_index=None):
    if key_index is None:
        key_index=0
    if key_index==None and len(all_dets.keys())>1:
        print "warning, using key index 0, but all_dets have {} indexes".format(len(all_dets.keys()))
    return all_dets[all_dets.keys()[key_index]]['rows'][0]['rows']

def get_info_from_all_dets(all_dets, key_index=None):
    if key_index is None:
        key_index=0
    if key_index==None and len(all_dets.keys())>1:
        print "warning, using key index 0, but all_dets have {} indexes".format(len(all_dets.keys()))
    return all_dets[all_dets.keys()[key_index]]['infos']

def count_dets4clss_for_noimdb_dets(preds, lbl_clss):
    pred_counter = {}
    tot = 0

    for kk in preds.keys():

        img0 = kk

        if preds[kk].shape[0]>len(lbl_clss):
            pp = preds[kk][:-1]
        else:
            pp = preds[kk]
        # get arg max
        max_cls = np.argmax(pp)
        pred_cls = lbl_clss[int(max_cls)]
        if not pred_counter.has_key(pred_cls):
            pred_counter[pred_cls] = 0
        pred_counter[pred_cls] += 1
        tot += 1

    return pred_counter, tot

def count_dets_if_pred_cls(max_dets, rows, info, lbl_clss, project_name='fish'):
    pred_counter = {}
    for kk in max_dets.keys():

        img0 = kk
        #true_cls = lbl_clss[rows[img0][-1][0].astype(int)]

        #ann_p, imset_file, path_imgs = get_paths_from_imdb(info['imdb_name'], project_name=project_name)
        #im_path = (path_imgs + img0 + '.jpg')
        for k in max_dets[img0].keys():

            pred_cls = lbl_clss[int(k)]
            if not pred_counter.has_key(pred_cls):
                pred_counter[pred_cls] = 0
            pred_counter[pred_cls] += 1

            # if pred_cls!=only_if_pred_cls:
            #    print "{} {} diverse, continue".format(pred_cls, only_if_pred_cls)


            # dets = np.expand_dims(max_dets[img0][k], axis=0)

            # save_detections_old(img_path=im_path, dets=dets, true_class=true_cls, pred_class=pred_cls, save=False, figsize=(8,5))
    return pred_counter


def get_max_for_all_clsses(max_for_clss, class_to_ind, rows, info, lbl_clss, project_name='fish'):
    """
    while get_max_for_all_classes in check_detections_imbd get only the dets with the max confidence
    [because for computing loss we need only one row for image]
    here we count detections > thresh, not only the max
    so the counts for correct detection are different with confusion matrix ones

    Detection for background are ignored.

    :param max_for_clss:
    :param class_to_ind:
    :param rows:
    :param info:
    :param lbl_clss:
    :param project_name:
    :return:
    """
    mx={}
    for kc, c in enumerate(lbl_clss):
        d_max = max_for_class(
            max_for_clss, rows, kc
        )
        d_max = rm_empty_from_list_of_list(d_max)
        #print "---------det for {}".format(c)
        mx[c]=count_dets_if_pred_cls(d_max, rows, info, lbl_clss, project_name)
    return mx

def get_counter_from_ytrue(y_true):
    from collections import Counter
    return Counter(y_true.squeeze())


def get_only_rows_from_rows_dict(rows, lbl_clss, keep_trues=False, ret_trues=False):

    plus = 0 if not keep_trues else 1
    only_row=np.empty((len(rows.keys()),len(lbl_clss)+plus))
    only_true = np.empty((len(rows.keys())))
    idx=0
    for im_name in rows.keys():
        # if shape[9]>len(classes)
        #last index is for true class, remove it
        if rows[im_name].shape[0]>len(lbl_clss):
            if keep_trues:
                only_row[idx] = rows[im_name].squeeze()
            else:
                only_row[idx]=rows[im_name][:-1].squeeze()
        else:
            only_row[idx]=rows[im_name].squeeze()
        only_true[idx] = rows[im_name][-1]
        idx+=1

    if ret_trues:
        return only_row, only_true
    return only_row


def do_clip(arr, mx): return np.clip(arr, (1 - mx) / 7, mx)
def do_clip_min_max(arr, min, max): return np.clip(arr, min, max)

def get_loss_from_dets(dets, key_index, lbl_clss, doclip=False, cmin=None, cmax=None):

    rows = get_ypred_from_all_dets(dets, key_index=key_index)

    prob_pred, y_true = get_only_rows_from_rows_dict(rows, lbl_clss, ret_trues=True)

    if doclip:
        if cmin is not None and cmax is not None:
            prob_pred = do_clip_min_max(prob_pred, cmin, cmax)
        else:
            prob_pred = do_clip(prob_pred, 0.85)
            # pdb.set_trace()

    # delete background column
    prob_pred = np.delete(prob_pred, 0, 1)
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_true.squeeze())

    dbg_print("true classes {}".format(lb.classes_))
    true_prob = lb.transform(y_true)
    # print pred.shape, y.shape
    # loss = log_loss(y_pred=pred, y_true=y)
    loss = my_multi_log_loss(y_pred=prob_pred, y_true=true_prob)

    return  prob_pred, loss, true_prob

def merge_by_means(dets, k1, k2, lbl_clss, cmin=0.015, cmax=.99, doclip=True):
    x, loss, true_prob1 = get_loss_from_dets(dets, k1, lbl_clss, cmin=cmin, cmax=cmax, doclip=doclip)
    x2, loss2, true_prob2 = get_loss_from_dets(dets, k2, lbl_clss, cmin=cmin, cmax=cmax, doclip=doclip)

    meanx = x+x2/2.
    new_loss = my_multi_log_loss(y_pred=meanx, y_true=true_prob1)
    return (new_loss, loss, loss2, abs(new_loss-loss), abs(new_loss-loss2))

def create_submission(rows, lbl_clss, out_path, prefix='', doclip=False, clip_min=None, clip_max=None):
    file_names = rows.keys()
    
    # if there isnt .jpg add to filenames
    if file_names[0][-4:] != '.jpg':
        file_names = [f+'.jpg' for f in file_names]

    y_pred = get_only_rows_from_rows_dict(rows, lbl_clss)

    # if file_names.shape[0] != pred_class.shape[0]:
    #    print "missing same image in predictions..."
    #    return

    # reorder class probabilities according to fish competition order
    # my zero class become 4 and 1-4 shift left by one index
    # CLASSES = ('__background__',  # always index 0, class NoF as background ?
    #                    'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft'
    #                   )

    permutation = [1, 2, 3, 4, 0, 5, 6, 7]
    y_pred_perm = y_pred[:, permutation].copy()

    if doclip:
        if clip_min is not None and clip_max is not None:
            subm = do_clip_min_max(y_pred_perm, clip_min, clip_max)
        else:
            subm = do_clip(y_pred_perm, 0.85)
        #pdb.set_trace()
    else:
        subm = y_pred_perm

    subm_name = out_path + '/{}_subm_bb.gz'.format(prefix)

    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'image', file_names)
    submission.head()
    submission.to_csv(subm_name, index=False, compression='gzip')
    # FileLink(subm_name)

def found_img_not_detected_from_path(glob_path_imgs, all_dets):
    """
    return the list of images in path_img that not have a detection

    :param path_img:
    :param rows:
    :return:
    """

    rows = get_ypred_from_all_dets(all_dets)
    not_dets=[]
    file_names = rows.keys()
    im_set = glob(glob_path_imgs)
    for f in im_set:
        fn = f.split('/')[-1]
        if fn not in file_names:

            print "fn {} not in filenames".format(fn)
            not_dets.append(f)
    return not_dets


import shutil

def get_df_test_stg1_yolo_by_it(iteration):     

    clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
    lbl_clss = ('NoF', 'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
    prepath_stg = '/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/pose-estimation-detection-localization/darknet/data/fish/test_stg1/results/'
    format_file = 'comp4_det_test_{}.txt'

    path_yolo_res ='yolo_{}/'.format(iteration)
    imdb_name = 'test_stg1'

    output_dir=prepath_stg+path_yolo_res+'/output_info/'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    du.mkdirs(output_dir)

    args = edict()
    args.caffemodel = 'yolo_{}'.format(iteration)
    args.path_to_detect = ''
    args.model_iteration = iteration
    args.imdb_name = imdb_name
    args.all_out_dir = output_dir
    
    file_2format = prepath_stg+path_yolo_res+format_file
    test_stg1_path='/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/test_stg1/*.jpg'
    imdb_type = 'no_imdb'
    all_boxes = create_bboxes_no_imdb(lbl_clss, test_stg1_path,output_dir, file_2format)
    
    create_tar_and_info(imdb_type, output_dir, args, all_boxes)
    
    df_yolo, dets_yolo, recs_yolo = test_extract(output_dir, clss, save_vis=False)
    with open(output_dir+'pred_analysis.pkl', 'w') as f:
        cPickle.dump({'df':df_yolo, 'dets':dets_yolo, 'recs':recs_yolo}, f, cPickle.HIGHEST_PROTOCOL)

    return df_yolo, dets_yolo, recs_yolo

def get_df_newsynth300x_yolo_by_it(iteration):
    clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
    prepath_stg = '/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/pose-estimation-detection-localization/darknet/data/fish/new_synth_300x/results/'
    imdb_name = 'fish_new_synth_300x'
    return get_df_for_dataset_yolo_by_it(iteration, prepath_stg, imdb_name, clss)

def get_df_validfromtrain_yolo_by_it(iteration):
    clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
    prepath_stg = '/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/pose-estimation-detection-localization/darknet/data/fish/validfromtrain/results/'
    imdb_name = 'fish_validfromtrain'
    return get_df_for_dataset_yolo_by_it(iteration, prepath_stg, imdb_name, clss)

def get_df_for_dataset_yolo_by_it(iteration, prepath_stg, imdb_name, clss):
    format_file = 'comp4_det_test_{}.txt'

    path_yolo_res ='yolo_{}/'.format(iteration)
    
    output_dir=prepath_stg+path_yolo_res+'/output_info/'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    du.mkdirs(output_dir)

    args = edict()
    args.caffemodel = 'yolo_{}'.format(iteration)
    args.path_to_detect = ''
    args.model_iteration = iteration
    args.imdb_name = imdb_name
    args.all_out_dir = output_dir
    
    file_2format = prepath_stg+path_yolo_res+format_file    
    imdb_type = 'imdb'
    imdb = get_imdb(imdb_name)
    all_boxes = create_bboxes(imdb, output_dir, file_2format)

    imdb.evaluate_detections(all_boxes, output_dir)
    
    create_tar_and_info(imdb_type, output_dir, args, all_boxes)
    
    df_yolo, dets_yolo, recs_yolo = test_extract(output_dir, clss, save_vis=False)
    with open(output_dir+'pred_analysis.pkl', 'w') as f:
        cPickle.dump({'df':df_yolo, 'dets':dets_yolo, 'recs':recs_yolo}, f, cPickle.HIGHEST_PROTOCOL)
        
    return df_yolo, dets_yolo, recs_yolo

def get_scores_for_cls(c, file_for_format_cls):
    dict_dets = {}
    format_path_dets=file_for_format_cls.format(c)    
    with open(format_path_dets) as fp:
        for line in fp:
            det_line = line.split(' ')
            
            if not dict_dets.has_key(det_line[0]):
                dict_dets[det_line[0]] = []
            det_line[-1] = det_line[-1].replace('\n','')
            
            box_yolo = det_line[1:]
            box_yolo = [float(b) for b in box_yolo]
            dict_dets[det_line[0]].append(box_yolo) 
            
    return dict_dets

def create_bboxes(imdb, output_dir, file_for_format_cls):

    dict_dets = {}
    num_images = len(imdb.image_index)
    thresh = 0.05
    all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]

    for i in xrange(num_images):
        im_n = imdb.image_index[i]
        for j in xrange(1, imdb.num_classes):               

                    c = imdb.classes[j]
                    if not dict_dets.has_key(c):
                        dict_dets[c] = get_scores_for_cls(c, file_for_format_cls)
                    dets = dict_dets[c]
                        
                    if not dets.has_key(im_n):
                        continue
                    scores_boxes = np.array(dets[im_n])
                    cls_scores = scores_boxes[:, 0]
                    cls_boxes = scores_boxes[:, 1:]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    #print len(cls_dets)
                    #keep = nms(cls_dets, cfg.TEST.NMS)
                    #print len(keep)                    
                    #cls_dets = cls_dets[keep, :]                
                    all_boxes[j][i] = cls_dets

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    
    return all_boxes

def create_bboxes_no_imdb(classes, path_images_glob, output_dir, file_for_format_cls):
        
    ##all_imgs_det[img_name] = all_boxes only for img_name
        
    #classes with background, but 0 index is skipped
    
    imset = glob(path_images_glob)
    dict_dets = {}
    all_imgs_det = {}
    num_images = len(imset)
    thresh = 0.05

    for i in xrange(num_images):
        im_n = imset[i].split('/')[-1][:-4]
        for j in xrange(1, len(classes)):               

                    c = classes[j]
                
                    # ensure all keys for images and classes
                    if not all_imgs_det.has_key(im_n):
                        all_imgs_det[im_n] = {}                    
                    all_imgs_det[im_n][c] = []                    
                    
                    if not dict_dets.has_key(c):
                        dict_dets[c] = get_scores_for_cls(c, file_for_format_cls)
                    dets = dict_dets[c]
                    if not dets.has_key(im_n):
                        continue
                    scores_boxes = np.array(dets[im_n])
                    cls_scores = scores_boxes[:, 0]
                    cls_boxes = scores_boxes[:, 1:]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                    #print len(cls_dets)
                    #keep = nms(cls_dets, cfg.TEST.NMS)
                    #print len(keep)                    
                    #cls_dets = cls_dets[keep, :]                
                    all_imgs_det[im_n][c] = cls_dets
         

    #det_file = os.path.join(output_dir, 'detections.pkl')
    #with open(det_file, 'wb') as f:
    #    cPickle.dump(all_imgs_det, f, cPickle.HIGHEST_PROTOCOL)
    
    return all_imgs_det

import uuid
def create_base_info(args):

    info = {}
    info['uid'] = str(uuid.uuid4())
    return info

def create_tar_and_info(imdb_type, output_dir, args, all_boxes):
    # imdb case
    # for imdb
    evalutation_pkl = 'info_evalutation.pkl'

    if imdb_type == 'imdb':
        p_eval_info = output_dir+'info_evalutation.pkl'

        with open(p_eval_info, 'r') as f:
            info = cPickle.load(f)    
    else:
        info = None

    out_dir = output_dir
    if info is None:
        info = create_base_info(args)
    info['iteration'] = args.model_iteration
    info['model'] = args.caffemodel
    info['training_imdb'] = args.imdb_name
    info['type'] = imdb_type

    print('Saving {} to {}'.format(evalutation_pkl, out_dir))

    if imdb_type == 'imdb':
        to_save = {'info':info}
    else:
        to_save = {'info':info, 'dets':all_boxes }
    with open(os.path.join(out_dir, evalutation_pkl), 'w') as f:
        cPickle.dump(to_save, f)

    create_tar_info(args, info, out_dir)    

def yolo_create_submission(df, all_dets, folder_ds, out_path='/tmp', clip=True):

        clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
        lbl_clss = ('NoF', 'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
        class_to_ind = du.array_to_ind(clss)
        
        df = prepare_df_for_analysis(df)

        rows = get_ypred_from_all_dets(all_dets)
        create_submission(rows=rows, lbl_clss=lbl_clss, out_path=out_path, prefix=folder_ds, doclip=clip)
        
        return rows    

# merging for newsynth300x
def merge_dets(dets1, dets2, clss, add_ext=[False, False], directly=[False, False], k1=0, k2=0):
    # classes without background
    
    if directly[0]:
        pred1 = dets1
    else:
        pred1 = get_ypred_from_all_dets(dets1, key_index=k1)
        
    if directly[1]:
        pred2 = dets2
    else:
        pred2 = get_ypred_from_all_dets(dets2, key_index=k2)
        
    new_pred = {}
    n_classes = len(clss)+1
    has_extra_col = len(pred1[pred1.keys()[0]])>n_classes
    
    # if equal argmax continue
    # else get argmax with p higher
    # but if one of the argmax is 0 gets the other
    for k in pred1.keys():
        k1 = k
        k2 = k
        
        if add_ext[0]:
            k1+='.jpg'
        if add_ext[1]:
            k2+='.jpg'
            
        r1 = pred1[k1][:-1] if has_extra_col else pred1[k1][:]
        r2 = pred2[k2][:-1] if has_extra_col else pred2[k2][:]
        max1 = np.argmax(r1)
        max2 = np.argmax(r2)
        low_val = np.argmin(r1)
        if max1 == max2:
            new_pred[k1] = pred2[k2]
            continue
            
        if r1.shape[0] != r2.shape[0]:
            print "error, different shape"
            return None, None, None
        
        new_r = np.ones_like(r1)*float(r1[low_val])
        if (r1[max1] > r2[max2]) and max1>0:
            max_idx = max1
            max_p = r1[max1] 
        elif (r1[max1] <= r2[max2]) and max2>0:
            max_idx = max2
            max_p = r2[max2]
        elif max1 == 0:
            max_idx = max2
            max_p = r2[max2]
        elif max2 == 0:
            max_idx = max1
            max_p = r1[max1]

        new_r[max_idx]=max_p
        if has_extra_col:
            new_r = np.vstack((new_r, pred1[k1][-1]))
        new_pred[k1] = new_r
    return new_pred, pred1, pred2


# merge by mean
def merge_dets_by_mean(dets1, dets2, clss, k1=0, k2=0):
    # classes without background

    pred1 = get_ypred_from_all_dets(dets1, key_index=k1)

    pred2 = get_ypred_from_all_dets(dets2, key_index=k2)

    new_pred = {}
    n_classes = len(clss) + 1
    has_extra_col = len(pred1[pred1.keys()[0]]) > n_classes

    # if equal argmax continue
    # else get argmax with p higher
    # but if one of the argmax is 0 gets the other
    for k in pred1.keys():
        k1 = k
        k2 = k

        r1 = pred1[k1][:-1] if has_extra_col else pred1[k1][:]
        r2 = pred2[k2][:-1] if has_extra_col else pred2[k2][:]

        new_r = (r1 + r2) / 2.

        if has_extra_col:
            new_r = np.vstack((new_r, pred1[k1][-1]))
        new_pred[k1] = new_r
    return new_pred, pred1, pred2


def merge_dets_and_boxes(dets1, dets2, boxes1, boxes2, clss, add_ext=[False, False], directly=[False, False]):
    # classes without background

    if directly[0]:
        pred1 = dets1
    else:
        pred1 = get_ypred_from_all_dets(dets1)

    if directly[1]:
        pred2 = dets2
    else:
        pred2 = get_ypred_from_all_dets(dets2)

    new_pred = {}
    new_bboxes = {}
    n_classes = len(clss)+1
    has_extra_col = len(pred1[pred1.keys()[0]])>n_classes

    # if equal argmax continue
    # else get argmax with p higher
    # but if one of the argmax is 0 gets the other
    for k in pred1.keys():
        k1 = k
        k2 = k

        if add_ext[0]:
            k1+='.jpg'
        if add_ext[1]:
            k2+='.jpg'

        r1 = pred1[k1][:-1] if has_extra_col else pred1[k1][:]
        r2 = pred2[k2][:-1] if has_extra_col else pred2[k2][:]
        max1 = np.argmax(r1)
        max2 = np.argmax(r2)
        low_val = np.argmin(r1)
        
        if max1 == max2:
            new_pred[k1] = pred2[k2]
            new_bboxes[k1] = boxes1[k1]
            continue

        if r1.shape[0] != r2.shape[0]:
            print "error, different shape"
            return None, None, None

        new_r = np.ones_like(r1)*float(r1[low_val])
        
        if (r1[max1] > r2[max2]) and max1>0:
            max_idx = max1
            max_p = r1[max1]
            box = boxes1[k1]
        elif (r1[max1] <= r2[max2]) and max2>0:
            max_idx = max2
            max_p = r2[max2]
            box = boxes2[k2]
        elif max1 == 0:
            max_idx = max2
            max_p = r2[max2]
            box = boxes2[k2]
        elif max2 == 0:
            max_idx = max1
            max_p = r1[max1,:]
            box = boxes1[k1]

        new_r[max_idx]=max_p
        
        if has_extra_col:
            new_r = np.vstack((new_r, pred1[k1][-1]))
        new_pred[k1] = new_r
        new_bboxes[k1] = box
    return new_pred, pred1, pred2, new_bboxes

def create_ds_from_bbox_and_conf(base_path, max_boxes, rows, base_path_imgs, clss):
    """
    true row deve avere la colonna finale con la classe corretta
    clss deve contenere Background_class
    """
    
    for ic,c in enumerate(clss):
        if ic == 0:
            continue
        du.mkdirs(base_path+'/'+c.upper())
    
    keys = max_boxes.keys()    
    
    for idx,k_img in enumerate(keys):
        box = max_boxes[k_img]
        
        true_cls_idx = rows[k_img][-1].astype(np.int)
        #print "true clss idx: {}".format(true_cls_idx)
        
        true_cls = clss[true_cls_idx]
        #print "true clss: {}".format(true_cls)
        
        img_p = base_path_imgs+k_img+'.jpg'
        #print "im:{}".format(img_p)
                
        write_bbox(img_p, box, base_path+'/'+true_cls.upper(), k_img, ext='jpg', plot=False)
        if idx % 100 == 0:
            print "done {}/{}".format(idx, len(keys))

def create_ds_from_bbox_no_gt(base_path, max_boxes, rows, base_path_imgs):
    """
    
    """
    backgrounds = []
    du.mkdirs(base_path)
    
    keys = max_boxes.keys()    
    
    for idx,k_img in enumerate(keys):
        box = max_boxes[k_img]
        if len(box)==0:
            print "skip {}, background prediction"
            backgrounds.append(k_img)
            continue
        img_p = base_path_imgs+k_img+'.jpg'
        #print "im:{}".format(img_p)
                
        write_bbox(img_p, box, base_path+'/', k_img, ext='jpg', plot=False)
        if idx % 100 == 0:
            print "done {}/{}".format(idx, len(keys))            
    return backgrounds    

def simple_vis_detections(im_path, bbox, confidence=None, pred_cls=None, true_cls=None):
    
    im = cv2.imread(im_path)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(im, aspect='equal')

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )

    
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

    
import numpy.random  as rnd
def vis_n_box_pred(max_boxes, n, base_path_imgs, random=True, rows=None):
    keys = max_boxes.keys()
    if random:
        mbk = rnd.choice(keys, n)
    else:
        mbk = keys
    print mbk
    for idx,k_img in enumerate(mbk):
        box = max_boxes[k_img]
        if rows is not None:
            confid = rows[k_img]
        img_p = base_path_imgs+k_img+'.jpg'
        print "im:{}".format(img_p)
        simple_vis_detections(img_p, box)
        if idx==(n-1):
            break


if __name__ == '__main__':
    clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
    lbl_clss = ('NoF', 'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')

    #folder_alb_b10 = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/pkl_dets/all_pkl_exp_alb_rot_sc_right'
    #df, all_dets, all_recs = test_extract(folder_alb_b10, clss, save_vis=False)

    # test imdb by uid
    #path_ds_selection = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/pkl_dets/all_pkl_ds_selection'
    #uid_15000_42 = 'b888a431-d683-4bdb-952b-1da985eba754'
    #df_1500042, all_dets_1500042, all_recs_1500042 = test_extract(path_ds_selection, clss, get_uid_only=uid_15000_42,save_vis=False)

    # test no imdb by uid
    uid_test_stg1_41500 = 'f1d163c5-19e6-48b2-bc1d-30dd54c4b15b'
    path_test_stg1 = '/media/sergio/6463910e-9535-4dcb-a780-a59bbb72e081/temp_data/fish_res/pkl_dets/test_stg1_train_2rot_2sc_6b_all_vgg16_41500'
    df_test_stg1_41500, all_dets_test_stg1_41500, all_recs_test_stg1_41500 = test_extract(path_test_stg1, clss,
                                                                                          get_uid_only=uid_test_stg1_41500,
                                                                                          save_vis=False)
    df_test_stg1_41500 = prepare_df_for_analysis(df_test_stg1_41500)

    # test create submission
    rows = get_ypred_from_all_dets(all_dets_test_stg1_41500)
    create_submission(rows=rows, lbl_clss=lbl_clss, out_path='/home/sergio/Scrivania/', prefix='test_stg1_41500_clp', doclip=True)
