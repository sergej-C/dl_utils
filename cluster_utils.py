from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os, sys
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans    
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth, MeanShift
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import imread, imwrite, resize
from features_utils import *
import data_utils as du
from skimage.feature import hog as skhog
import io_utils

def vis_clusters(labels, n_clusters, imgs, w=2000, h=2000, max_img_per_cluster=10,zoom=.07,padding = 10, pix_w=3, pix_h=1.5):

    h_cluster = int(h/float(n_clusters))

    ax = plt.gca()

    #images = [OffsetImage(image, zoom=zoom) for image in cvimgs]
    artists = []
    half_height = 100

    start_x = padding
    start_y = padding

    for k in range(n_clusters):
        my_members = labels == k
        idx = np.array(range(len(labels)))
        idx_for_k = idx[my_members]

        y0 = start_y + (h_cluster*k*pix_h) + (half_height)
        
        for im_i in range(max_img_per_cluster):
            
            if im_i >= idx_for_k.shape[0]:
                print "less then min imgs i:{} len:{}".format(im_i, idx_for_k.shape)
                break
            im_p = imgs[idx_for_k[im_i]]
            img = imread(im_p)[:,:,(2,1,0)]
            im0 = OffsetImage(img, zoom=zoom)
            
            wf = img.shape[1]*zoom*pix_w
            x0 = start_x + (wf*im_i) + (wf/2)

            ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.set_ylim([0,h])
        ax.set_xlim([0,w])

    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()
    
import shutil

def save_clusters(labels, n_clusters, imgs, cls2ind=None, folder_name='cluster_result', path_save='/tmp', test=False):

    counter_cls = dict()
    path_ = os.path.join(
            path_save, 
            folder_name
        )
    if os.path.exists(path_):
        shutil.rmtree(path_)                  
    os.mkdir(path_) 
    
    for k in range(n_clusters):
        path_k = path_+ '/'+ str(k)
        if not os.path.exists(path_k):
            os.mkdir(path_k)
            
        my_members = labels == k
        idx = np.array(range(len(labels)))
        idx_for_k = idx[my_members]

        if not counter_cls.has_key(k):
            counter_cls[k] = dict()
                
        for im_i in range(idx_for_k.shape[0]):
                        
            im_p = imgs[idx_for_k[im_i]]
            if test==False:
                type_ = im_p.split('/')[-2]
            else:
                type_ = 'unknown'

            if not counter_cls[k].has_key(type_):
                counter_cls[k][type_] = 0
            counter_cls[k][type_] += 1
            img = imread(im_p)
            img = resize(img, (int(img.shape[1]/3.),int(img.shape[0]/3.)), interpolation =cv2.INTER_CUBIC)    
            fil_n = im_p.split('/')[-1]
            if cls2ind is not None:
                fil_n = str(cls2ind[type_])+'_'+fil_n
            else:
                if test==False:
                    fil_n = type_+'_'+fil_n
            cv2.imwrite(path_k+'/'+fil_n, img)
    return counter_cls

def bandwith_and_meanshift(Data, quantile=0.2, n_samples_bdw=500):
    bandwidth = estimate_bandwidth(Data, quantile, n_samples=n_samples_bdw)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Data)
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    return ms, labels, cluster_centers,n_clusters_


def kmeans(data, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, copy_x=True, n_jobs=1, algorithm='auto', seed=1):
    np.random.seed(seed)
    
    n_samples, n_features = data.shape
    print("n_samples %d, \t n_features %d"
      % (n_samples, n_features))
    
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        init=init,
        algorithm=algorithm,        
        random_state=seed, 
        verbose=verbose)
    
    model.fit(data)
    
    return model
        

def bench_cluster_model(estimator, name, data, sample_size, labels):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))



def merge_cluster_by_groups(dict_groups, model=None, labels_=None, hogs=None, true_labels=None):

    """ example of grouping
        group2merge = {
            0: [0,4,7,9,14],
            1: [1,2,3,5,6,8,10,11,12,13]
        }
        every index is a label in  cluster_model.labels_
        
    """
    groups = dict()

    if model is not None:
        clust_labels = model.labels_
    else:
        clust_labels = labels_

    # prendo gli indici corrispondenti ad ogni elemento del cluster
    # e inserisco nel gruppo corrispondente
    # gli stessi indici mi serviranno per prendere gli hog, le immagini, i labels
    total_el = 0
    for group in dict_groups:
        groups[group] = []

        for clust_idx in dict_groups[group]:

            my_members = clust_labels == clust_idx
            idx = np.array(range(len(clust_labels)))
            idx_for_k = idx[my_members]

            for idx_ in idx_for_k:
                groups[group].append(idx_)
                total_el+=1

    if hogs is None and true_labels is None:
        return groups
    
    if hogs is not None:
        hogs_by_group = dict()
        for g in groups:
            hogs_by_group[g] = []
            for idx_ in groups[g]:
                hogs_by_group[g].append(hogs[idx_])

        if true_labels is None:
            return groups, hogs_by_group
        
    if true_labels is not None:
        labels_by_group = dict()
        for g in groups:
            labels_by_group[g] = []
            for idx_ in groups[g]:
                labels_by_group[g].append(true_labels[idx_])
                
        if hogs is None:
            return groups, labels_by_group
        
    return groups, hogs_by_group, labels_by_group

def prepare_features_by_group_on_clustered_imgs(
                path_glob, clust_model, grouping_dict, save_clusters_=False, path_save=None, test=False, 
                ratio_resize = .75, h_resize=100, cls2ind=None):

    """
    params:
    - path_glob: the directory for loading images from in glob format [folder of images is class if not test]
    - clust_model: the model for predict the clusters 
    - grouping_dict: the grouping criteria to apply to the clustered images 
    - save_clusters_: if True save thumbnail copy of images in tmp folder grouped by clusters
    - path_save: if not None, dump all features inside the specified path
    - ratio_resize: for hog features resize images with h = h_resize and w = h*ratio_resize, default .75
    - h_resize: default 100
    - cls2ind: if test is False, gets the folder name of the image and set true labels based on value in cls2ind
   
    return:
    - all_color_hists
    - all_hogs
    - labels
    - groups
    - hogs_by_group 
    - labels_by_group
    """

    # get the list of images 
    imgs_list = glob(path_glob)
    assert len(imgs_list)>0, "no imgs found"

    # extract color histogram features
    # on first image so can get the feature length
    im_ex = cv2.imread(imgs_list[0])
    color_hists_ex = extract_color_hist(im_ex)

    all_color_hists = np.empty((len(imgs_list), color_hists_ex.shape[0]))

    error_imgs = 0
    error_imgs_idx = []
    for i,im in enumerate(imgs_list):
    
        img = cv2.imread(im)
        if img is None:
            error_imgs+=1
            error_imgs_idx.append(i)
            continue

        all_color_hists[i]= extract_color_hist(img)
        if i % 100 == 0:
            print "done extracting color hist {}".format(i)


    print "imgs_list size: {}".format(len(imgs_list))
    for index in sorted(error_imgs_idx, reverse=True):
        del imgs_list[index]
    all_color_hists = np.delete(all_color_hists, error_imgs_idx, 0)
    print "new imgs_list size: {}, color_hist_shape {}".format(len(imgs_list), all_color_hists.shape)

    labels = clust_model.predict(all_color_hists)

    if save_clusters_:
        save_clusters(labels, clust_model.n_clusters, imgs_list, folder_name='cluster_test', test=test)
    
    if path_save is not None:
        du.mkdirs(path_save) 
        io_utils.dump(labels, path_save+'/labels.pkl')
        io_utils.dump(all_color_hists, path_save+'/color_hists.pkl')

    # extract hog features
    ratio = ratio_resize
    h = h_resize
    w = int(h*ratio)
    imscaled = resize(im_ex, (w,h))
    hog_ex = skhog(imscaled[:,:,0].transpose(1,0), visualise=False)
    all_hogs = np.empty((len(imgs_list), hog_ex.shape[0]))
    
    if test is False:
        true_labels = np.empty((len(imgs_list),))
    else:
        true_labels = None

    for i,im in enumerate(imgs_list):

        if test is False:
            cls = im.split('/')[-2]
            true_labels[i] = cls2ind[cls]
            
        cvim = cv2.imread(im)
        imscaled = resize(cvim, (w,h))
        
        img = imscaled[:,:,0].transpose(1,0)
        #print img.shape
        all_hogs[i]= skhog(img)
        if i % 100 == 0:
            print "done extracting hog {}".format(i)
    
    if path_save is not None:
        io_utils.dump(all_hogs, path_save+'/hogs.pkl')


    # now merge by specified grouping
    if test is False:
        groups, hogs_by_group, labels_by_group = merge_cluster_by_groups(grouping_dict, labels_=labels, hogs=all_hogs, true_labels=true_labels)
    else:
        groups, hogs_by_group = merge_cluster_by_groups(grouping_dict, labels_=labels, hogs=all_hogs, true_labels=None)
        labels_by_group = []

    tot_el = 0
    for g in groups:
        tot_el += len(groups[g])

    assert tot_el == len(imgs_list), 'error in grouping, number in grouping {}, in imgs list {}'.format(tot_el, len(imgs_list))

    return all_color_hists, all_hogs, labels, groups, hogs_by_group, labels_by_group, imgs_list

 
