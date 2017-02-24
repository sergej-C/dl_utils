import collections
import PIL
import cPickle
from skimage.feature import hog
import cv2
from glob import glob
import sys, os
import numpy as np
from PIL import Image
from PIL import ImageStat

#%matplotlib inline
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PATH_DL_UTILS=os.environ.get('DL_UTILS_PATH')
sys.path.append(PATH_DL_UTILS)
import data_utils as du

# from csn231 practicals
def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # return histogram
  return imhist


# where there is my train folder (from original datasets downloaded from Kaggle)
# base folder in which i save fish data folder
KAGGLE_CONP_DATA=os.environ.get('KAGGLE_PATH')

# some utils
import data_utils
#import dlc_utils

PATH_FISH_ORIG=KAGGLE_CONP_DATA+'/fish'
TRAIN_ORIG_PATH=PATH_FISH_ORIG+'/train'
bins = np.arange(256).reshape(256,1)
                              


def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y



def plot_rgb_hist(im,read=False, plot=True):
    # from http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    if read:
        im = cv2.imread(im)        
        
    chans = cv2.split(im)
    colors = ("b", "g", "r")

    if plot:
        #plt.figure()
        plt.title("'Flattened' Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        if plot:
            # plot the histogram
            plt.plot(hist, color = color)
            plt.xlim([0, 256])

    # here we are simply showing the dimensionality of the
    # flattened color histogram 256 bins for each channel
    # x 3 channels = 768 total values -- in practice, we would
    # normally not use 256 bins for each channel, a choice
    # between 32-96 bins are normally used, but this tends
    # to be application dependent
    #print "flattened feature vector size: %d" % (np.array(features).flatten().shape)
                
    return np.array(features).flatten()
    


def plot_im_with_hist(im_path, hue_too=False):
    
    n_cols=2
    if hue_too:
        n_cols=3
        
    fig = plt.figure(figsize=(10, 6))
    sub1 = plt.subplot(2,n_cols,1)
    sub1.set_xticks(())
    sub1.set_yticks(())
    im = cv2.imread(im_path)
    plt.imshow(im[:,:,(2,1,0)])
    plt.subplot(2,n_cols,2)
    plot_rgb_hist(im)
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    if hue_too:        
        hue = color_histogram_hsv(im[:,:,(2,1,0)])       
        plt.subplot(2,n_cols,3)
        plt.plot(hue)
        
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()

def plot_by_indexes_in_path_list(path_list, 
                                 idx, 
                                 title='', 
                                 plot_subtitles=False,                                 
                                 save=False,
                                 path_to_save=None
                                ):
    #if save:
       
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(len(idx))))
    #print "num idx {}, rows: {}".format(len(idx), n)
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title)
    pidx=1
    for i in idx:
            
        img_name = path_list[i].split('/')[-1]    
        sub1 = plt.subplot(n,n,pidx)        
        sub1.set_xticks(())
        sub1.set_yticks(())
        if plot_subtitles:
            sub1.axes.title.set_text(img_name)
        im = cv2.imread(path_list[i])
        plt.imshow(im[:,:,(2,1,0)])
        pidx+=1
    #fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    if save:
        
        du.mkdirs(path_to_save)
        print "saving in {}".format(path_to_save+'/'+img_name)
        plt.savefig((path_to_save+'/'+img_name))
        plt.close('all')
        

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

def plot_all_clusters(klabes, 
                      path_list=None, 
                      is_path=True, 
                      img_array=None, 
                      is_array=False, 
                      save=False,
                      path_to_save=None
                     ):
    n_clusters = np.sort(np.unique(klabes))
    for l,lab in enumerate(n_clusters):        
        tit = "plotting for cluster {}".format(l)
        idx_cls0 = ClusterIndicesNumpy(l, klabes)
        if(len(idx_cls0)>0):
            if is_path:
                plot_by_indexes_in_path_list(
                    path_list, 
                    idx_cls0, 
                    title=tit, 
                    save=save,
                    path_to_save=path_to_save
                )
            elif is_array:
                plot_by_indexes_in_img_array(
                    img_array,
                    idx=idx_cls0, 
                    title=tit,
                    save=save,
                    path_to_save=path_to_save
                )
        else:
            print "empty indexes for cluster {}".format(l)
        
def plot_by_indexes_in_img_array(img_array, 
                                 idx, 
                                 title='',
                                 save=False,
                                 path_to_save=None
                                ):
    
       
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(len(idx))))
    #print "num idx {}, rows: {}".format(len(idx), n)
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title)
    pidx=1
    for i in idx:
            
        sub1 = plt.subplot(n,n,pidx)        
        sub1.set_xticks(())
        sub1.set_yticks(())        
        plt.imshow(img_array[i])
        pidx+=1
    #fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    if save:
        print "saving in {}".format(path_to_save+'/img_'+title+'.jpg')
        du.mkdirs(path_to_save)
        plt.savefig((path_to_save+'/img_'+title+'.jpg')) 
        plt.close('all')
        
    
def plot_random_img_array(img_array, num, title=''):
    
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(num)))
    n = min(n, int(np.ceil(np.sqrt(len(img_array)))))
    
    idx = np.random.randint(0,len(img_array),n*n)    
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title)
    pidx=1
    for i in idx:
            
        sub1 = plt.subplot(n,n,pidx)        
        sub1.set_xticks(())
        sub1.set_yticks(())        
        plt.imshow(img_array[i])
        pidx+=1
    #fig.subplots_adjust(hspace=0)
    fig.tight_layout()

def count_img_sizes(img_path_lists):
    sizes = [PIL.Image.open(i).size for i in img_path_lists]
    id2size = list(set(sizes))
    size2id = {o:i for i,o in enumerate(id2size)}
    return sizes, id2size, size2id, collections.Counter(sizes)


def add_to_imgsbysize(img, images_by_size, sz_counter):    
    for s in sz_counter.iterkeys():    
        if img.shape[0]==s[1] and img.shape[1]==s[0]:
            if not images_by_size.has_key(s):
                images_by_size[s] = []
            images_by_size[s].append(img)     
    return images_by_size

def feats_for_imgs(p_imgs, 
                   read=True, 
                   hue=False, 
                   hog=False, 
                   sizes=False,
                   pil_stats=False,
                   save=False, 
                   save_path=None, 
                   class_name=None,
                   cluster=False,
                   n_clust_intra_size=None,
                   save_fig=False,
                   save_cluster_path=None,
                   random_state=0
                  ):
    
    #from cs231n.features import *
    from skimage.feature import hog
    
    if save and (save_path is None or class_name is None):
        print "if save is True, save_path and class_name must be specified"
        return

    if cluster and n_clust_intra_size is None:
        print "if clusters, specify number of n_clust_intra_size"
        return
        
    if cluster and (save_fig == False or save_cluster_path == None):
        print "if save_fig of clusters, save_cluster_path must be specified"
        return
    
    hists_back = []
    hists_back_hue = []
    hogs = []
    images_by_size = {}
    stats = []
    counter = []
 
    if sizes or cluster:
        sizes, id2size, size2id, counter = count_img_sizes(p_imgs)
        
    for imp in p_imgs: 
        if read:
            im = cv2.imread(imp)
            im2rgb = im[:,:,(2,1,0)]
            hists_back.append(plot_rgb_hist(imp, read=True, plot=False))
            gray_img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im2rgb = imp
            hists_back.append(plot_rgb_hist(im2rgb, read=False, plot=False))
            gray_img=cv2.cvtColor(im2rgb, cv2.COLOR_RGB2GRAY)
        
        if hue:
            hists_back_hue.append(color_histogram_hsv(im2rgb))
        
        if hog:
            #hogs.append(hog_feature(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
            fd, hog_image = hog(gray_img, \
                            orientations=8, pixels_per_cell=(16, 16), \
                            cells_per_block=(1, 1), visualise=True)
            hogs.append(hog_image.ravel().shape)
        
        if sizes:
            images_by_size = add_to_imgsbysize(im2rgb, images_by_size, counter)
            
        if pil_stats:    
            stats.append(PIL.ImageStat.Stat(Image.open((imp))))
    
    hists_arr = np.asarray(hists_back)
    hists_arr_hue = np.asarray(hists_back_hue)
    hogs = np.asarray(hogs)
    all_feats = []
    all_kmeans = []
    if hue and hog: 
        all_feats = np.hstack([hists_arr, hists_arr_hue, hogs])
    
    if cluster:
        #print "doing cluster for sizes {} counter: {}".format(images_by_size, counter)
        for k,val in images_by_size.iteritems():
            path_to_save = save_cluster_path+str(k)
            imgs_1 = images_by_size[k]
            info_fish,_= feats_for_imgs(p_imgs=imgs_1, read=False)
            hists1 = info_fish[class_name]['histograms']
            n_cluster = min(len(hist1), n_clust_intra_size)
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit_predict(hists1)
            plot_all_clusters(klabes=kmeans, 
                              is_path=False, 
                              is_array=True, 
                              img_array=imgs_1, 
                              save=save_fig, 
                              path_to_save=path_to_save
                            )
            all_kmeans.append(kmeans)
        
    
    info_img_fish={}
    info_img_fish[class_name] = {}
    info_img_fish[class_name]['hogs'] = hogs
    info_img_fish[class_name]['histograms'] = hists_arr
    info_img_fish[class_name]['hues'] = hists_arr_hue
    info_img_fish[class_name]['sizes'] = sizes
    info_img_fish[class_name]['sizes_counter'] = counter  
    info_img_fish[class_name]['img_by_size'] = images_by_size
    info_img_fish[class_name]['stats'] = stats
    info_img_fish[class_name]['all_kmeans'] = all_kmeans
      
    path_ = '' 
    if save: 
       path_=save_path+'_{}_info_imgs_.pkl'.format(class_name)
       du.mkdirs(save_path)
       with open(path_, 'w+') as f:
           cPickle.dump(info_img_fish, f)
              
    return info_img_fish, path_


def check_imgs_list_and_rm_empty(glob_path_imgs):
    """
    iterate over image list and if empty file remove it
    """
    new_path = []
    for i in glob_path_imgs:
        image_ok = True
        try:
            PIL.Image.open(i)
        except:
            image_ok=False
            print "error with img {}, remove it".format(i)
            os.remove(i)
        if image_ok:
           new_path.append(i)
 
    return new_path


def features_from_path(path_imgs, c_class, dest_path):
  p_img = glob(path_imgs+'/'+c_class+"/*.jpg")
  p_img = check_imgs_list_and_rm_empty(p_img)
 
  info_fish, path_pkl = feats_for_imgs(
     p_img,
     hue=True,
     hog=True,
     sizes=True,
     pil_stats=True,
     save=True,
     save_path=dest_path,
     class_name=c_class,
     cluster=False,
     n_clust_intra_size=0,
     save_fig=False,
     save_cluster_path=None,
     random_state=0
  )
  
  return info_fish, path_pkl

def read_pkl(path_feat):
  with open(path_feat,'r') as f:
      return cPickle.load(f)  
  
def kmeans_from_features(class_name, n_clusters, features=None,path_pkl=None, random_state=0):

  if features is None and path_pkl is None:
    print "either features or path_pkl must be specified"
    return
  
  if features is None:  
    features = read_pkl(path_pkl)
  hists1=features[class_name]['histograms'] 
  kmeans=KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(hists1)    
  return kmeans
  
def check_is_folder_path(p):
    return len(p.split('/')[-1].split('.'))==1

def cluster_imgs_to_folder(p_imgs, class_name, n_clusters, dest_path, features=None,path_pkl=None):
  """
  p_imgs path from which feature pkl was generated
  """  
  klabels = kmeans_from_features(class_name, n_clusters, features=features,path_pkl=path_pkl, random_state=0)

  # if p_imgs is a path to a folder, glob it
  if check_is_folder_path(p_imgs):
    p_imgs = glob(p_imgs+'/'+class_name+"/*.jpg")   

  n_clusters = np.sort(np.unique(klabels))
  for l,lab in enumerate(n_clusters):        
      tit = "saving for cluster {}".format(l)
      idx_cls0 = ClusterIndicesNumpy(l, klabels)
      if(len(idx_cls0)>0):
          for idx in idx_cls0:
            path_to_save=dest_path+'/clusters/'+class_name+'/clust_'+str(l)+'/'
            du.mkdirs(path_to_save)
            img_name = p_imgs[idx].split('/')[-1]    
            path_=path_to_save+'/'+img_name
            print ("saving in {}").format(path_)
            im = cv2.imread(p_imgs[idx])
            cv2.imwrite(path_, im)
  
      else:
          print "empty indexes for cluster {}".format(l)
  
    

# from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def try_ncluster_kmeans(class_name, path_pkl, n_clusters_to_try):

  features = read_pkl(path_pkl)
  
  from sklearn.metrics import silhouette_samples, silhouette_score
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm

  
  #range_n_clusters = xrange(n_clusters_from_to[0], n_clusters_from_to[1])
  X=features[class_name]['histograms'] 

  for n_clusters in n_clusters_to_try:
      # Create a subplot with 1 row and 2 columns
      fig, (ax1, ax2) = plt.subplots(1, 2)
      fig.set_size_inches(18, 7)

      # The 1st subplot is the silhouette plot
      # The silhouette coefficient can range from -1, 1 but in this example all
      # lie within [-0.1, 1]
      ax1.set_xlim([-0.1, 1])
      # The (n_clusters+1)*10 is for inserting blank space between silhouette
      # plots of individual clusters, to demarcate them clearly.
      ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

      # Initialize the clusterer with n_clusters value and a random generator
      # seed of 10 for reproducibility.
      clusterer = KMeans(n_clusters=n_clusters, random_state=10)
      cluster_labels = clusterer.fit_predict(X)

      # The silhouette_score gives the average value for all the samples.
      # This gives a perspective into the density and separation of the formed
      # clusters
      silhouette_avg = silhouette_score(X, cluster_labels)
      print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

      # Compute the silhouette scores for each sample
      sample_silhouette_values = silhouette_samples(X, cluster_labels)

      y_lower = 10
      for i in range(n_clusters):
          # Aggregate the silhouette scores for samples belonging to
          # cluster i, and sort them
          ith_cluster_silhouette_values = \
              sample_silhouette_values[cluster_labels == i]

          ith_cluster_silhouette_values.sort()

          size_cluster_i = ith_cluster_silhouette_values.shape[0]
          y_upper = y_lower + size_cluster_i

          color = cm.spectral(float(i) / n_clusters)
          ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

          # Label the silhouette plots with their cluster numbers at the middle
          ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

          # Compute the new y_lower for next plot
          y_lower = y_upper + 10  # 10 for the 0 samples

      ax1.set_title("The silhouette plot for the various clusters.")
      ax1.set_xlabel("The silhouette coefficient values")
      ax1.set_ylabel("Cluster label")

      # The vertical line for average silhouette score of all the values
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

      ax1.set_yticks([])  # Clear the yaxis labels / ticks
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
      
      plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

  plt.show()
    
    




if __name__ == '__main__':
                    
    path_test_imgs = "/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/data/fish/test_stg1"          
    #p_img = glob(TRAIN_ORIG_PATH+"/LAG/*.jpg")
    p_img = glob(path_test_imgs+"/*.jpg")
    """feats_for_imgs(
       p_img,
       hue=False, 
       hog=False, 
       sizes=False,
       pil_stats=True,
       save=True, 
       save_path=TRAIN_ORIG_PATH+"/../stat_img/test_stg1/", 
       class_name='test_stg1',
       cluster=True,
       n_clust_intra_size=4,
       save_fig=True,
       save_cluster_path=TRAIN_ORIG_PATH+"/../stat_img/test_stg1/clusters",
       random_state=0
   )"""


