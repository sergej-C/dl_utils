import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile, copy
from glob import glob
import numpy as np

def test():
    print 'test'

def mkdir_ifnotexists(path):
    """
    create a folder with the specified path if not exists
    """
    if not os.path.exists(path): 
        os.mkdir(path)

def ls(path, only_files=False):
    if only_files==True:
        fs = [f for f in listdir(path) if isfile(join(path, f))]
    else:
        fs = [f for f in listdir(path)]

    print fs
    return fs

def create_sample_dirs(competition_path):
    current_dir = os.getcwd()
    os.chdir(competition_path)
    
    # Create directories    
    mkdirs('data/valid') 
    mkdirs('results')
    mkdirs('data/sample/train')
    mkdirs('data/sample/test')
    mkdirs('data/sample/valid')
    mkdirs('data/sample/results')
    mkdirs('test/unknown')
    
    # back to cwd
    os.chdir(current_dir)

def mkdirs(directory_name):
    try:
        os.makedirs(directory_name)
    except OSError as err:
        if err.errno!=17:
            raise

def cp_random_file_to_dir(from_dir, to_dir, size=-1, ext='jpg', move=False):
    current_dir = os.getcwd()
    print ("cwd: "+current_dir)

    os.chdir(from_dir)
    print ("going to :"+from_dir)
    print ("copy to :"+to_dir)
    g = glob('*.'+ext)
    if size==-1:
        size=len(g)        
    shuf = np.random.permutation(g)
    if move==False:
        for i in range(size): copyfile(shuf[i], to_dir + '/' + shuf[i])
    else:
        for i in range(size): os.rename(shuf[i], to_dir + '/' + shuf[i])
    os.chdir(current_dir)
    
    
def create_sample_sets(data_path, val_size, train_sample_size, val_sample_size):
    """
    create validation set and training + validation sample sets
    training data is in data_path/train
    validation data is taken from data_path/train
    data_path/valid is created
    data_path/sample/train and sample/valid are created
    """
    create_sample_dirs(data_path + '/..')
    
    # move data from training to validation dir
    cp_random_file_to_dir(data_path+ '/train', data_path+'/valid', move=True, size=val_size )
    
    # create a sample for training from train dir 
    cp_random_file_to_dir(data_path+ '/train',  data_path+'/sample/train/', move=False, size=train_sample_size )
    
    # create a sample for validation from valid dir
    cp_random_file_to_dir(data_path+ '/valid',  data_path+'/sample/valid/', move=False, size=val_sample_size )

def mvfiles(expr, to_dir):
 for file in glob(expr):
    copy(file, to_dir)    

def create_dir_for_cat(cat, path, ext='jpg'):
    """
    create a folder with name cat + 's' and puts them all files with name cat.*.ext taken from path
    """
    os.chdir(path)        
    mkdirs(cat + 's')
    mvfiles(cat+'.*.'+ext, cat+'s')
    
def create_categories_folders(data_path, categories, ext='jpg'):
    """
    create folder for every category in categories (i.e. [cat, dog])
    in data_path/sample/train sample/valid valid train
    puts all files from these directories to the new ones
    files are search with wildcard such that the prefix is equal to the category name
    i.e. cat.*.ext to cats folder...
    """
    current_dir = os.getcwd()
    for c in categories:
        create_dir_for_cat(c, data_path + '/sample/train', ext=ext)        
        create_dir_for_cat(c, data_path + '/sample/valid', ext=ext)    
        create_dir_for_cat(c, data_path + '/valid', ext=ext)    
        create_dir_for_cat(c, data_path + '/train', ext=ext)    
    
    os.chdir(data_path + '/test')
    mvfiles('*.'+ext, 'unknown/')    
    os.chdir(current_dir)

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]
    
