import os, sys, argparse
from xml_utils import *
from faster_rcnn_utils import *
from shutil import copyfile

def add_img_to_ds(im_name, root_path, dataset_name, new_name, anno_p, out):
    """

    :param im_p:
    :return:
    """
    print "add to ds {}".format(im_name)
    path_dest_images = os.path.join(
        out,
        new_name,
        'JPEGImages'
    )
    path_dest_anno = os.path.join(
        out,
        new_name,
        'Annotations'
    )
    path_dest_imgset = os.path.join(
        out,
        new_name,
        'ImageSets',
        'Main',
    )
    from_p = os.path.join(
        root_path,
        'JPEGImages'
    )
    copyfile(from_p +'/'+ im_name +'.jpg', path_dest_images +'/'+ im_name + '.jpg')
    copyfile(anno_p, path_dest_anno +'/'+ im_name + '.xml')

    with open(path_dest_imgset + '/' +new_name + '.txt', 'a') as f:
        f.write("%s\n" % im_name)


def extract(classes, n, dataset_name, root_path, new_name, class_field='class', count_from=1, only_class=None, out='/tmp'):
    """
    voc style data has structure
    root/Annotations/img_[i].xml
    root/ImageSets/Main/name.txt
    root/JPEGImages/img_[i].jpg

    :param classes:
    :param n:
    :param dataset_name:
    :param root_path:
    :param out:
    :return:
    """


    ##
    # get all images from imageset txt file
    # for every image
    # get xml and class from name annotation field
    # get the images for every class until n or no more images for that class is found


    fu = faster_rcnn_utils()
    fu.create_base_folders(
        out, new_name
    )

    class_to_ind = du.array_to_ind(classes)

    img_set_path = os.path.join(
        root_path,
        'ImageSets',
        'Main',
        dataset_name + '.txt'
    )
    with open(img_set_path, 'r') as f:
        list_images = f.readlines()

    list_images = [x.strip() for x in list_images]

    print "read {} images".format(len(list_images))

    counters = {}
    full_classes = {}

    if count_from>1:
        list_images = list_images[count_from:]

    i=0
    for im in list_images:
        ann_p = os.path.join(
            root_path,
            'Annotations',
            im + '.xml'
        )

        tree = parse_file(ann_p)
        mism, img_name, img_with_ext = get_and_check_img_name(tree, ann_p)

        # if there are more than one object get the first and associate to its class
        objs = fu.parse_obj(tree, len(classes), \
                            class_to_ind, use_diff=False, minus_one=False)
        class_ = objs['gt_classes'][0]

        if only_class is not None:
            if class_ != only_class:
                continue

        if full_classes.has_key(class_) and len(full_classes)==len(classes):
            print "all done, exit"
            break

        if not counters.has_key(class_):
            counters[class_] = 1;
            add_img_to_ds(im, root_path=root_path, dataset_name=dataset_name, anno_p=ann_p, new_name=new_name, out=out)
        else:
            if counters[class_] < n:
                add_img_to_ds(im, root_path=root_path, dataset_name=dataset_name, anno_p=ann_p, new_name=new_name, out=out)
                counters[class_] += 1
            else:
                if not full_classes.has_key(class_):
                    print "completed for class {}, n: {}".format(class_, counters[class_])
                    full_classes[class_] = True


def parse_args():

    parser = argparse.ArgumentParser(description='Extract n element for each class of a voc style dataset, and create new voc style dataset')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    """Parse input arguments."""

    parser.add_argument('--path-ds', dest='path_ds', help='path of the dataset to extrat images from')
    parser.add_argument('--dataset_name', dest='name_ds', help='path of the dataset to extrat images from')
    parser.add_argument('--out-dir', dest='out_dir', default='/tmp',help='path of the dataset to extrat images from')
    parser.add_argument('--n', dest='n_images', help='path of the dataset to extrat images from')

    args = parser.parse_args()

    return args

from glob import glob

if __name__ == '__main__':

    #args = parse_args()

    classes = (
               'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft'
    )

    root = ''
    extract(
        classes, 50, 'validfromtrain',  root, new_name='validfromtrainmini50'
    )
