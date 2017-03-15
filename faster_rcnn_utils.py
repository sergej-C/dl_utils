import ujson as json
import cPickle
import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis
import os, sys
from glob import glob
from shutil import copyfile
import scipy.sparse
import scipy.io as sio
import ujson as json

import xml.etree.ElementTree as ET
import pprint


# configure plotting
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

import data_utils as du
#import dlc_utils as dlcu
KAGGLE_PATH=os.environ.get('KAGGLE_PATH')
FISH_DATA_PATH=KAGGLE_PATH+'/fish'
FASTER_RCNN_PATH = os.environ.get('PYFASTER_PATH')
FASTER_RCNN_TOOLS_PATH = FASTER_RCNN_PATH + 'tools'
FASTER_RCNN_LIB_PATH = FASTER_RCNN_PATH + 'lib'

class faster_rcnn_utils():

        
    # some utils
    #sys.path.append(KAGGLE_PATH+'/dl_utils')


    def __init__(self):
        print "init faster_rcnn_utils"

    def get_root_folder(self):
        return FASTER_RCNN_PATH

    def get_data_folder(self, project_name=''):
        return os.path.join(
            self.get_root_folder(),
            'data',
            project_name
        )

    def create_base_folders(self, path_dataset, dataset_name):
        # create folders structure for use with caffe faster-rcnn
        du.mkdirs(path_dataset)
        path = path_dataset+'/' +dataset_name
        du.mkdirs(path)
        
        path_dest_annotations = path+'/Annotations'
        path_dest_list = path+'/ImageSets/Main'
        path_dest_images = path+'/JPEGImages'
        du.mkdirs(path_dest_annotations)
        du.mkdirs(path_dest_list)
        du.mkdirs(path_dest_images)
        
        return path_dest_annotations, path_dest_list, path_dest_images

    ######################################################################
    ## create an xml file with name of image, and on object for each bbox
    #
    # presume no more than one class object for image
    # (only more than one instance of the same class)
    #
    # classes_for_forlder=None, #if none use classes
    # no_classes_for_images=False, # if True get images path from orig_image_path/*.jpg else 
    # from orig_image_path/[cat or CAT is use_upper_case]/*.jpg
    # use_upper_case_classes=True #if true use classes uppercase 
    # EXAMPLE USAGE
    # import sys, os

    # KAGGLE_PATH=os.environ.get('KAGGLE_PATH')
    # sys.path.append(KAGGLE_PATH+'/dl_utils')
    # import data_utils, dlc_utils
    # from faster_rcnn_utils import faster_rcnn_utils
    #                            
    # fu = faster_rcnn_utils()        
    # CLASSES = ('__background__',  # always index 0, class NoF as background ?
    #                          'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft'
    #                          )

    # class_to_ind = dict(zip(CLASSES, xrange(len(CLASSES))))                           

    # # where there is my train folder (from original datasets downloaded from Kaggle)
    # PATH_FISH_ORIG=KAGGLE_PATH+'/fish'
    # TRAIN_ORIG_PATH=PATH_FISH_ORIG+'/train'

    # # where to save new created dataset, created if doesnt exists 
    # PATH_NEW_DATASETS = PATH_FISH_ORIG+'/for_faster_rcnn1'

    # # orig annotations file path
    # annotation_path = PATH_FISH_ORIG+'/Annotations/'
    #                            
    # fu.json_to_xml_imdb(
    #     PATH_NEW_DATASETS, 
    #     annotation_path, 
    #     TRAIN_ORIG_PATH,
    #     CLASSES[1:],                     
    #     dataset_name='train',
    #     classes_for_forlder=None,
    #     no_classes_for_images=False,
    #     use_upper_case_classes=True
    # )          
    #####################################################################
    def json_to_xml_imdb(self,
                         destination_folder, 
                         annotation_path, 
                         orig_image_path, 
                         classes,                     
                         dataset_name='train',
                         classes_for_forlder=None,
                         no_classes_for_images=False,
                         use_upper_case_classes=False
                        ):
            
        path_dest_annotations, path_dest_list, path_dest_images = \
            self.create_base_folders(destination_folder, dataset_name)
        
        bb_json = {}

        _num_tot_annotations = 0

        dict_images = {}

        # skip background
        
        template_xml_file = '{:s}.xml'
        
        image_index_filename = dataset_name
        img_list_dest = "{}/".format(path_dest_list) + image_index_filename + '.txt'
        img_list = []

        nc=0
        for c in classes:

            j = json.load(open('{}/{}_labels.json'.format(annotation_path, c), 'r'))

            for l in j:

                ann_xml, img_name = self.fish_to_voc(l, c)                                
                
                if ann_xml is not None:
                    
                    img_list.append(img_name[:-4])
                    
                    _num_tot_annotations += 1
                    
                    # remove extension only 4 lenght extension (.jpg)
                    self.save_tree(ann_xml, "{}/".format(path_dest_annotations)+template_xml_file.format(img_name[:-4]))
                    
                    # copy image
                    if no_classes_for_images:
                        orig_p = orig_image_path
                    elif classes_for_forlder is not None:                    
                        orig_p = orig_image_path + "/{}/".format(classes_for_forlder[nc])
                    else:    
                        cl = classes[nc]
                        if use_upper_case_classes:
                            cl = cl.upper()                    
                            
                        orig_p = orig_image_path + "/{}/".format(cl)
                        
                    copyfile(orig_p + img_name, "{}/".format(path_dest_images) + img_name)
                else:
                    print "problem with {}, no xml created".format(img_name)

                if _num_tot_annotations % 100 == 0:
                    print "done {}".format(_num_tot_annotations)
            nc+=1
            
        # save list of images                
        thefile = open(img_list_dest, 'a')
        for item in img_list:
              thefile.write("%s\n" % item)

    def create_xml(self, root_node_name):
        return ET.Element(root_node_name)

        
    def save_tree(self, root_el, filename_no_ext, add_ext=False):
        tree = ET.ElementTree(root_el)
        if add_ext:
            fn = filename_no_ext+".xml"
        else:
            fn = filename_no_ext
        tree.write(fn)
        

    def add_file_name(self, anno_root, image_name):
        el = ET.SubElement(anno_root, "filename").text = image_name
        return el

    #       <object>
    #               <name>boat</name>
    #               <pose>Unspecified</pose>
    #               <truncated>0</truncated>
    #               <difficult>1</difficult>
    #               <bndbox>
    #                       <xmin>440</xmin>
    #                       <ymin>226</ymin>
    #                       <xmax>455</xmax>
    #                       <ymax>261</ymax>
    #               </bndbox>
    #       </object>
    def add_node_to_annotation(self, anno_root, class_name, pose='Unspecified', difficult="0", xmin="0", ymin="0", xmax="0", ymax="0"):
            
        obj = ET.SubElement(anno_root, "object")
        cln = ET.SubElement(obj, "name").text = class_name
        pose = ET.SubElement(obj, "pose").text = pose
        difficult = ET.SubElement(obj, "difficult").text = difficult
        bbox = ET.SubElement(obj, "bndbox") 
        ET.SubElement(bbox, "xmin").text = xmin 
        ET.SubElement(bbox, "ymin").text = ymin
        ET.SubElement(bbox, "xmax").text = xmax
        ET.SubElement(bbox, "ymax").text = ymax
        return obj

    def create_annotation(self):
        return self.create_xml("annotation")

    # VOC2007 XML STYLE
    # xml file, one for each image, one object for every subject of the class if present.
    # faster-rcnn only use object key
    #<annotation>
    #       <folder>VOC2007</folder>
    #       <filename>000080.jpg</filename>
    #       <size>
    #               <width>500</width>
    #               <height>375</height>
    #               <depth>3</depth>
    #       </size>
    #       <object>
    #               <name>boat</name>
    #               <pose>Unspecified</pose>
    #               <truncated>0</truncated>
    #               <difficult>1</difficult>
    #               <bndbox>
    #                       <xmin>440</xmin>
    #                       <ymin>226</ymin>
    #                       <xmax>455</xmax>
    #                       <ymax>261</ymax>
    #               </bndbox>
    #       </object>
    #</annotation>


    # KAGGLE FISH ANNOTATION STYLE
    # from https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/25902
    # json file [sloth], one for each class, there may be more than one bbox (fish) for image
    # {
    #        "annotations": [
    #            {
    #                "class": "rect",
    #                "height": 65.00000000000023,
    #                "width": 166.00000000000063,
    #                "x": 469.00000000000165,
    #                "y": 448.0000000000016
    #            },
    #            {
    #                "class": "rect",
    #                "height": 143.0000000000005,
    #                "width": 98.00000000000036,
    #                "x": 92.00000000000033,
    #                "y": 495.00000000000176
    #            }
    #        ],
    #        "class": "image",
    #        "filename": "img_07915.jpg"
    #  },

    # bbox coordinate in VOC xmin, ymin, xmax, ymax
    # bbox coordinate in Fish xmin, ymin, width, height

    @staticmethod
    def convert_width_to_x2(x1, w):
        return x1 + w

    @staticmethod
    def convert_height_to_y2(y1, h):
        return y1 + h

    @staticmethod
    def convert_points_from_json_to_roi(bb_json_el):

        bb_params = ['height', 'width', 'x', 'y']
        # gt_roidb = []
        # Load object bounding boxes into a data frame.
        # for ix, bb in enumerate(bb_json):
        bbox = [bb_json_el[p] for p in bb_params]

        x1 = float(bbox[2]) - 1
        y1 = float(bbox[3]) - 1
        # annotations are h,w,x,y we want-> x1,y1,x2,y2
        x2 = faster_rcnn_utils.convert_width_to_x2(x1, float(bbox[1]) - 1)
        y2 = faster_rcnn_utils.convert_height_to_y2(y1, float(bbox[0]) - 1)
        return x1, y1, x2, y2

    @staticmethod
    def parse_obj(tree, num_classes, class_to_ind, use_diff=True, minus_one=False):
        """
        get bboxes and compute area for each one, gets overlaps and classes too
        minus_one for Imagenet dataset annotations
        from original pascal_voc imdb class in py-faster-rcnn
        """
        objs = tree.findall('object')
        if not use_diff:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        minus =  (1 if minus_one else 0)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - minus
            y1 = float(bbox.find('ymin').text) - minus
            x2 = float(bbox.find('xmax').text) - minus
            y2 = float(bbox.find('ymax').text) - minus

            cls_name=obj.find('name').text.lower().strip()

            #if cls_name is None:
            #    cls_name = class_to_ind[obj.find('class').text.lower().strip()]
            cls = class_to_ind[cls_name]
        
            #print "found class {} cls2ind {}".format(cls_name, cls)

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
        
    def fish_to_voc(self, fish_json_obj, class_name):
        
        ann_xml = None
        
        img_name = fish_json_obj['filename']
        
        if "/" in img_name:
            img_name = img_name.split('/')[-1]
            

        # search for annotations        
        key_box = 'annotations'

        l = fish_json_obj
        
        if key_box in l.keys() and len(l[key_box]) > 0:
            annotations = fish_json_obj['annotations']
            
            ann_xml = self.create_annotation()
            self.add_file_name(ann_xml, img_name)
            
            for obj in annotations:
                x1, y1, x2, y2 = faster_rcnn_utils.convert_points_from_json_to_roi(obj)                        

                self.add_node_to_annotation(ann_xml, class_name, 
                    xmin=str(x1), ymin=str(y1), xmax=str(x2), ymax=str(y2))
        
        return ann_xml, img_name    


