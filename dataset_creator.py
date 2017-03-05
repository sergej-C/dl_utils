import os

import img_utils as imu
import range_utils
from iter_utils import CombinatorialTransformer, TransformationTracer
from preprocessing import *
from data_struct_utils import *
from xml_utils import *
from io_utils import *
import faster_rcnn_utils as fast_ut
import db_record as db_rec
import db_utils
from structured import dict2xml
import gc

class DatasetCreator():
    """
    we have some modified pictures [PNG with transparent areas around the subject] we want to use as foreground
    we want to insert this pictures into a list of background images

    we want to apply some transformations on both

    """

    def __init__(self,
                 name,
                 base_path_background_glob,
                 base_path_foreground_glob,
                 list_transformation_background,
                 list_transformation_foreground,
                 list_transformation_to_call_back,
                 list_transformation_to_call_foreg,
                 output_folder_path,
                 choose_n_random_background=-1,
                 random_background_for_all=False,
                 shuffle=True,
                 blur_merge=True,
                 base_path_background_points_xml=None,
                 class_names=None,
                 db_conf=None,
                 debug=False

                 ):

        self.DEBUG = debug

        #
        # dataset name
        self.name = name

        #
        # the list of images to use as foregrounds
        self.list_foreground = []

        #
        # the list of images to use as backgrounds
        self.list_background = []


        #
        # list of functions and parameters to apply on foreground images
        # see format in CombinatorialTransformer
        self.list_transformation_foreground = list_transformation_foreground
        self.list_transformation_to_call_foreg = list_transformation_to_call_foreg

        #
        # list of functions and parameters to apply on background images
        # before and after background and foreground are merged
        #
        # for pre_merge and post_merge see format in CombinatorialTransformer
        #list_transformation_background = {
        #
        #    'pre_merge': {},
        #
        #    'post_merge': {}
        #}
        self.list_transformation_background = list_transformation_background
        self.list_transformation_to_call_back = list_transformation_to_call_back

        #
        # where there are the background images, in glob style text i.e. /path/to/*/*.jpg
        self.base_path_backgrounds_glob = base_path_background_glob

        #
        # where there are the foreground images
        # if class_names are defined, iterate over them and
        # load images from base_path_foreground, in glob style text i.e. /path/to/*/*.png
        self.base_path_foreground_glob = base_path_foreground_glob

        #
        # if not none we search xml  file for every background image
        # with annotation of points in which to insert the foreground
        # there could be more than one point, but only in one (randomly chosen)
        # is selected [TODO - multi foreground in background, multi class per background]
        #
        # if none, one random point inside the background image is randomly selected
        self.base_path_background_points_xml = base_path_background_points_xml


        #
        # where there are the voc style annotations (one xml for each image)
        # of the original image (the image from which the foreground was extracted)
        # if for example the name of foreground is img_0000_1.pbg the original
        # image name is img_0000.jpg [bbox at index 1, 2th]
        #
        # if it is None no annotation where taken
        # otherwise is merged with new annotations
        #self.base_path_foreground_xml = base_path_foreground_xml

        #
        #
        self.class_names = class_names

        #
        # where to save the dataset
        self.output_folder_path = output_folder_path

        #
        # if it is not None, i will create a mysql connection with the db specified
        # to get additional annotations from the specified table name to merge with new annotations
        # defined with this structure

        """
        -- phpMyAdmin SQL Dump
        -- version 4.0.10deb1
        -- http://www.phpmyadmin.net
        --
        -- Host: localhost
        -- Generato il: Feb 24, 2017 alle 11:54
        -- Versione del server: 5.5.54-0ubuntu0.14.04.1
        -- Versione PHP: 5.5.9-1ubuntu4.21

        SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
        SET time_zone = "+00:00";

        --
        -- Database: `fish`
        --

        -- --------------------------------------------------------

        --
        -- Table structure `bbox_selection`
        --

        CREATE TABLE IF NOT EXISTS `bbox_selection` (
          `bs_id` int(11) NOT NULL AUTO_INCREMENT,
          `bs_img_name` varchar(255) NOT NULL,
          `bs_difficult` tinyint(1) NOT NULL,
          `bs_pose_values` enum('top','bottom','lateral','lateral_top','lateral_bottom') DEFAULT NULL,
          `bs_sizes_values` enum('little','medium','big') DEFAULT NULL,
          `bs_body_vis` enum('all','no_head','no_tail','body_only') DEFAULT NULL,
          `bs_occlusion` enum('none','minimal','big') DEFAULT NULL,
          `bs_class` enum('LAG','ALB','YTN','BET','DOL','OTHER','SHARK') DEFAULT NULL,
          `bs_color_mask` enum('normal','green','blue','red','dazzling') DEFAULT 'normal',
          `bs_bleeding` tinyint(1) DEFAULT '0',
          PRIMARY KEY (`bs_id`),
          UNIQUE KEY `bs_img_name` (`bs_img_name`),
          UNIQUE KEY `bs_img_name_2` (`bs_img_name`)
        ) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=103 ;

        """
        self.db_conf = db_conf
        self.db_conn = None
        self.dbu = None

        #
        # final pil images and tracers
        self.final_tracers = []
        self.final_pil_images = []
        self.final_names = []
        self.final_bboxes = []

        #
        # if True shuffle the list of final images
        self.shuffle = shuffle

        #
        # if True use opencv poisson method for merging background and foreground images
        self.blur_merge = blur_merge

        ##
        # if True random background for all foreground
        self.random_background_for_all = random_background_for_all

        ##
        # if != -1 choose n random backgrounds from the list of backgrounds
        # valid only if random_background_for_all is False
        self.choose_n_random_background = choose_n_random_background

        #
        # utility class for parsing voc style xml
        self.fast_ut = fast_ut.faster_rcnn_utils()

        ##
        # number of final saved images
        self.img_counter = 0

        ##
        # used to iterate over all or some maybe random background
        self.curr_back_img_list = None


    def choose_point(self, back_pil, fore_pil):
        """
        get points randomly or from xml annotation file associate with
        back_pil (PIL Image)

        :param back_pil: pil background image
        :param fore_pil: pil foreground image
        :return:
        """

        # TODO - get points from xml annotation file
        annotated_points = []
        l_annotated = len(annotated_points)
        if l_annotated>0:
            if l_annotated==1:
                return annotated_points[0]
            else:
                rnd_id = np.random.randint(0, l_annotated, 1)
                return annotated_points[rnd_id]
        else:
            wb,hb = back_pil.size
            wf,hf = fore_pil.size
            center_x = range_utils.choice_n_rnd_numbers_from_to_linspace(0, wb-wf, wb-wf ,1)[0]
            center_y = range_utils.choice_n_rnd_numbers_from_to_linspace(0, hb-hf, hb-hf ,1)[0]
            return (center_x, center_y)

    def check_random_list_and_select(self):

        if self.choose_n_random_background != -1 and not self.random_background_for_all:
            self.background_list = np.random.choice(self.background_list, self.choose_n_random_background)

    def check_random_list_and_select_copy(self):
        if self.choose_n_random_background != -1 and self.random_background_for_all:
            return np.random.choice(self.background_list, self.choose_n_random_background)

    ##
    # too high ram usage, use create_random_background_for_foreground
    def create(self):

        self.get_glob_paths()

        self.check_random_list_and_select()

        # every element is a list of Tracer with PIL Image as subject
        transformed_list_foreg = []
        foreg_names = [imp.split('/')[-1] for imp in self.foreground_list]
        all_foreg_names = []

        #
        # apply transformation for all foreground
        # every transformation return a list of modified foregrounds
        foreg_img_list = [preprocessing.open_image(imp) for imp in self.foreground_list]
        for idx, im in enumerate(foreg_img_list):

            foreg_transformer = CombinatorialTransformer(
                                    self.list_transformation_to_call_foreg,
                                    self.list_transformation_foreground,
                                    im)

            foreg_transformer.apply_transformations()
            transformed_list_foreg.append(foreg_transformer.all_items)
            for it in foreg_transformer.all_items:
                all_foreg_names.append(foreg_names[idx])

        if self.DEBUG:
            print "transformed_list_foreg len {}".format(len(transformed_list_foreg))

        #
        # list of transformed background list, one for pre_merge
        transformed_list_back_premerge = []

        #
        # len of pre_merged transformations
        len_premerge_transformation = len(self.list_transformation_to_call_back['pre_merge'])

        #
        # list of background PIL images
        back_img_list = [preprocessing.open_image(imp) for imp in self.background_list]
        all_bkgr_names = []
        backgr_names = [imp.split('/')[-1] for imp in self.background_list]

        #
        # pre_merge transformation if exists
        if len_premerge_transformation>0:

            for idx, im in enumerate(back_img_list):

                back_transformer = CombinatorialTransformer(
                                    self.list_transformation_to_call_back['pre_merge'],
                                    self.list_transformation_background['pre_merge'],
                                    im
                                    )
                back_transformer.apply_transformations()
                transformed_list_back_premerge.append(back_transformer.all_items)
                for it in back_transformer.all_items:
                    all_bkgr_names.append(backgr_names[idx])

        #
        # if there was pre_merge transformation merge foreground with those new background
        # otherwise merge with original background
        list_back_to_merge = []
        pre_merge_done=False
        if len(transformed_list_back_premerge) > 0:
            pre_merge_done=True
            for back_list in transformed_list_back_premerge:
                for back in back_list:
                    list_back_to_merge.append(back)
            del transformed_list_back_premerge
            del back_img_list
        else:
            list_back_to_merge = back_img_list
            all_bkgr_names = backgr_names

        #
        # from list of list to list
        all_foreg_list = []
        for list_foreg in transformed_list_foreg:
            for el in list_foreg:
                all_foreg_list.append(el)
        del transformed_list_foreg

        #
        # merging
        all_merged = []
        all_boxes = []
        all_tracers = []
        all_names = []
        for idb, back in enumerate(list_back_to_merge):
            for idf, modified_foreg in enumerate(all_foreg_list):

                #
                # if pre_merge is not done back is a PIL Image otherwise is a Tracer
                if pre_merge_done:
                    back_obj=back.obj
                    back_tracer=back
                else:
                    back_obj=back
                    back_tracer=TransformationTracer(back)

                preprocessing.check_merging_size(
                    back_obj,
                    modified_foreg.obj
                )

                chosen_point = self.choose_point(back_obj, modified_foreg.obj)                
                if self.DEBUG:
                    print "chosen point {}".format(chosen_point)

                pil_merged, bboxes = (
                    preprocessing.merge_img_in_background(

                        back_obj,
                        modified_foreg.obj,
                        [chosen_point],
                        self.blur_merge
                    )
                )

                all_merged.append(pil_merged[0])
                all_boxes.append(bboxes[0])
                all_tracers.append(TransformationTracer.merge(modified_foreg, back_tracer))
                all_names.append({'foreg':all_foreg_names[idf], 'backgr':all_bkgr_names[idb]})

        #
        # len of post_merged transformations
        len_postmerge_transformation = len(self.list_transformation_to_call_back['post_merge'])


        #
        # append here list of transformed post_merged backgrounds
        all_final_transformed = []
        all_final_bbox = []
        all_final_names = []

        #
        # if there are post_merge transformation become True
        post_merge_done = False

        #
        # if exist do post_merge transformations
        if (len_postmerge_transformation) > 0:

            #
            # for all merged images
            for i,im in enumerate(all_merged):

                    post_merge_done=True

                    back_transformer = CombinatorialTransformer(
                        self.list_transformation_to_call_back['post_merge'],
                        self.list_transformation_background['post_merge'],
                        im,
                        all_tracers[i]
                    )
                    back_transformer.apply_transformations()
                    all_final_transformed.append(back_transformer.all_items)

                    #
                    # all have the same bbox
                    for it in back_transformer.all_items:
                        all_final_bbox.append(all_boxes[i])
                        all_final_names.append(all_names[i])

           
        if post_merge_done:
            del all_merged
            del all_boxes
            del all_tracers
            del all_names

            #
            # to list
            for list in all_final_transformed:
                for tracer in list:
                    self.final_tracers.append(tracer)
                    self.final_pil_images.append(tracer.obj)

            self.final_names = all_final_names
            self.final_bboxes = all_final_bbox
            del all_final_transformed

        else:
            self.final_tracers = all_tracers
            self.final_pil_images = all_merged
            self.final_names = all_names
            self.final_bboxes = all_boxes


        print  "final_tracers len {}, final_pil_images {}, final_names len {}, type tracer[0] {}, type pil[0]".format(
            len(self.final_tracers), len(self.final_pil_images), len(self.final_names),
            type(self.final_tracers[0]), type(self.final_pil_images[0])
        )

        self.save_finals()

    def get_glob_paths(self):
        #
        # get the list of foregrounds and backgrounds
        self.foreground_list = glob(self.base_path_foreground_glob)

        self.background_list = glob(self.base_path_backgrounds_glob)


        # checks all images can be loaded from PIL (if empty file -> removed)
        imu.check_imgs_list_and_rm_empty(self.foreground_list)
        imu.check_imgs_list_and_rm_empty(self.background_list)

        print len(self.foreground_list), len(self.background_list)


    @profile
    def create_random_background_for_foreground(self):
        ##
        # in this case every foreground will be merged to a different random chosen background
        # not append in list but save directly

        self.get_glob_paths()

        self.check_random_list_and_select()

        # every element is a list of Tracer with PIL Image as subject
        foreg_names = [imp.split('/')[-1] for imp in self.foreground_list]
        #foreg_img_list = [preprocessing.open_image(imp) for imp in self.foreground_list]

        ##
        # apply transformation for all foreground
        # every transformation return a list of modified foregrounds
        idx=-1
        for imp in self.foreground_list:

            idx+=1
            im = preprocessing.open_image(imp)
            if len(self.list_transformation_to_call_foreg)>0:

                foreg_transformer = CombinatorialTransformer(
                    self.list_transformation_to_call_foreg,
                    self.list_transformation_foreground,
                    im)

                foreg_transformer.apply_transformations()
                all_foreg = foreg_transformer.all_items
            else:
                all_foreg = [TransformationTracer(im)]

            for transformed_foreg in all_foreg:
                foreg_name = (foreg_names[idx])

                ##
                # for every foreground transformed image choice n random background
                # if option is True else do for all background
                new_back_list = self.check_random_list_and_select_copy()
                random_back = True
                if new_back_list is None:
                    new_back_list = self.background_list
                    random_back = False

                if self.DEBUG:
                    print "using n background {}".format(len(new_back_list))

                #
                # len of pre_merged transformations
                len_premerge_transformation = len(self.list_transformation_to_call_back['pre_merge'])

                #
                # list of background PIL images
                if random_back or self.curr_back_img_list is None:
                    #self.curr_back_img_list = [preprocessing.open_image(imp) for imp in new_back_list]
                    #self.backgr_names = [imp.split('/')[-1] for imp in self.background_list]
                    self.curr_back_img_list = new_back_list

                #
                # pre_merge transformation if exists
                #
                # if there was pre_merge transformation merge foreground with those new background
                # otherwise merge with original background
                bidx=0
                for imp in (self.curr_back_img_list):

                    im = preprocessing.open_image(imp)
                    backgr_name = imp.split('/')[-1]

                    if len_premerge_transformation > 0:

                            back_transformer = CombinatorialTransformer(
                                self.list_transformation_to_call_back['pre_merge'],
                                self.list_transformation_background['pre_merge'],
                                im
                            )

                            back_transformer.apply_transformations()

                            ##
                            # for every background create final image
                            for merged_back in back_transformer.all_items:
                                self.merge_and_save(merged_back, transformed_foreg, foreg_name,  backgr_name)
                    else:
                            self.merge_and_save(TransformationTracer(im), transformed_foreg, foreg_name, backgr_name)

                    bidx+=1

    @profile
    def merge_and_save(self, back_tracer, foreg_tracer, foreg_name, back_name):

        preprocessing.check_merging_size(
            back_tracer.obj,
            foreg_tracer.obj
        )

        chosen_point = self.choose_point(back_tracer.obj, foreg_tracer.obj)

        pil_merged, bboxes = (
            preprocessing.merge_img_in_background(
                back_tracer.obj,
                foreg_tracer.obj,
                [chosen_point],
                self.blur_merge
            )
        )

        ##
        # for tracing old transformations
        new_tracer = TransformationTracer.merge(foreg_tracer, back_tracer)

        #
        # len of post_merged transformations
        len_postmerge_transformation = len(self.list_transformation_to_call_back['post_merge'])

        #
        # if exist do post_merge transformations
        if (len_postmerge_transformation) > 0:

            # TODO - important if rotate or scale recalculate bbox
            back_transformer = CombinatorialTransformer(
                self.list_transformation_to_call_back['post_merge'],
                self.list_transformation_background['post_merge'],
                pil_merged[0],
                new_tracer
            )
            back_transformer.apply_transformations()

            #
            # all have the same bbox
            for transformed_back in back_transformer.all_items:
                self.save_final_image(transformed_back, bboxes[0], foreg_name, back_name)

        else:
            self.save_final_image(TransformationTracer(pil_merged[0]), bboxes[0], foreg_name, back_name)

    def create_annotation(self, original_name_noext):

        img_annotation = {}

        img_annotation['object'] = {}

        #
        # get extra annotation from db
        if self.db_conf is not None:
            # print "search for img {}".format(original_name_noext)
            rec_img = self.get_bbox_from_db(original_name_noext + '.jpg')

            if rec_img is not None:

                for k in rec_img.get_keys():
                    #
                    # insert without prefix bs_
                    img_annotation['object'][str(k[3:])] = rec_img.__getattr__(k)

            try:
                img_annotation['object']['name'] = img_annotation['object']['class']
            except:
                print "image {} have no class field in db ".format(rec_img)
                return

        return img_annotation

    def merge_annotation(self, img_annotation, extra_annotations):
        #
        # add extra annotations from tracers

        #
        # to dict
        img_annotation = merge_lists(img_annotation,
             two_list_to_dict(extra_annotations.applied_funcs_names,
                              extra_annotations.params))

        return img_annotation

    def set_bbox_annotation(self, img_annotation, bboxes):

        #
        # set new bounding boxes
        bounding_boxes = bboxes

        img_annotation['object']['bndbox'] = {}
        img_annotation['object']['bndbox']['xmin'] = bounding_boxes[0]
        img_annotation['object']['bndbox']['ymin'] = bounding_boxes[1]
        img_annotation['object']['bndbox']['xmax'] = bounding_boxes[2]
        img_annotation['object']['bndbox']['ymax'] = bounding_boxes[3]

        return img_annotation

    def set_voc_style_annotation(self, img_annotation, im, new_img_name, back_orig_name_prefix):

        # voc style adjust
        img_annotation['filename'] = new_img_name + '.jpg'
        img_annotation['orig_background'] = back_orig_name_prefix
        img_annotation['folder'] = self.name

        #
        # TODO - annotate
        img_annotation['object']['truncated'] = 0
        img_annotation['segmented'] = 0

        img_annotation['size'] = {}
        sz = im.size
        img_annotation['size']['width'] = sz[0]
        img_annotation['size']['height'] = sz[1]

        return img_annotation

    def set_depth_in_annotation(self,img_annotation, path_save):

        #
        # reload image to know bits [only jpeg]
        try:
            img_annotation['size']['depth'] = im.bits
        except:
            imjpg = PIL.Image.open(path_save)
            img_annotation['size']['depth'] = imjpg.bits

        return img_annotation

    @profile
    def create_folders_and_write_img(self, new_img_name, im):

        #
        # create faster_rcnn voc style folders
        self.fast_ut.create_base_folders(self.output_folder_path, self.name)

        #
        # save images
        path_save = os.path.join(self.output_folder_path, self.name, 'JPEGImages',
                                 new_img_name + '.jpg')

        if self.DEBUG:
            print "saving {}".format(path_save)

        preprocessing.write_pil_im(im, path_save)

    @profile
    def create_xml_and_save(self, img_annotation, new_img_name):


        #
        # create xml
        _xml = dict2xml(img_annotation, roottag='annotation')
        # print _xml

        xml_path = os.path.join(self.output_folder_path,
                               self.name, 'Annotations',
                               new_img_name + '.xml')
        if self.DEBUG:
            print "saving {}".format(xml_path)

        #
        # save xml annotations
        save_text_in_file(_xml,
                  xml_path)

    @profile
    def cat_img_name_in_txt_file(self, new_img_name):

        if self.DEBUG:
            print "call to cat in txt for {}".format(new_img_name)
        #
        # append in image list file
        append_txt_to_file(new_img_name,
                   os.path.join(self.output_folder_path,
                                self.name, 'ImageSets', 'Main',
                                self.name + '.txt'))

    @profile
    def save_final_image(self, pil_merged_tracer, bboxes, foreg_name, back_name):

        im = pil_merged_tracer.obj

        #
        # original foreground image name
        original_name = foreg_name
        original_name_noext = original_name[:-4]

        #
        # original background image name
        back_orig_name_prefix = back_name

        #
        # new image name
        new_img_name = original_name_noext + '_mod_' + str(self.img_counter)

        img_annotation = self.create_annotation(original_name_noext)

        if img_annotation is None:
            gc.collect()
            return

        img_annotation = self.merge_annotation(img_annotation, pil_merged_tracer)

        img_annotation = self.set_bbox_annotation(img_annotation, bboxes)

        img_annotation = self.set_voc_style_annotation(img_annotation, im, new_img_name, back_orig_name_prefix)

        self.create_folders_and_write_img(new_img_name, im)

        self.create_xml_and_save(img_annotation, new_img_name)

        self.cat_img_name_in_txt_file(new_img_name)

        self.img_counter+=1

        gc.collect()



    def save_finals(self, plot=False):

        #
        # utility class for parsing voc style xml
        fu = fast_ut.faster_rcnn_utils()

        #
        # shuffle the list
        if self.shuffle:
            indexes = np.arange(len(self.final_pil_images))
            np.random.shuffle(indexes)
        else:
            indexes = np.arange(len(self.final_pil_images))

        #
        # save images list
        for idx in indexes:

            im = self.final_pil_images[idx]

            #print self.final_names[idx], self.final_bboxes[idx]

            #
            # original foreground image name
            original_name = self.final_names[idx]['foreg']
            original_name_noext = original_name[:-4]

            #
            # original background image name
            back_orig_name_prefix = self.final_names[idx]['backgr']

            #
            # new image name
            new_img_name = original_name_noext+'_mod_'+str(idx)

            img_annotation = {}

            img_annotation['object'] = {}

            #
            # get extra annotation from db
            if self.db_conf is not None:
                #print "search for img {}".format(original_name_noext)
                rec_img = self.get_bbox_from_db(original_name_noext+'.jpg')

                if rec_img is not None:

                    for k in rec_img.get_keys():
                        #
                        # insert without prefix bs_
                        img_annotation['object'][str(k[3:])] = rec_img.__getattr__(k)

                try:
                    img_annotation['object']['name'] = img_annotation['object']['class']
                except:
                    print "image {} have no class field in db ".format(new_img_name)
                    continue

            #
            # add extra annotations from tracers
            extra_annotations = self.final_tracers[idx]

            #
            # to dict
            img_annotation = merge_lists(img_annotation,
                     two_list_to_dict(extra_annotations.applied_funcs_names, extra_annotations.params))

            #
            # set new bounding boxes
            bounding_boxes = self.final_bboxes[idx]

            img_annotation['object']['bndbox'] = {}
            img_annotation['object']['bndbox']['xmin'] = bounding_boxes[0]
            img_annotation['object']['bndbox']['ymin'] = bounding_boxes[1]
            img_annotation['object']['bndbox']['xmax'] = bounding_boxes[2]
            img_annotation['object']['bndbox']['ymax'] = bounding_boxes[3]


            #
            # voc style adjust
            img_annotation['filename']=new_img_name+'.jpg'
            img_annotation['orig_background']=back_orig_name_prefix
            img_annotation['folder']=self.name

            #
            # TODO - annotate
            img_annotation['object']['truncated'] = 0
            img_annotation['segmented'] = 0
            img_annotation['size'] = {}
            sz = im.size
            img_annotation['size']['width'] = sz[0]
            img_annotation['size']['height'] = sz[1]

            #
            # create faster_rcnn voc style folders
            fu.create_base_folders(self.output_folder_path, self.name)

            #
            # save images
            path_save = os.path.join(self.output_folder_path, self.name, 'JPEGImages',
                                                        new_img_name + '.jpg')

            preprocessing.write_pil_im(im, path_save)

            #
            # reload image to know bits [only jpeg]
            try:
                img_annotation['size']['depth'] = im.bits
            except:
                imjpg = PIL.Image.open(path_save)
                img_annotation['size']['depth'] = imjpg.bits


            #
            # create xml
            _xml = dict2xml(img_annotation, roottag='annotation')
            #print _xml

            #
            # save xml annotations
            save_text_in_file(_xml, os.path.join(self.output_folder_path, self.name, 'Annotations', new_img_name+'.xml'))

            #
            # append in image list file
            append_txt_to_file(new_img_name, os.path.join(self.output_folder_path, self.name, 'ImageSets', 'Main', self.name+'.txt'))


        if plot:
            import matplotlib.pyplot as plt
            f, array_axes = plt.subplots(1, len(self.final_pil_images))
            for i, img in enumerate(self.final_pil_images):
                bboxes = self.final_bboxes
                im = preprocessing.pil_image_to_array(img)
                array_axes[i].imshow(im)
                array_axes[i].scatter(bboxes[i][0], bboxes[i][1], c='r')  # min
                array_axes[i].scatter(bboxes[i][2], bboxes[i][3], c='g')  # max
            plt.show()



    def get_bbox_from_db(self, bbox_name):

        self.ensure_db_conn()

        table_name = self.db_conf['table_name']

        sql = "SELECT * FROM  `{}` WHERE  `bs_img_name` =  '{}' LIMIT 0 , 30".format(table_name, bbox_name)
        bb = self.dbu.exec_sql(sql)
        self.dbu.close_connection()

        if len(bb) > 0:
            return  db_rec.db_record(bb[0])


    def ensure_db_conn(self):

        if self.db_conn is None:
            self.dbu = db_utils.DbUtils(self.db_conf)
            self.db_conn = self.dbu.connect()











if __name__ == '__main__':

    import fish_utils as fu

    fut = fu.fish_utils()
    back_path = fut.get_selected_background_path_glob(type="debug")
    foreg_path = fut.get_bboxed_selected_path_all_class_glob(type="debug")



    func_foreg_params = {
        'rotate': {
            'func': preprocessing.rotate_img_by_angle,
            'param': range_utils.choice_n_rnd_numbers_from_to_linspace(0, 360, 30, 1, integer=True)
        },
        'scale': {
            'func': preprocessing.rescale_img_by_ratio,
            'param': range_utils.choice_n_rnd_numbers_from_to_linspace(.8, 1.2, 10, 1)
        }
    }

    func_back_params = {
        'pre_merge' : {
            'contrast': {
                'func': preprocessing.change_contrast,
                'param': range_utils.choice_n_rnd_numbers_from_to_linspace(.8, 1.2, 20, 2, round=True)
            }
        },
        'post_merge': {
            'color': {
                'func': preprocessing.change_color,
                'param': range_utils.choice_n_rnd_numbers_from_to_linspace(-.8, 1.2, 10, 2)
            },
            'contrast': {
                'func': preprocessing.change_contrast,
                'param': range_utils.choice_n_rnd_numbers_from_to_linspace(.8, 1.2, 20, 2, round=True)
            },
            'light': {
                'func': preprocessing.change_light,
                'param': range_utils.choice_n_rnd_numbers_from_to_linspace(.8, 1.2, 20, 2, round=True)
            },
            'green': {
                'func': preprocessing.apply_green_mask_pil_img,
                'param': [np.NaN]
            }
        }
    }

    db_conf = {
        'host': 'localhost',
        'user': 'guest',
        'password': 'guest',
        'db': 'fish',
        'table_name': 'bbox_selection'
    }

    ##
    # no preprocessing
    # only merging, choose n background random for every foreg
    TEST_1=False

    ##
    # no preprocessing, choose all background, not random
    TEST_2=False

    ##
    # preprocessing on foreg and background, random back, verify ram usage
    TEST_3=True


    if TEST_1:
        func_to_foreg = []
        func_to_back = {
            'pre_merge': [],
            'post_merge':[]
        }
        n_back = 2
        rand_back_for_all = True


    if TEST_2:
        func_to_foreg = []
        func_to_back = {
            'pre_merge': [],
            'post_merge':[]
        }
        n_back = -1
        rand_back_for_all = False

    if TEST_3:
        func_to_foreg = ['rotate']
        func_to_back = {
            'pre_merge': [],
            'post_merge':['color']
        }
        n_back = 2
        rand_back_for_all = True


    d = DatasetCreator(
        'debug',
        back_path, foreg_path,
        func_back_params, func_foreg_params,
        func_to_back, func_to_foreg,
        output_folder_path='/tmp',
        db_conf=db_conf,
        choose_n_random_background=n_back,
        random_background_for_all=rand_back_for_all,
        debug=True
    )


    d.create_random_background_for_foreground()
