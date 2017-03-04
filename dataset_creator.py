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
                 shuffle=True,
                 blur_merge=True,
                 base_path_background_points_xml=None,
                 class_names=None,
                 db_conf=None

                 ):

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



    def create(self):


        #
        # get the list of foregrounds and backgrounds
        self.foreground_list = glob(self.base_path_foreground_glob)

        self.background_list = glob(self.base_path_backgrounds_glob)

        print len(self.foreground_list), len(self.background_list)

        # checks all images can be loaded from PIL (if empty file -> removed)
        imu.check_imgs_list_and_rm_empty(self.foreground_list)
        imu.check_imgs_list_and_rm_empty(self.background_list)

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

            #print "transformed_list_back_premerge pre_merge len {}".format(len(transformed_list_back_premerge))

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

                # print "chosen point {}".format(chosen_point)
                preprocessing.check_merging_size(
                    back_tracer.obj,
                    modified_foreg.obj
                )

                chosen_point = self.choose_point(back_obj, modified_foreg.obj)

                pil_merged, bboxes = (
                    preprocessing.merge_img_in_background(

                        back_tracer.obj,
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

                img_annotation['object']['name'] = img_annotation['object']['class']

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

    #
    # TODO - do merging without any transformation
    func_to_foreg = ['rotate']

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

    func_to_back = {
        'pre_merge': [],
        'post_merge':[]
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

    d = DatasetCreator(
        'debug',
        back_path, foreg_path,
        func_back_params, func_foreg_params,
        func_to_back, func_to_foreg,
        output_folder_path='/tmp',
        db_conf=db_conf
    )


    d.create()
