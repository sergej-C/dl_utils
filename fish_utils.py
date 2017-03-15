import os, sys

KAGGLE_PATH=os.environ.get('KAGGLE_PATH')

import numpy as np
import img_utils as imu
import db_utils as dbutils
import db_record as db_rec
from preprocessing import *

from dataset_creator import *
from data_struct_utils import *
#import data_utils as du
#import faster_rcnn_utils as fastu
from glob import glob
from extract_predictions import *

class fish_utils():
    

    classes_for_faster= ('__background__',  # always index 0, class NoF as background ?
                         'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft'
                         )

    class_to_ind = dict(zip(classes_for_faster, xrange(len(classes_for_faster))))

    classes_for_submission = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    dbconn = None
    db_conf = {
        'host': 'localhost',
        'user': 'guest',
        'password': 'guest',
        'db': 'fish',
        'table_name': 'bbox_selection'
    }
    def __init__(self):
        self.FISH_DATA_PATH=KAGGLE_PATH+'/fish'
        self.dbu = dbutils.DbUtils(self.db_conf)



    def get_fish_path(self):
        return self.FISH_DATA_PATH

    def get_all_selected_background_path_gob(self, type='train'):
        return [
            KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/background_selection/*.jpg'.format(type),
            KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/background_selection/from_fishes/*/modified/*.jpg'.format(type)
            ]

    def get_selected_background_path(self, type='train'):
        return KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/background_selection/'.format(type)

    def get_selected_background_path_glob(self, type="train"):
        return self.get_selected_background_path(type)+'*.jpg'

    def get_test_selected_background(self):
        return self.get_selected_background_path()+'img_07512.jpg'

    def get_bboxed_selected_path(self, class_name, type="train"):
        return  KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/bboxed_selection/{}/modified/'.format(type, class_name)

    def get_bboxed_selected_path_all_class_glob(self, type="train"):
        return  KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/bboxed_selection/*/modified/*.png'.format(type)

    def get_bboxed_selected_modified_path_for_class_glob(self, cls, type="train"):
        return  KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/bboxed_selection/{}/modified/*.png'.format(type, cls)

    def get_bboxed_selected_variants_path_for_class_glob(self, cls, type="train"):
        return  KAGGLE_PATH + '/fish/for_faster_rcnn1/{}/bboxed_selection/{}/modified/variants/*.png'.format(type, cls)

    def get_bboxed_selected_modified_and_variants_path_for_class_glob(self, cls, type="train"):
        return [
            self.get_bboxed_selected_modified_path_for_class_glob(cls, type),
            self.get_bboxed_selected_variants_path_for_class_glob(cls, type)
                ]

    def get_test_bboxed_selected(self):
        return self.get_bboxed_selected_path('ALB')+'img_00032_1.png'

    def get_random_test_bboxed_selected(self):
        path_forg = self.get_bboxed_selected_path_all_class_glob()
        path_back = self.get_selected_background_path_glob()
        glob_foreg = glob(path_forg)
        glob_backgr = glob(path_back)
        return np.random.permutation(glob_backgr)[0],np.random.permutation(glob_foreg)[0]


    def get_features_try_kmeans_for_class(self, cid, clusters_to_try):

        path_bboxed_imgs = "/media/sergio/0eb90434-bbe8-4218-a191-4fa0159e1a36/ml_nn/my_proj/kaggle_competition/fish/for_faster_rcnn1/train/bboxed/train/"

        # exclude background
        classes = self.classes_for_faster[1:]
        #done
        #ALB, 
        cidx = cid
        c_class = classes[cidx].upper()

        p_img = path_bboxed_imgs
        fish_path = self.get_fish_path()
        save_path=fish_path+"/bboxed_stat/"

        info_fish, path_pkl = imu.features_from_path(p_img, c_class, save_path)

        imu.try_ncluster_kmeans(class_name=c_class, path_pkl=path_pkl, n_clusters_to_try=n_clusters_to_try)
    

    def cluster_to_folders(p_img, c_class, n_clusters, path_pkl):
        imu.cluster_imgs_to_folder(p_img, c_class, n_clusters, dest_path=save_path, features=None, path_pkl=path_pkl)

    def get_bbox_from_db(self, bbox_name):

        self.ensure_db_conn()

        sql = "SELECT * FROM  `bbox_selection` WHERE  `bs_img_name` =  '{}' LIMIT 0 , 30".format(bbox_name)
        bb = self.dbu.exec_sql(sql)
        self.dbu.close_connection()

        if len(bb) > 0:
            return  db_rec.db_record(bb[0])


    def ensure_db_conn(self):

        if self.dbconn is None:
            self.dbconn = self.dbu.connect()

    def create_synth_dataset(self):
        """
        we have original images in fish/train folder
        first step in creating voc style data sets was to create one xml file for each image in Annotation folder
        then we select n background from train/NoF folder
        some background image was taken from fishes images and modified to hide fish
        n fishes from train/CAT/ [bbox area of image, that was modified with gimp]
        fishes was annotated in mysql table with new extra parameters [i.e. pose, occlusion]

        now create new data set with this element
        when creating new image merging a background and a modified bboxed fish,
        we get annotation from db, and related xml annotation of the original image from which the current fish was extracted
        finally we get new annotations [i.e. type of filters, rotation angle, scaling ratio for width or height,...]
        and merge all together

        :return:
        """

    def read_original_fish_xml(self, boxed_fish_img_name):
        """
        search in default path of annotations for the xml file associated with the specified fish name

        :param boxed_fish_img_name:

        :return:
        a dict with params extracted from xml file
        """

    def extract_prediction_create_submission(prepath_pkl, folder_ds, out_path='/tmp', clip=True):

        clss = ('alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
        lbl_clss = ('NoF', 'alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft')
        class_to_ind = du.array_to_ind(clss)
      
        pkl_path=prepath_pkl+folder_ds
        print "searching tar in {}".format(pkl_path)
        df, all_dets, all_recs = test_extract(pkl_path, clss, project_name='fish')
        df = prepare_df_for_analysis(df)

        rows = get_ypred_from_all_dets(all_dets)
        create_submission(rows=rows, lbl_clss=lbl_clss, out_path=out_path, prefix=folder_ds, doclip=clip)
        
        return df, all_dets, all_recs, rows

    def get_glob_paths(self, base_path_backgrounds_glob, base_path_foreground_glob):

        #
        # get the list of foregrounds and backgrounds
        if isinstance(base_path_backgrounds_glob, list):
            background_list = []
            for p in base_path_backgrounds_glob:
                for e in (glob(p)):
                    background_list.append(e)
        else:
            background_list = glob(base_path_backgrounds_glob)


        if isinstance(base_path_foreground_glob, list):
            foreground_list = []
            for p in base_path_foreground_glob:
                for e in (glob(p)):
                    foreground_list.append(e)
        else:
            foreground_list = glob(base_path_foreground_glob)

        #print len(foreground_list), len(background_list)
        return foreground_list, background_list

    def get_n_random_item_from_list(self, flist, blist, N):

        n_imgs = 0

        f_imset = list(flist)
        b_imset = list(blist)

        f_list = []
        b_list = []
        # until we have N images, do
        while (n_imgs <= N):

            # we want new elements at each iteration
            np.random.shuffle(f_imset)
            np.random.shuffle(b_imset)

            lf = len(f_imset)
            lb = len(b_imset)
            # get the number of elements from both
            # so min between the two length
            n_to_get = min(lf, lb)

            f_list = f_list + f_imset[:n_to_get]
            b_list = b_list + b_imset[:n_to_get]

            llist = len(f_list)

            #print "f_list has len {}".format(llist)

            if llist > N:
                diff_l = llist - N
                #print "f_list has len > {}, diff {}".format(N,diff_l)
                f_list = f_list[:-diff_l]
                b_list = b_list[:-diff_l]
            n_imgs += n_to_get
        return f_list, b_list

    def is_variants(self, path):
        return path.find('variants') > -1

    def is_green(self, name, ext='jpg'):
        if self.db_conf is not None:

            self.ensure_db_conn()

            # print "search for img {}".format(original_name_noext)
            rec_img = DatasetCreator.get_bbox_from_db(name+'.'+ext, self.db_conf, self.dbu)

            #
            # background is green - if will find a fish in db with its name, check with that fish
            if rec_img is not None:
                return rec_img.bs_color_mask == 'green'

    def create_n_random_images(self, N, img_name_prefix, name, output_folder_path='/tmp', debug=False):
        """
        return N images merging


        :return:
        """
        classes = self.classes_for_faster[1:]

        for c in classes:

            cls = c.upper()

            fast_ut = faster_rcnn_utils()
            p_back = self.get_all_selected_background_path_gob(type='')
            p_foreg = self.get_bboxed_selected_modified_and_variants_path_for_class_glob(cls, type='')

            foreground_list, background_list = self.get_glob_paths(p_back, p_foreg)
            imf_list, imb_list = self.get_n_random_item_from_list(foreground_list, background_list, N)

            # open images
            for imd, foreg_imp in enumerate(imf_list):
                back_imp = imb_list[imd]

                is_variant = False
                # if variants name for search in db is different
                if self.is_variants(foreg_imp):
                    is_variant = True


                f_name = foreg_imp.split('/')[-1]
                b_name = back_imp.split('/')[-1]

                print "f_name {}, b_name {}".format(f_name, b_name)
                pil_f = preprocessing.open_image(foreg_imp)
                pil_b = preprocessing.open_image(back_imp)

                #
                # original foreground image name
                original_name = f_name
                if is_variant:
                    original_name_noext = original_name[:-4][:-2]
                else:
                    original_name_noext = original_name[:-4]
                #
                # original background image name
                back_orig_name_prefix = b_name

                #
                # if exists selected foreground from this background,
                # its name is name of background without ext plus _1 or 2 (1 at least) + ext
                back_name_for_search = back_orig_name_prefix[:-4] + '_1'

                back_is_green = False
                fore_is_green = False

                #
                # fish is green - check in db
                back_is_green = self.is_green(back_name_for_search)
                if back_is_green is None:
                    back_is_green = False
                fore_is_green = self.is_green(original_name_noext)

                print back_name_for_search,back_is_green, original_name_noext, fore_is_green

                #
                # new image name
                new_img_name = img_name_prefix + '_' + original_name_noext + '_mod_' + str(imd)

                tracer = TransformationTracer(pil_f)

                #
                # preprocessing foreg

                # random scale but based on size of image - TODO
                scale_r = range_utils.choice_n_rnd_numbers_from_to_linspace(.95, 1.2, 10, 1)[0]
                pil_f = preprocessing.rescale_img_by_ratio(pil_f, scale_r)
                tracer.add_func_param(preprocessing.rescale_img_by_ratio, scale_r)

                # random rotation
                rot_r = range_utils.choice_n_rnd_numbers_from_to_linspace(0, 180, 30, 1, integer=True)[0]
                pil_f = preprocessing.rotate_img_by_angle(pil_f, rot_r)
                tracer.add_func_param(preprocessing.rotate_img_by_angle, rot_r)

                # if is green background and foreground isnt, mask green foreg
                if back_is_green and not fore_is_green:
                    pil_f = preprocessing.apply_green_mask_pil_img(pil_f, False)
                    tracer.add_func_param(preprocessing.apply_green_mask_pil_img, 'on_foreg')

                # if is green foreground and backgorund isnt, mask green back
                if not back_is_green and fore_is_green:
                    pil_b = preprocessing.apply_green_mask_pil_img(pil_b, False)
                    tracer.add_func_param(preprocessing.apply_green_mask_pil_img, 'on_background')

                #
                # merging
                preprocessing.check_merging_size(
                    pil_b,
                    pil_f
                )

                chosen_point = DatasetCreator.choose_point(pil_b, pil_f)

                pil_merged, bboxes = (
                    preprocessing.merge_img_in_background(

                        pil_b,
                        pil_f,
                        [chosen_point],
                        False
                    )
                )

                pil_merged = im = pil_merged[0]
                #
                # preprocessing merged
                light_r = range_utils.choice_n_rnd_numbers_from_to_linspace(.85, 1.15, 20, 1, round=True)[0]
                pil_merged = preprocessing.change_light(pil_merged, light_r)
                tracer.add_func_param(preprocessing.change_light, light_r)

                contrast_r = range_utils.choice_n_rnd_numbers_from_to_linspace(.85, 1.15, 20, 1, round=True)[0]
                pil_merged = preprocessing.change_contrast(pil_merged, contrast_r)
                tracer.add_func_param(preprocessing.change_contrast, contrast_r)

                #
                # save anno, images
                self.ensure_db_conn()

                img_annotation = DatasetCreator.create_annotation(original_name_noext, self.db_conf, self.dbu)

                if img_annotation is not None:
                    img_annotation = DatasetCreator.merge_annotation(img_annotation, tracer)

                    img_annotation = DatasetCreator.set_bbox_annotation(img_annotation, bboxes)

                    img_annotation = DatasetCreator.set_voc_style_annotation(name, img_annotation, im, new_img_name,
                                                                             back_orig_name_prefix)

                    DatasetCreator.create_folders_and_write_img(new_img_name, im, fast_ut, output_folder_path, name, debug)

                    DatasetCreator.create_xml_and_save(img_annotation, new_img_name, output_folder_path, name, debug)

                    DatasetCreator.cat_img_name_in_txt_file(new_img_name.strip(), output_folder_path, name,
                                                            debug)
                else:
                     print   "warning, annotation for im {} not created".format(foreg_imp)

            #
            # save image list


if __name__ == '__main__':

    fu = fish_utils()
    box_rec = (fu.get_bbox_from_db('img_03038_1.jpg'))
