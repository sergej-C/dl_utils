import numpy as np
import PIL.Image
import PIL.ImageOps
from skimage import data, color, io, img_as_float
import PIL.ImageFilter as ImageFilter
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from glob import glob
import cv2

from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from resizeimage import resizeimage
import data_utils as du
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import scipy.ndimage as ndi

class preprocessing():
    """
    some utilities for apply a variety of filters and transformations on a set of images
    """

    filters_dict = {'blur': ImageFilter.BLUR,
                    'contour': ImageFilter.CONTOUR,
                    'detail': ImageFilter.DETAIL,
                    'edge_enhance': ImageFilter.EDGE_ENHANCE,
                    'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
                    'emboss': ImageFilter.EMBOSS,
                    'find_edges': ImageFilter.FIND_EDGES,
                    'smooth': ImageFilter.SMOOTH,
                    'smooth_more': ImageFilter.SMOOTH_MORE,
                    'sharpen': ImageFilter.SHARPEN
                    }

    @staticmethod
    def pil_image_to_array(pil_img):
        """
        Loads PIL JPEG image into 3D Numpy array of shape
        (width, height, channels)
        """
        im_arr = np.fromstring(pil_img.tobytes(), dtype=np.uint8)

        if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
            #
            # alpha channel
            n_channels = 4
        else:
            n_channels = 3

        im_arr = im_arr.reshape((pil_img.size[1], pil_img.size[0], n_channels))
        return im_arr

    @staticmethod
    def paste_centered_in_point(pil_backg, pil_foreg, points, i, blur=True):
        """
        for plotting an image with the center at points[i]


        :param pil_backg: the background image
        :param pil_foreg: the foreground image [fromm PNG image]
        :param points: an array of x,y coordinate
        :param i: selected point from points
        :return: xmin, ymin, xmax, ymax, pil_backg_cpy
        """

        pil_backg_cpy = pil_backg.copy()
        wb, hb = pil_backg.size
        w, h = pil_foreg.size

        # ensure limits
        cent_x = max(0, int(points[i][0] - w / 2.))
        cent_y = max(0, int(points[i][1] - h / 2.))
        cent_x = min(cent_x, int(wb - w))
        cent_y = min(cent_y, int(hb - h))

        if blur:
            pil_backg_cpy = preprocessing.blur_and_merge(pil_backg_cpy, pil_foreg, (cent_x, cent_y))
        else:
            pil_backg_cpy.paste(pil_foreg, (cent_x, cent_y), pil_foreg)
        xmin = cent_x
        ymin = cent_y
        xmax = cent_x + w
        ymax = cent_y + h

        return xmin, ymin, xmax, ymax, pil_backg_cpy

    @staticmethod
    def min_max_from_centers(points, f_im, b_im):

        wb, hb = b_im.size
        w, h = f_im.size

        # ensure limits
        cent_x = max(0, int(points[0] - w / 2.))
        cent_y = max(0, int(points[1] - h / 2.))
        cent_x = min(cent_x, int(wb - w))
        cent_y = min(cent_y, int(hb - h))

        xmin = cent_x
        ymin = cent_y
        xmax = cent_x + w
        ymax = cent_y + h
        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def merge_img_in_background(pil_backg, pil_foreg, points, blur=True):
        """

        :param pil_backg:
        :param pil_foreg:
        :param points:
        :return:
        merged: PIL image
        bboxes: where is the merged image
        """
        merged = []
        bboxes = []
        for i,p in enumerate(points):
            xmin, ymin, xmax, ymax, pil_backg_merged = preprocessing.paste_centered_in_point(pil_backg, pil_foreg, points, i, blur=blur)
            merged.append(pil_backg_merged)
            bboxes.append(np.array((xmin, ymin, xmax, ymax)))
        return merged, bboxes

    @staticmethod
    def open_image(path):
        return PIL.Image.open(path)

    @staticmethod
    def rotate_img_by_angle(pil_img, angle):
        """
        rotate pil loaded image, for each angles
        :param pil_img: image loaded with PIL.open
        :param angle: one angle
        :return:
         rotated: rotated image (PIL images)
        """

        return  pil_img.rotate(angle, expand=True)


    @staticmethod
    def rotate_img_by_angles(pil_img, angles):
        """
        rotate pil loaded image, for each angles
        :param pil_img: image loaded with PIL.open
        :param angles: array with angles
        :return:
         rotated: list of rotated images (PIL images)
         new_sizes: list of new sizes
        """
        rotated = []
        new_sizes = []
        for i, a in enumerate(angles):
            rotated_im = preprocessing.rotate_img_by_angle(pil_img, a)
            rotated.append(rotated_im)
            new_sizes.append(np.array((rotated_im.size)))
        return rotated

    @staticmethod
    def rescale_img_by_ratio(pil_img, ratio):
        """

        :param pil_img:
        :param ratio:
        :return:
        """

        w, h = pil_img.size
        return pil_img.resize(size=(int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))


    @staticmethod
    def rescale_img_by_ratios(pil_img, ratios):
        """

        :param pil_img:
        :param ratios:
        :return:
        """
        rescaled=[]
        new_sizes = []
        w,h=pil_img.size
        for i, r in enumerate(ratios):
            rescaled_im = preprocessing.rescale_img_by_ratio(pil_img, r)
            rescaled.append(rescaled_im)
            new_sizes.append(np.array((rescaled_im.size)))
        return rescaled

    @staticmethod
    def apply_rgb_mask(img, rgb_mask, alpha=0.6, plot=False):
        """
        from from http://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
        Apply the mask with the specified color to the img_color

        Parameters:
        - img: array of rgb image
        - rgb_mask: i.e. [255, 255, 0]  # RGB
        - alpha: transparency level

        Return:
        - masked image
        """

        img_color = img.copy()

        rows, cols = img_color.shape[0], img_color.shape[1]

        # Construct a colour image to superimpose
        color_mask = np.zeros((rows, cols, 3))
        color_mask[:, :] = rgb_mask

        # Construct RGB version of grey-level image
        # img_color = np.dstack((img, img, img))


        # Convert the input image and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(color_mask)

        # Replace the hue and saturation of the original image
        # with that of the color mask
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)

        if plot:
            # Display the output
            f, (ax1, ax2) = plt.subplots(1, 2,
                                         subplot_kw={'xticks': [], 'yticks': []})
            # ax0.imshow(img, cmap=plt.cm.gray)
            ax1.imshow(color_mask)
            ax2.imshow(img_masked)
            plt.show()

        del img_color
        del img_hsv
        del color_mask
        del color_mask_hsv

        return img_masked

    @staticmethod
    def apply_blue_mask(img, plot=False):
        return preprocessing.apply_rgb_mask(img, [0, 1, 1], plot=plot)

    @staticmethod
    def apply_green_mask(img, plot=False):
        return preprocessing.apply_rgb_mask(img, [0, 1, 0], plot=plot)


    @staticmethod
    def apply_green_mask_pil_img(pil_img, plot=False):

        img_arr = preprocessing.pil_image_to_array(pil_img)
        img_arr_masked = preprocessing.apply_green_mask(img_arr, plot=plot)
        del img_arr
        #print img_arr_masked.dtype
        masked = PIL.Image.fromarray(img_as_ubyte(img_arr_masked), mode='RGB')
        del img_arr_masked
        return masked

    @staticmethod
    def apply_filter(pil_img, filter_key):
        """
        apply a PIL filter specified by key taken from preprocessing.filter_key
        :param pil_img:
        :param filter_key:
        :return:
         filtered_pil_img
        """
        return pil_img.filter(preprocessing.filters_dict[filter_key])

    @staticmethod
    def apply_filters(pil_img, filter_keys_dict):
        """
        apply all filters specified in filter_filter_keys_dict
        :param pil_img:
        :param filter_keys_dict:
        :return:
        """
        filtered = []
        applied_filters = []

        for k in filter_keys_dict:
            filtered.append(preprocessing.apply_filter(pil_img, k))
            applied_filters.append(k)

        return filtered


    @staticmethod
    def auto_contrast(pil_img):
        return PIL.ImageOps.autocontrast(pil_img)

    @staticmethod
    def equalize(pil_img):
        return PIL.ImageOps.equalize(pil_img)

    @staticmethod
    def open_with_cv(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def change_light(pil_img, factor, show=False):
        """
        http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html

        :param pil_img:
        :param factor:
        :param show:
        :return:
        """

        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(factor)

        if show:
            enhanced.show()

        return enhanced

    @staticmethod
    def change_contrast(pil_img, factor, show=False):
        """
        :param pil_img:
        :param factor:
        :param show:
        :return:
        """

        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)

        if show:
            enhanced.show()

        return enhanced

    @staticmethod
    def change_color(pil_img, factor, show=False):
        """
        :param pil_img:
        :param factor:
        :param show:
        :return:
        """

        enhancer = ImageEnhance.Color(pil_img)
        enhanced = enhancer.enhance(factor)

        if show:
            enhanced.show()
        return enhanced

    @staticmethod
    def change_light_img_by_factors(pil_img, factors, show=False):
        """

        :param pil_img:
        :param factors:
        :return:

        """
        relight = []

        for i, f in enumerate(factors):
            relight.append(preprocessing.change_light(pil_img, f, show=show))

        return relight

    @staticmethod
    def change_color_img_by_factors(pil_img, factors, show=False):
        """

        :param pil_img:
        :param factors:
        :return:

        """
        relight = []

        for i, f in enumerate(factors):
            relight.append(preprocessing.change_color(pil_img, f, show=show))

        return relight


    @staticmethod
    def change_contrast_img_by_factors(pil_img, factors, show=False):
        """

        :param pil_img:
        :param factors:
        :return:

        """
        relight = []

        for i, f in enumerate(factors):
            relight.append(preprocessing.change_contrast(pil_img, f, show=show))

        return relight

    @staticmethod
    def write_pil_im(pil_im, path_name_with_ext, type="JPEG"):
        pil_im.save(path_name_with_ext, type)


    @staticmethod
    def blur_and_merge(pil_background, pil_foreground, points, radius=5):
        """

        :param pil_background:
        :param pil_foreground:
        :param points:
        :param radius:
        :return:
        """
        src = preprocessing.pil_image_to_array(pil_foreground)

        alpha = src[:, :, -1]

        im2, contours, hierarchy = cv2.findContours(alpha.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print contours[0]

        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(alpha.shape, np.uint8)

        cv2.drawContours(mask, [contours[0]], -1, (255, 255, 255), cv2.FILLED)

        mask = cv2.erode(mask, kernel)
        mask_to_blur = mask.copy()
        bluried_mask = cv2.GaussianBlur(mask_to_blur, (radius, radius), 0)

        pil_mask_bluried = PIL.Image.fromarray(bluried_mask)

        pil_back_copy = pil_background.copy()
        pil_back_copy.paste(pil_foreground, (int(points[0]), int(points[1])), pil_mask_bluried)
        return pil_back_copy

    @staticmethod
    def check_merging_size(pil_back, pil_foreg, pix_to_remove_ratio_respect_back=.8):
        sz_back=pil_back.size
        sz_foreg=pil_foreg.size

        if sz_back[0]<sz_foreg[0] or sz_back[1]<sz_foreg[1]:
            print "background image size is < foreground image, resize foreground"
            max_size=max(sz_foreg[0], sz_foreg[1])
            if max_size==sz_foreg[0]:
                idx_max=0
                idx_min=1
            else:
                idx_max=1
                idx_min=0

            new_side=sz_back[idx_max]*pix_to_remove_ratio_respect_back
            old_ratio = float(sz_foreg[idx_max]/sz_foreg[idx_min])
            new_other_side=float(new_side/old_ratio)
            new_sz=[0, 0]
            new_sz[idx_max]=new_side
            new_sz[idx_min]=new_other_side

            if sz_back[idx_min] < new_sz[idx_min]:
                new_side_other=sz_back[idx_min]*pix_to_remove_ratio_respect_back
                new_side=new_other_side*old_ratio
                new_sz = [0, 0]
                new_sz[idx_max] = new_side
                new_sz[idx_min] = new_side_other

            preprocessing.resize(pil_foreg, new_sz)


    # @staticmethod
    # def resize_from_other_ratio_to_new(pil_toresize, pil_get_ratio_from, pil_get_ratio_to):
    #     """
    #     get aspect ratio respect one image and recalculate for merging in a new image
    #     to maintain the original size in the new image [respect to the width]
    #     :param pil_toresize:
    #     :param pil_get_ratio_from:
    #     :param pil_get_ratio_to:
    #     :return:
    #     """
    #     sz = pil_toresize.size
    #     from_sz = pil_get_ratio_from.size
    #     to_sz = pil_get_ratio_to.size
    #
    #     toresize_width = sz[0]
    #     from_width = from_sz[0]
    #     to_width = to_sz[0]
    #
    #     #
    #     # ratio between toresize width and its original image width
    #     orig_w_ratio = float(toresize_width / from_width)


    @staticmethod
    def resize(pil_img, new_sz):
        pil_img.thumbnail(new_sz, PIL.Image.ANTIALIAS)

    @staticmethod
    def resize_contain_image_in_glob(glob_path, new_size, outp, use_class_name=True):
        """

        if use_class_name use folder name as class name
        """

        imgs_set = glob(glob_path)
        szw=new_size[0]
        szh=new_size[1]

        hist_equal_ = False

        for i,img in enumerate(imgs_set):
            
            img_name = img.split('/')[-1]
            if use_class_name:
                cls_name = img.split('/')[-2]
                du.mkdirs(outp+cls_name)
            else:
                cls_name = ''

            #print cls_name.upper()
            
            im = cv2.imread(img)
            #print im.shape
            if im.shape[2]==4:
                cvc = cv2.COLOR_BGRA2RGB
            else:
                cvc = cv2.COLOR_BGR2RGB
            im = cv2.cvtColor(im, cvc)
            h, w = im.shape[0], im.shape[1]   
            
            dst='{}/{}/{}'.format(outp, cls_name, img_name)
            
            
            if h <= w:
                #print '---ok'
                #print "ok, continue h{} < w{}, cp {} to {}".format(h,w,img, outp+img_name)   
                pim = PIL.Image.open(img)
                pim = resizeimage.resize_contain(pim,[szw, szh])
               
                im = preprocessing.pil_image_to_array(pim)
                
                if hist_equal_:
                    im = hist_equalize(im)
                    
                cv2.imwrite(dst, im)
                continue
            else:
                #print '---transpose'                
                im = im.transpose((1,0,2))                
                pim = PIL.Image.fromarray(np.uint8(im))
                pim = resizeimage.resize_contain(pim,[szw, szh])
                
                im = preprocessing.pil_image_to_array(pim)
                
                if hist_equal_:
                    im = hist_equalize(im)
                cv2.imwrite(dst, im)



    #from keras preprocessing
    @staticmethod
    def random_shear(pil_im, intensity, row_axis=1, col_axis=2, channel_axis=0,
                     fill_mode='nearest', cval=0.):
        """Performs a random spatial shear of a Numpy image tensor.

         # Arguments
             x: Input tensor. Must be 3D.
             intensity: Transformation intensity.
             row_axis: Index of axis for rows in the input tensor.
             col_axis: Index of axis for columns in the input tensor.
             channel_axis: Index of axis for channels in the input tensor.
             fill_mode: Points outside the boundaries of the input
                 are filled according to the given mode
                 (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
             cval: Value used for points outside the boundaries
                 of the input if `mode='constant'`.

         # Returns
             Sheared Numpy image tensor.
         """

        x = preprocessing.pil_image_to_array(pil_im)
        shear = intensity
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        h, w = x.shape[row_axis], x.shape[col_axis]

        transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
        x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return PIL.Image.fromarray(x)

#from keras preprocessing
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

#from keras preprocessing
def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


import skimage
from PIL import Image
import math

def resize_and_rotate_taller(outp, glob_imgs, szh, szw, taller_ratio=1./4.):

    ratios = []
    for i,img in enumerate(glob_imgs):
     
        img_name = img.split('/')[-1]
        cls_name = img.split('/')[-2]
        #name_cropped = img_name.split('.')[0]+'_1.jpg'
        name_cropped = img_name
        # if num 2 (there are only one pts for images)
        # choose the nearer to the bbox
        num = img_name.split('_')[-1].split('.')[0]
        #if num=='2':
         
        #name_orig = '_'.join(name_cropped.split('_')[:2])+'.jpg'
        #im_p = prepath_cropped+'/'+cls_name+'/'+name_cropped
        du.mkdirs(outp+cls_name)
        #print cls_name.upper()

        im = skimage.io.imread(img)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[0], im.shape[1]
        ratios.append(float(h)/w) 
        dst='{}/{}/{}'.format(outp, cls_name, img_name)
        #fig, axs = plt.subplots(1,2)    
        #axs[0].imshow(im)
        if h > w and (h-w)>(w*taller_ratio):
            im = rotate(im, angle=90)

        im = resize(im, (szw, szh))
        skimage.io.imsave(dst, im)
    return ratios

def padd_from_edge(im, pos_w, pos_h, neww, newh):
    for i in range(im.shape[2]):
        if pos_w > 1:
            v_edge = im[:,pos_w,i]    
            region_shape = im[:,:pos_w,i].shape
            region_pre = np.array([v_edge,]*(region_shape[1])).transpose()
            im[:,:pos_w,i] = region_pre

            v_edge = im[:,(pos_w+neww-1),i]    
            region_shape = im[:,(pos_w+neww):,i].shape
            region_post = np.array([v_edge,]*(region_shape[1])).transpose()
            im[:,(pos_w+neww):,i] = region_post


        if pos_h > 1:
            h_edge = im[pos_h,:,i]    
            region_shape = im[:pos_h,:,i].shape
            region_pre = np.array([h_edge,]*(region_shape[0]))
            im[:pos_h,:,i] = region_pre

            h_edge = im[pos_h+newh-1,:,i]            
            region_shape = im[pos_h+newh:,:,i].shape
            print region_shape
            #rounding not always the same 
            region_post = np.array([h_edge,]*(region_shape[0]))
            im[pos_h+newh:,:,i] = region_post
    return im



def resizecontain_and_rotate_taller(outp, glob_imgs, szh, szw, pad_mode="black", taller_ratio=1./4.):

    ratios = []
    for i,img in enumerate(glob_imgs):
     
        img_name = img.split('/')[-1]
        cls_name = img.split('/')[-2]
        #name_cropped = img_name.split('.')[0]+'_1.jpg'
        name_cropped = img_name
        # if num 2 (there are only one pts for images)
        # choose the nearer to the bbox
        num = img_name.split('_')[-1].split('.')[0]
        #if num=='2':
         
        #name_orig = '_'.join(name_cropped.split('_')[:2])+'.jpg'
        #im_p = prepath_cropped+'/'+cls_name+'/'+name_cropped
        du.mkdirs(outp+cls_name)
        #print cls_name.upper()

        im = skimage.io.imread(img)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[0], im.shape[1]
        ratios.append(float(h)/w) 
        dst='{}/{}/{}'.format(outp, cls_name, img_name)
        #fig, axs = plt.subplots(1,2)    
        #axs[0].imshow(im)
        pim = PIL.Image.open(img)
        #print "h{} w{} imgname{}".format(h,w,img_name)
        if pad_mode=='mean':
            m_color = np.hstack((np.mean(im.astype(np.uint8),  axis=(0, 1)), 255))
            pad_color = totuple(m_color.astype(np.uint8))
        elif pad_mode == 'black':
            pad_color=(0, 0, 0, 0)
        elif pad_mode == 'white':
            pad_color = (255, 255, 255, 0)

        pim,(pos_w, pos_h, neww, newh) = loc_resize_contain(pim, (szw, szh), pad_color=pad_color)

        if h > w and (h-w)>(w*(taller_ratio)):
            pim = pim.rotate(angle=90)

        im = preprocessing.pil_image_to_array(pim)
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        if pad_mode=="edge":
            im =  padd_from_edge(im, pos_w, pos_h, neww, newh)
        skimage.io.imsave(dst, im)
    return ratios

def padd_from_edge(im, pos_w, pos_h, neww, newh):
    for i in range(im.shape[2]):
        if pos_w > 1:
            v_edge = im[:,pos_w,i]    
            region_shape = im[:,:pos_w,i].shape
            region_pre = np.array([v_edge,]*(region_shape[1])).transpose()
            im[:,:pos_w,i] = region_pre

            v_edge = im[:,(pos_w+neww-1),i]    
            region_shape = im[:,(pos_w+neww):,i].shape
            region_post = np.array([v_edge,]*(region_shape[1])).transpose()
            im[:,(pos_w+neww):,i] = region_post


        if pos_h > 1:
            h_edge = im[pos_h,:,i]    
            region_shape = im[:pos_h,:,i].shape
            region_pre = np.array([h_edge,]*(region_shape[0]))
            im[:pos_h,:,i] = region_pre

            h_edge = im[pos_h+newh-1,:,i]            
            region_shape = im[pos_h+newh:,:,i].shape
            print region_shape
            #rounding not always the same 
            region_post = np.array([h_edge,]*(region_shape[0]))
            im[pos_h+newh:,:,i] = region_post
    return im

def resize_pts(pts, ow, oh, nw, nh, pos_w, pos_h):
    """
    calculate coo of points in a resized img
    in pts pair of coo (x,y)
    """
    new_pts = []
    for p in pts:
        ox = p[0]
        oy = p[1]
        newx = (ox/float(ow)*nw) + pos_w
        newy = (oy/float(oh)*nh) + pos_h
        new_pts.append((newx, newy))

    return new_pts


def loc_resize_contain(image, size, pad_color=(255, 255, 255, 0)):
    """
    Resize image according to size.
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.format
    img = image.copy()
    img.thumbnail((size[0], size[1]), Image.LANCZOS)
    
    neww, newh = img.size
    background = Image.new('RGBA', (size[0], size[1]), pad_color)
    diff_w = (size[0] - img.size[0])
    diff_h = (size[1] - img.size[1])
    pos_h = int(math.ceil( diff_h / 2))
    pos_w = int(math.ceil( diff_w / 2))
    img_position = (
        pos_w,pos_h        
    )
    background.paste(img, img_position)
    background.format = img_format
    
    return background, (pos_w, pos_h, neww, newh)


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def one_hot_encode(y):
     lb = LabelBinarizer()
     lb.fit(y.squeeze())
     y = lb.transform(y)
     return y

def prepare_data_from_folder_classes(imgs_glob_path, clsses, gt=True):
    imgs = glob(imgs_glob_path+'/*/*.jpg')
    cls2ind = du.array_to_ind(clsses)
    y_train = []
    X_train = []
    img_k = []
    for img in imgs:
        if gt:
            # presume folder name of images as gt class
            cls = img.split('/')[-2].lower()
            y_train.append(cls2ind[cls])

        img_k.append(img.split('/')[-1])

        im = skimage.io.imread(img)
        #im2 = resize(im_, (sz, sz), mode='reflect')
        X_train.append(im)

    
    return np.array(X_train), np.array(y_train), img_k

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import fish_utils as fu

    ALL_TRUE=False
    TEST_MERGE=False
    TEST_ROTATION=False
    TEST_RESCALED=False
    TEST_MASK=False
    TEST_FILTERS=False
    TEST_LIGHT=False
    TEST_COLOR=False
    TEST_CONTRAST=False
    TEST_RESIZE=True

    fut = fu.fish_utils()
    ex_background, ex_bbox = fut.get_random_test_bboxed_selected()

    pil_backg = preprocessing.open_image(ex_background)
    pil_foreg = preprocessing.open_image(ex_bbox)

    if TEST_MERGE or ALL_TRUE:
        points = [(200, 120), (523, 231), (0,0), (125,780)]
        merged, bboxes = preprocessing.merge_img_in_background(pil_backg, pil_foreg, points)
        print "merged images {}".format(len(merged))
        f, array_axes = plt.subplots(1, len(merged))
        for i, img in enumerate(merged):
            im = preprocessing.pil_image_to_array(img)
            array_axes[i].imshow(im)
            array_axes[i].scatter(points[i][0], points[i][1])  # new center
            array_axes[i].scatter(bboxes[i][0], bboxes[i][1], c='r')  # min
            array_axes[i].scatter(bboxes[i][2], bboxes[i][3], c='g')  # max
        plt.show()

    if TEST_ROTATION or ALL_TRUE:
        angles = [30, 45, 50]
        rotated = preprocessing.rotate_img_by_angles(pil_foreg, angles)
        print "rotated images {}".format(len(rotated))
        for i, im in enumerate(rotated):
            print "new size {}".format(im.size)
            im.show()

    if TEST_RESCALED or ALL_TRUE:
        ratios = [.7, .25, 2]
        rescaled = preprocessing.rescale_img_by_ratios(pil_foreg, ratios)
        print "rescaled images {}".format(len(rescaled))
        for i, im in enumerate(rescaled):
            print "new size {}".format(im.size)
            im.show()

    if TEST_MASK or ALL_TRUE:
        img = preprocessing.pil_image_to_array(pil_backg)
        preprocessing.apply_green_mask(img, plot=True)

    if TEST_FILTERS or ALL_TRUE:
        filtered = preprocessing.apply_filters(pil_backg, ['blur','detail'])
        print "filtered images {}".format(len(filtered))
        for i, im in enumerate(filtered):
            im.show()

    if TEST_LIGHT or ALL_TRUE:
        # factor < 1 lower illumination
        #
        factors = [.2, 1.8]
        new_ = preprocessing.change_light_img_by_factors(pil_backg, factors, show=True)

    if TEST_CONTRAST or ALL_TRUE:
        # factor < 1 lower illumination
        #
        factors = [.2, 1.8]
        new_ = preprocessing.change_contrast_img_by_factors(pil_backg, factors, show=True)

    if TEST_COLOR or ALL_TRUE:
        # factor < 1 lower illumination
        #
        #factors = [.2, 1.8]
        #new_ = preprocessing.change_color_img_by_factors(pil_backg, factors, show=True)
        _new = preprocessing.apply_green_mask_pil_img(pil_backg, plot=False)
        _new.show()

    if TEST_RESIZE or ALL_TRUE:
        import range_utils
        def choose_point(back_pil, fore_pil):
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
            if l_annotated > 0:
                if l_annotated == 1:
                    return annotated_points[0]
                else:
                    rnd_id = np.random.randint(0, l_annotated, 1)
                    return annotated_points[rnd_id]
            else:
                wb, hb = back_pil.size
                wf, hf = fore_pil.size
                center_x = range_utils.choice_n_rnd_numbers_from_to_linspace(0, wb - wf, wb - wf, 1)[0]
                center_y = range_utils.choice_n_rnd_numbers_from_to_linspace(0, hb - hf, hb - hf, 1)[0]
                return (center_x, center_y)

        for i in range(1,100):
            ex_background, ex_bbox = fut.get_random_test_bboxed_selected()

            pil_foreg = preprocessing.open_image(ex_background)
            pil_backg = preprocessing.open_image(ex_bbox)
            print "pre size {} {}".format(pil_foreg.size, pil_backg.size)
            preprocessing.check_merging_size(pil_backg, pil_foreg)
            print "post size {} {}".format(pil_foreg.size, pil_backg.size)
            chosen_point = choose_point(pil_backg, pil_foreg)
            _new, bb = preprocessing.merge_img_in_background(pil_backg,pil_foreg, [chosen_point])
            #_new[0].show()
