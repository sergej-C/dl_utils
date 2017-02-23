import numpy as np
import PIL.Image
import PIL.ImageOps
from skimage import data, color, io, img_as_float
import PIL.ImageFilter as ImageFilter
from PIL import ImageEnhance

import cv2

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
        im_arr = im_arr.reshape((pil_img.size[1], pil_img.size[0], 3))
        return im_arr

    @staticmethod
    def paste_centered_in_point(pil_backg, pil_foreg, points, i):
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

        pil_backg_cpy.paste(pil_foreg, (cent_x, cent_y), pil_foreg)
        xmin = cent_x
        ymin = cent_y
        xmax = cent_x + w
        ymax = cent_y + h

        return xmin, ymin, xmax, ymax, pil_backg_cpy

    @staticmethod
    def merge_img_in_background(pil_backg, pil_foreg, points):
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
            xmin, ymin, xmax, ymax, pil_backg_merged = preprocessing.paste_centered_in_point(pil_backg, pil_foreg, points, i)
            merged.append(pil_backg_merged)
            bboxes.append(np.array((xmin, ymin, xmax, ymax)))
        return merged, bboxes

    @staticmethod
    def open_image(path):
        return PIL.Image.open(path)


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
            rotated_im = pil_img.rotate(a, expand=True)
            rotated.append(rotated_im)
            new_sizes.append(np.array((rotated_im.size)))
        return rotated, new_sizes

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
            rescaled_im = pil_img.resize(size=(int(np.ceil(w*r)), int(np.ceil(h*r))))
            rescaled.append(rescaled_im)
            new_sizes.append(np.array((rescaled_im.size)))
        return rescaled, new_sizes

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

        return img_masked

    @staticmethod
    def apply_blue_mask(img, plot=False):
        return preprocessing.apply_rgb_mask(img, [0, 1, 1], plot=plot)

    @staticmethod
    def apply_green_mask(img, plot=False):
        return preprocessing.apply_rgb_mask(img, [0, 1, 0], plot=plot)

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

        return filtered, applied_filters


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


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import fish_utils as fu

    TEST_MERGE=False
    TEST_ROTATION=False
    TEST_RESCALED=False
    TEST_MASK=False
    TEST_FILTERS=False
    TEST_LIGHT=False
    TEST_COLOR=True
    TEST_CONTRAST=False

    fut = fu.fish_utils()
    ex_background = fut.get_test_selected_background()
    ex_bbox = fut.get_test_bboxed_selected()

    pil_backg = preprocessing.open_image(ex_background)
    pil_foreg = preprocessing.open_image(ex_bbox)

    if TEST_MERGE:
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

    if TEST_ROTATION:
        angles = [30, 45, 50]
        rotated, sizes = preprocessing.rotate_img_by_angles(pil_foreg, angles)
        print "rotated images {}".format(len(rotated))
        for i, im in enumerate(rotated):
            print "new size {}".format(sizes[i])
            im.show()

    if TEST_RESCALED:
        ratios = [.7, .25, 2]
        rescaled, sizes = preprocessing.rescale_img_by_ratios(pil_foreg, ratios)
        print "rescaled images {}".format(len(rescaled))
        for i, im in enumerate(rescaled):
            print "new size {}".format(sizes[i])
            im.show()

    if TEST_MASK:
        img = preprocessing.pil_image_to_array(pil_backg)
        preprocessing.apply_green_mask(img, plot=True)

    if TEST_FILTERS:
        filtered, filters = preprocessing.apply_filters(pil_backg, ['blur','detail'])
        print "filtered images {}".format(len(filtered))
        for i, im in enumerate(filtered):
            print "filter {}".format(filters[i])
            im.show()

    if TEST_LIGHT:
        # factor < 1 lower illumination
        #
        factors = [.2, 1.8]
        new_ = preprocessing.change_light_img_by_factors(pil_backg, factors, show=True)

    if TEST_CONTRAST:
        # factor < 1 lower illumination
        #
        factors = [.2, 1.8]
        new_ = preprocessing.change_contrast_img_by_factors(pil_backg, factors, show=True)

    if TEST_COLOR:
        # factor < 1 lower illumination
        #
        factors = [.2, 1.8]
        new_ = preprocessing.change_color_img_by_factors(pil_backg, factors, show=True)