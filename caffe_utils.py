import sys
import os
import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis
import caffe
import errno

class caffe_utils():
    
    MU = None
    g_transformer = None
    CAFFE_PATH=''
    EXAMPLE_PATH=''
    model_def = ''
    model_wights = ''
    CAFFE_TEST_MODELS_PATH=''
    
    def __init__(self, caffe_path=None, caffe_test_model_path=None, caffe_test_model_name='caffe_test_models', debug=False, mode=1):        
        """
        if caffe_test_model_path is not set, current working dir is selected 
        and a folder with name caffe_test_model_name is created
        if caffe_path is None, searching for environ variable CAFFE_PATH
        default set gpu mode
        """
        sys.path.append(self.CAFFE_PATH+'/python')

        if caffe_path==None:
            self.CAFFE_PATH=os.environ.get('CAFFE_PATH')
            if self.CAFFE_PATH=='':
                print 'error caffe path not setted!!'
        else:
            self.CAFFE_PATH=caffe_path

        self.EXAMPLE_PATH=self.CAFFE_PATH+'/examples'
        
        sys.path.append(self.CAFFE_PATH+'/python')
 
    
        self.model_def = self.CAFFE_PATH + '/models/bvlc_reference_caffenet/deploy.prototxt'
        self.model_weights = self.CAFFE_PATH + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        self.DEBUG_MODE=debug
    
        if caffe_test_model_path==None:
            self.CAFFE_TEST_MODELS_PATH = os.getcwd() + '/'+caffe_test_model_name

        if not os.path.exists(self.CAFFE_TEST_MODELS_PATH):
            try:
                os.mkdir(self.CAFFE_TEST_MODELS_PATH)
                print "created caffe test models folder as {}".format(self.CAFFE_TEST_MODELS_PATH)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise exc
        else:
                print "caffe test models folder exists: {}".format(self.CAFFE_TEST_MODELS_PATH)

        self.change_mode(mode)


    def change_mode(self, mode=1):
        if mode==0:
            caffe.set_mode_cpu()
            self.mode='cpu'
        elif mode==1:
            caffe.set_mode_gpu()
            self.mode='gpu'
        else:
            print "error mode can be either 1=gpu or 0=cpu, setting cpu mode as default"
            self.mode = 'cpu'
            caffe.set_mode_cpu() 

    def get_caffe_path(self):
        return self.CAFFE_PATH

    def get_caffe_test_models_path(self):
        return self.CAFFE_TEST_MODELS_PATH

    def get_images_mean(self):        
        if self.MU is None:
            # load the mean ImageNet image (as distributed with Caffe) for subtraction
            mu = np.load(self.CAFFE_PATH + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
            mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
            print 'mean-subtracted values:', zip('BGR', mu)
            self.MU = mu
            return mu
        else:
            return self.MU

    def get_transformer(self, net, mu=None, layer_name='data', swap_channels=True):
        
        if mu==None:
            mu=self.get_images_mean()

        if self.g_transformer is None:
            # create transformer for the input called 'data'
            transformer = caffe.io.Transformer({'data': net.blobs[layer_name].data.shape})

            transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
            transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
            transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            if swap_channels:
                transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
            self.g_transformer = transformer
            return  transformer
        else:
            return self.g_transformer


    def reshape_input_in_net(self, net, batch_size=1, channels=3, width=227, height=227, layer_name='data'):
        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        net.blobs[layer_name].reshape(batch_size,        # batch size
                              channels,         # 3-channel (BGR) images
                              height, width)  # image size is 227x227 or 

    def preprocess_image_in_net(
            self,
            net, 
            img_path=None,
            show=False,
            batch_size=1, channels=3, width=227, height=227
        ):        
        """
        gets image and apply transformation for feeding it into a caffe layer
        """
        if img_path==None:        
            img_path = self.CAFFE_PATH + '/examples/images/cat.jpg'

        transformer = self.get_transformer(net)
        self.reshape_input_in_net(net, batch_size=batch_size, channels=channels, width=width, height=height)

        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        if show:
            plt.imshow(image)

        return transformed_image

    def forward_image_to_net(
            self,
            net, 
            data_layer_name='data', 
            img_path=None, 
            show=False,
            batch_size=1, channels=3, width=227, height=227
        ):

        if img_path==None:
            img_path = self.CAFFE_PATH + '/examples/images/cat.jpg'

        prep_img = self.preprocess_image_in_net(
            net, 
            show=show, 
            img_path=img_path, 
            batch_size=batch_size, 
            channels=channels,
            width=width,
            height=height)

        # copy the image data into the memory allocated for the net
        net.blobs[data_layer_name].data[...] = prep_img

        ### perform classification
        output = net.forward()

        return output

    def deprocess_img(self, transformer_out, show=False):
        
        deproc_img = self.g_transformer.deprocess('data', transformer_out)
        if show:
            plt.imshow(deproc_img)
        return deproc_img

    def show_filters(self, net, use_custom_min_max=True):
        # forward the data 
        net.forward()
        
        # create a plot
        plt.figure()
        
        # set min and  max value of imshow based on data min and max
        filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
        
        print net.blobs['conv'].data.shape
        
        for i in range(3):
            plt.subplot(1,4,i+2)
            plt.title("filter #{} output".format(i))
            plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
            
            # adjust spaces
            plt.tight_layout()

            # no axis
            plt.axis('off')
            # view color range
            #cbar = plt.colorbar()   

    def vis_square(self, data):
        """Take an array of shape (n, height, width) or (n, height, width, 3)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
        
        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())
        
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                   (0, 1), (0, 1))                 # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        
        plt.imshow(data); plt.axis('off') 

    def get_caffe_net(self, mode=caffe.TEST):
        caffe_net = self.create_net(
                self.model_def,      # defines the structure of the model
                self.model_weights,  # contains the trained weights
                mode=mode)     # use test mode (e.g., don't perform dropout)
        return caffe_net

    def create_net(self, proto_filename, path=None, weights_file='', mode=caffe.TEST):
        """
        create a net from the specified proto file searched inside path (default caffe_test_model_path)
        file searched as proto_filename.prototxt file 
        default mode = caffe.TEST
        """
        if path==None:
            path=self.CAFFE_TEST_MODELS_PATH
        mpath = path + '/' + proto_filename + '.prototxt'
 
        if self.DEBUG_MODE:
            print "opening model in {}, weights {}, mode {}".format(
                mpath,
                weights_file,
                mode
            )
 
        if not os.path.exists(mpath):
            print "error, file not exists! {}".format(mpath)
            return

        if weights_file != '':
         if not os.path.exists(weights_file):
            print "error, file not exists! {}".format(weights_file)
            return
        if weights_file!='':
            return caffe.Net(mpath, weights_file, mode)
        else:
            return caffe.Net(mpath, mode)

    def get_imagenet_labels(self):

        labels = []
        labels_file = self.CAFFE_PATH + '/data/ilsvrc12/synset_words.txt'
        if not os.path.exists(labels_file):
            print("ATTENTION!!! file synset_words.txt in caffe data path doesn't exist! download it, no imagenet labels...")
            return False
        else:
            labels = np.loadtxt(labels_file, str, delimiter='\t')
         
        return labels

    def get_predicted_class(self, forward_out, batch_index=0, n_top=5):
        """
        get the argmax of probabilities returned by a forward pass of a net [test on caffenet classifier, todo test on other models]
        and relative labels from imagenets labels
        for the batch sample at index batch_index [default the first]
        return output_prob.argmax(), labels[index_max] (if file of imagenet exists in caffe data path)
        """
        output_prob = forward_out['prob'][batch_index]

        labels = self.get_imagenet_labels()
        
        best_labels = ''        
        index_max = output_prob.argmax()        
        if labels!=False:
            best_labels=labels[index_max]

        top_inds = output_prob.argsort()[::-1][:n_top]  # reverse sort and take five largest items

        top_n_cats = zip(output_prob[top_inds], labels[top_inds])

        return index_max, best_labels, top_n_cats

    def classify_img(self, caffe_net=None, img_path=None, show=False):
        
        if img_path==None:
            img_path = self.CAFFE_PATH + '/examples/images/cat.jpg'
    
        if caffe_net==None:
            caffe_net=self.get_caffe_net()

        output = self.forward_image_to_net(caffe_net, img_path=img_path, show=show)
        return self.get_predicted_class(output)

        
    def show_layers(self, net):                
        # for each layer, show the output shape
        for layer_name, blob in net.blobs.iteritems():
            print layer_name + '\t' + str(blob.data.shape)

    def show_params(self, net):
        for layer_name, param in net.params.iteritems():
            print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape) 

    def describe_net(self, net):

        print("===================LAYERS=======================================")
        self.show_layers(net)
        print("-------------------PARAMS---------------------------------------")
        self.show_params(net)
        print("================================================================")


    def vis_filters(self, net, layer_name='conv1', filter_index=0):

        filters = net.params[layer_name][filter_index].data
        self.vis_square(filters.transpose(0, 2, 3, 1)) #todo - valid for different filter size?

    def vis_net_filters(self, net, layer_name):

        for fidx in  range(len(net.params[layer_name])):
            self.vis_filters(net, layer_name, fidx)


    def save_spec_to_file(self, net_spec, model_name, path=None, overwrite=False):
        if path==None:
            path=self.CAFFE_TEST_MODELS_PATH

        fpath = path+'/'+model_name+'.prototxt'
        if os.path.exists(fpath) and overwrite==False:
            print "file already exists, specify overwrite=True for substitution"            
            return

        with open(fpath, 'w') as f:
            f.write(str(net_spec.to_proto()))

    def print_net_params(self, net, lay_name):
        if net.params.has_key(lay_name):
            
            print('[==  PARAMS  ==]')
            print('Wheights ====>')
            prm = net.params[lay_name][0].data
            print(prm.shape)
            print(prm)
            
            print('Bias ====>')
            b = net.params[lay_name][1].data
            print(b.shape)
            print(b)
            

    def print_net_blob_data(self, net, lay_name):
        print('[==  BLOB DATA  ==]')
        if net.blobs.has_key(lay_name):
            bd = net.blobs[lay_name].data
            print(bd.shape)
            print bd

    def print_net_data(self, net):
        for ln,bl in net.blobs.iteritems():        
            print("[-------------- {} -------------]").format(ln)
            self.print_net_blob_data(net, ln)
            self.print_net_params(net, ln)    
            print("[--------------END {} -------------]\n\n").format(ln)
