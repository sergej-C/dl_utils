import itertools
from pprint import pprint
import copy
import collections
import numpy as np
import sys

class CombinatorialTransformer():
    def __init__(self, func_to_call, func_data, subject, tracer=None):
        """

        Example of func_to_call
        func_to_call = ['func1', 'func2', 'func3']

        Example of func_data to pass.
        func_data = {
            'func1': {
                'func': test.sum,
                'param': [0.7, .4, .2]
            },
            'func2': {
                'func': test.sum,
                'param': [12, 4]
            },
            'func3': {
                'func': test.sum,
                'param': [0.2, .3]
            },
        }

        :param func_to_call: a list of function name to search in func_data keys that will be applied to the object to transform
        :param func_data: a dict with association between function and params for that function
        :object: the object to pass to all functions
        :tracer: a tracer for merge old transformations to new ones
        """
        self.func_to_call = func_to_call
        self.func_data = func_data

        self.object = subject
        self.tracer = tracer
        self.input_data = []
        self.input_func = []

        # a list that will contain the transformed items
        self.all_items = []

        # create all combination of function and param
        for f in func_to_call:
            self.input_data.append(func_data[f]['param'])
            self.input_func.append(func_data[f]['func'])

        self.combinatorial_params = list(itertools.product(*self.input_data))
        pprint(self.combinatorial_params)

    def apply_transformations(self):
        """
        call each function with associated params
        :return:
        """

        # every r is an item to which apply a list of functions
        for r in self.combinatorial_params:

            tracer = TransformationTracer(self.object, copy.deepcopy(self.tracer))

            # get param to pass and call related func
            for idx, el in enumerate(r):


                #print "transform {} with {} and param {}".format(tracer.obj,self.input_func[idx], el)
                tracer.add_func_param_and_update(self.input_func[idx], el)

            self.all_items.append(tracer)



class TransformationTracer():
    """
    trace a list of functions and params applied to its object
    """

    # the obj on which apply some transformation
    obj = None

    # transformation applied
    applied_funcs = []
    applied_funcs_names = []

    # params of the transformation
    params = []

    def __init__(self, obj, tracer=None):
        self.obj = obj

        if tracer is not None:
            #print tracer.applied_funcs_names
            self.applied_funcs = tracer.applied_funcs
            self.applied_funcs_names = tracer.applied_funcs_names
            self.params = tracer.params
        else:
            self.applied_funcs = []
            self.applied_funcs_names = []
            self.params = []

    def add_func(self, f):
        self.applied_funcs.append(f)
        self.applied_funcs_names.append(f.__name__)
        #print "appended f {} funcs {}".format(f.__name__, self.applied_funcs_names)


    def add_param(self, p):
        self.params.append(p)
        #print "appended p {} param {}".format(p, self.params)

    def add_func_param(self, f, p):
        self.add_func(f)

        # if instance conditional param
        # check on conditional rules
        # add param based on result
        #if isinstance(p, 'ConditionalParam'):
        #    p.check_conditions(self.obj)
        #    p=p.conditional_param
        self.add_param(p)

    def get_func_param(self):
        return self.applied_funcs_names, self.params

    def update_obj(self):
        f = self.applied_funcs[-1]
        p = self.params[-1]

        #
        # no params
        if p == np.NaN:
            self.obj = f(self.obj)
        else:
            self.obj = f(self.obj, p)

    def add_func_param_and_update(self, f, p):
        self.add_func_param(f, p)
        self.update_obj()

    def ensure_iterable(self, el):
        if not isinstance(el, collections.Iterable):
            return [el]

    def clear(self):
        del self.obj
        del self.applied_funcs
        del self.applied_funcs_names

    @staticmethod
    def merge(tracer1, tracer2):
        tracer = TransformationTracer(tracer1.obj, tracer1)
        tracer.applied_funcs += tracer2.applied_funcs
        tracer.applied_funcs_names += tracer2.applied_funcs_names
        tracer.params += tracer2.params
        """print "========================================================"
        print "tracer1 func {}".format(tracer1.applied_funcs_names)
        print "tracer2 func {}".format(tracer2.applied_funcs_names)
        print "new tracer func {}".format(tracer.applied_funcs_names)"""
        return tracer

class ConditionalParam():

    def __init__(self, conditional_func):
        self.conditional_param=None
        self.conditional_func=conditional_func

    def check_conditions(self, obj):
        self.conditional_param = self.conditional_func(obj)

if __name__ == '__main__':
    class test():
        @staticmethod
        def sum(x, y):
            return x + y


    #
    # example usage:
    #
    func_data1 = {
        'func1': {
            'func': test.sum,
            'param': [0.7, .4, .2]
        },
        'func2': {
            'func': test.sum,
            'param': [12, 4]
        },
        'func3': {
            'func': test.sum,
            'param': [0.2, .3]
        },
    }

    func_to_call1 = ['func1', 'func3']
    object = 4

    transformer = CombinatorialTransformer(func_to_call1, func_data1, object)

    transformer.apply_transformations()

    for el in (transformer.all_items):
        pprint("{} {} {}".format(el.obj, el.applied_funcs_names, el.params))

    new_object = 5
    tracer0 = transformer.all_items[0]
    transformer2 = CombinatorialTransformer(func_to_call1, func_data1, new_object, tracer0)
    transformer2.apply_transformations()

    for el in transformer2.all_items:
        pprint("{} {} {}".format(el.obj, el.applied_funcs_names, el.params))
