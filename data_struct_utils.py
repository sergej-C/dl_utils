from itertools import izip

def two_list_to_dict(list1, list2):
    i = iter(list1)
    i2 = iter(list2)
    return dict(izip(i, i2))

def merge_lists(list1, list2):
    d = dict(list1, **list2)
    return d