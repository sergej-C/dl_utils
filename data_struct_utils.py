from itertools import izip

def two_list_to_dict(list1, list2):
    i = iter(list1)
    i2 = iter(list2)
    return dict(izip(i, i2))

def merge_lists(list1, list2):
    d = dict(list1, **list2)
    return d


def list_is_empty(list):
    return all(v is None or len(v) == 0 for v in list)


def rm_empty_from_list_of_list(list):
    new_l = {}
    for l in list.keys():
        nl = rm_empty_from_list(list[l])
        if len(nl)!=0:
            new_l[str(l)] = nl
    return new_l

def rm_empty_from_list(list):
    new_l = {}
    for l in list.keys():
        if len(list[l])!=0:
            new_l[str(l)]=list[l]
    return new_l

def get_where_more_than_x_sub_el(list, x=1):
    new_l = {}
    for l in list.keys():
        if len(list[l]) != 0:
            for k in list[l]:
                if len(list[l][k])>x:
                    if not new_l.has_key(l):
                        new_l[l]={}
                        new_l[l][k]=list[l][k]
    return new_l


def get_where_more_than_x_el(list, x=1, empty_first=True):
    if empty_first:
        list = rm_empty_from_list_of_list(list)

    new_l = {}
    for l in list.keys():
        if len(list[l]) > x:
            new_l[l]=list[l]
    return new_l