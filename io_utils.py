def save_list_to_txt(list_dest, list):
    # save list of images
    thefile = open(list_dest, 'a')
    for item in list:
        thefile.write("%s\n" % item)


def save_text_in_file(text, file_path):
    with open(file_path, 'w') as f:
        f.write(text)

def append_txt_to_file(text, file_path):
    with open(file_path, 'a') as f:
        f.write(text+'\n')

def ensure_last_slash(path):

    if len(path)==0:
        return
    if path[-1]!='/':
        return path+'/'
    else:
        return path