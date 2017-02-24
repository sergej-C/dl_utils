from glob import glob


# from https://pymotw.com/2/xml/etree/ElementTree/create.html
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# from xml.etree import ElementTree###############
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    root = elem.getroot()
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


##################################################

def get_xmls(path_xmls):
    xml_glob = path_xmls + "/*.xml"
    path_xmls = glob(xml_glob)
    return path_xmls


def read_xml(filename):
    return ET.ElementTree(file=filename)


def parse_file(filename):
    return ET.parse(filename)


def get_and_check_img_name(tree, path):
    """
    get image name from annotation file and from xml tree
    verify that are equal
    return:
        mismatch: false if are equal
        img_name_from_path: img name without extension
        img_from_annotation: img name with extension
    """
    annotation = tree.find('filename')
    img_from_annotation = annotation.text
    img_name_from_path = path.split('/')[-1][:-4]
    mismatch = False
    if img_name_from_path != img_from_annotation[:-4]:
        mismatch = True

    return mismatch, img_name_from_path, img_from_annotation


    def create_xml(self, root_node_name):
        return ET.Element(root_node_name)

    def save_tree(self, root_el, filename_no_ext, add_ext=False):
        tree = ET.ElementTree(root_el)
        if add_ext:
            fn = filename_no_ext + ".xml"
        else:
            fn = filename_no_ext
        tree.write(fn)

    def add_file_name(self, anno_root, image_name):
        el = ET.SubElement(anno_root, "filename").text = image_name
        return el

    #       <object>
    #               <name>boat</name>
    #               <pose>Unspecified</pose>
    #               <truncated>0</truncated>
    #               <difficult>1</difficult>
    #               <bndbox>
    #                       <xmin>440</xmin>
    #                       <ymin>226</ymin>
    #                       <xmax>455</xmax>
    #                       <ymax>261</ymax>
    #               </bndbox>
    #       </object>
    def add_node_to_annotation(self, anno_root, class_name, pose='Unspecified', difficult="0", xmin="0", ymin="0",
                               xmax="0", ymax="0"):

        obj = ET.SubElement(anno_root, "object")
        cln = ET.SubElement(obj, "name").text = class_name
        pose = ET.SubElement(obj, "pose").text = pose
        difficult = ET.SubElement(obj, "difficult").text = difficult
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = xmin
        ET.SubElement(bbox, "ymin").text = ymin
        ET.SubElement(bbox, "xmax").text = xmax
        ET.SubElement(bbox, "ymax").text = ymax
        return obj

    def create_annotation(self):
        return self.create_xml("annotation")