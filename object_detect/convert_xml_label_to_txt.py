# requirement
# original link
# https://github.com/Isabek/XmlToTxt
"""
declxml==0.9.1
"""
"""
xml and txt are two formats of object detection labels, they are same labels just in different format

different models take different format as input

"""

import sys
import logging
import os
import declxml as xml


class ObjectMapper(object):
    def __init__(self):
        self.processor = xml.user_object(
            "annotation", Annotation, [
                xml.user_object("size", Size, [
                    xml.integer("width"),
                    xml.integer("height"),
                ]),
                xml.array(
                    xml.user_object(
                        "object", Object, [
                            xml.string("name"),
                            xml.user_object(
                                "bndbox",
                                Box, [
                                    xml.floating_point("xmin"),
                                    xml.floating_point("ymin"),
                                    xml.floating_point("xmax"),
                                    xml.floating_point("ymax"),
                                ],
                                alias="box"
                            )
                        ]
                    ),
                    alias="objects"
                ),
                xml.string("filename")
            ]
        )

    def bind(self, xml_file_path, xml_dir):
        ann = xml.parse_from_file(
            self.processor, xml_file_path=os.path.join(xml_dir, xml_file_path)
        )
        ann.filename = xml_file_path
        return ann

    def bind_files(self, xml_file_paths, xml_dir):
        result = []
        for xml_file_path in xml_file_paths:
            try:
                result.append(self.bind(xml_file_path=xml_file_path, xml_dir=xml_dir))
            except Exception as e:
                logging.error("%s", e.args)
        return result


class Annotation(object):
    def __init__(self):
        self.size = None
        self.objects = None
        self.filename = None

    def __repr__(self):
        return "Annotation(size={}, object={}, filename={})".format(
            self.size, self.objects, self.filename
        )


class Size(object):
    def __init__(self):
        self.width = None
        self.height = None

    def __repr__(self):
        return "Size(width={}, height={})".format(self.width, self.height)


class Object(object):
    def __init__(self):
        self.name = None
        self.box = None

    def __repr__(self):
        return "Object(name={}, box={})".format(self.name, self.box)


class Box(object):
    def __init__(self):
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

    def __repr__(self):
        return "Box(xmin={}, ymin={}, xmax={}, ymax={})".format(
            self.xmin, self.ymin, self.xmax, self.ymax
        )


class Reader(object):
    def __init__(self, xml_dir):
        self.xml_dir = xml_dir

    def get_xml_files(self):
        xml_filenames = []
        for root, subdirectories, files in os.walk(self.xml_dir):
            for filename in files:
                if filename.endswith(".xml"):
                    file_path = os.path.join(root, filename)
                    file_path = os.path.relpath(file_path, start=self.xml_dir)
                    xml_filenames.append(file_path)
        return xml_filenames

    @staticmethod
    def get_classes(filename):
        with open(os.path.join(os.path.dirname(os.path.realpath('__file__')), filename), "r",
                  encoding="utf8") as f:
            lines = f.readlines()
            return {value: key for (key, value) in enumerate(list(map(lambda x: x.strip(), lines)))}


class Transformer(object):
    def __init__(self, xml_dir, out_dir, label_map):
        self.xml_dir = xml_dir
        self.out_dir = out_dir
        if label_map is None:
            self.label_map = dict()
            self.all_classes = list()
            self.label_map_exist = False
        else:
            self.label_map = label_map
            self.all_classes = list(label_map.values())
            self.label_map_exist = True


    def transform(self):
        reader = Reader(xml_dir=self.xml_dir)
        xml_files = reader.get_xml_files()
        object_mapper = ObjectMapper()
        annotations = object_mapper.bind_files(xml_files, xml_dir=self.xml_dir)
        if len(annotations) > 0:
            self.write_to_txt(annotations)

    def write_to_txt(self, annotations):
        for annotation in annotations:
            output_path = os.path.join(
                self.out_dir, self.darknet_filename_format(annotation.filename)
            )
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w+") as f:
                f.write(self.to_darknet_format(annotation))

    def to_darknet_format(self, annotation):
        result = []
        for obj in annotation.objects:
            if self.label_map_exist:
                try:
                    int_label = self.label_map[obj.name]
                except:
                    continue
            else:
                isin = obj.name in self.all_classes
                if not isin:
                    self.all_classes.append(obj.name)
                int_label = self.all_classes.index(obj.name)
                self.label_map[obj.name] =  int_label

            x, y, width, height = self.get_object_params(obj, annotation.size)
            result.append("%d %.6f %.6f %.6f %.6f" % (int_label, x, y, width, height))
        return "\n".join(result)

    @staticmethod
    def get_object_params(obj, size):
        image_width = 1.0 * size.width
        image_height = 1.0 * size.height

        box = obj.box
        absolute_x = box.xmin + 0.5 * (box.xmax - box.xmin)
        absolute_y = box.ymin + 0.5 * (box.ymax - box.ymin)

        absolute_width = box.xmax - box.xmin
        absolute_height = box.ymax - box.ymin

        x = absolute_x / image_width
        y = absolute_y / image_height
        width = absolute_width / image_width
        height = absolute_height / image_height

        return x, y, width, height

    @staticmethod
    def darknet_filename_format(filename):
        pre, ext = os.path.splitext(filename)
        return "%s.txt" % pre


def convert(xml_dir, out_dir, label_map=None, keep_empty=True):
    """

    @param xml_dir:
    @param out_dir:
    @param label_map: dict of int label map to label name, if None, will generate random mapping
    it has two purposes,
    1. when it is give it will use the label map give by the dict, so apply this method couple times won't
       same label won't be mapped to different int labels
    2. when lap_map dict only contains some of the labels, say all labels have [person, dog, cat], but label_map is
       {person: 0, cat: 1}, and the data created will omit dog annotations, on other word, dog will be treated as
       background

    @keep_empty: we can remove .txt files if there is no label in it, but we do suggest have some images without labels
    @return:
    """
    if not os.path.exists(xml_dir):
        print("Provide the correct folder for xml files.")
        sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.access(out_dir, os.W_OK):
        print("%s folder is not writeable." % out_dir)
        sys.exit()

    transformer = Transformer(xml_dir=xml_dir, out_dir=out_dir, label_map=label_map)
    transformer.transform()

    if not keep_empty:
        label_path = []
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.endswith(".txt"):
                    the_path = os.path.join(root, file)
                    label_path.append(the_path)
        for lb_p in label_path:
            with open(lb_p) as f:
                lines = f.read().splitlines()
            if len(lines) >0:
                continue
            else:
                os.remove(lb_p)

    return transformer.label_map


def demo():
    xml_data_dir = 'linmao_camera_data'
    output_txt_data_dir = 'linmao_camera_data_txt'
    label_map = {'nl_0438': 0,
                 'nl_0431': 1,
                 'nl_0239': 2,
                 'nl_0238': 3,
                 'nl_0271': 4,
                 'nl_0280': 5,
                 'nl_0433': 6,
                 'nl_0224': 7,
                 'nl_0098': 8}
    label_map = convert(xml_data_dir, output_txt_data_dir, label_map)
    print(label_map)
