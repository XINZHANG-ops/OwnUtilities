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
        self.processor = xml.user_object("annotation", Annotation, [
            xml.user_object("size", Size, [
                xml.integer("width"),
                xml.integer("height"),
            ]),
            xml.array(
                xml.user_object("object", Object, [
                    xml.string("name"),
                    xml.user_object("bndbox", Box, [
                        xml.floating_point("xmin"),
                        xml.floating_point("ymin"),
                        xml.floating_point("xmax"),
                        xml.floating_point("ymax"),
                    ], alias="box")
                ]),
                alias="objects"
            ),
            xml.string("filename")
        ])

    def bind(self, xml_file_path, xml_dir):
        ann = xml.parse_from_file(self.processor, xml_file_path=os.path.join(xml_dir, xml_file_path))
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
        return "Annotation(size={}, object={}, filename={})".format(self.size, self.objects, self.filename)


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
        return "Box(xmin={}, ymin={}, xmax={}, ymax={})".format(self.xmin, self.ymin, self.xmax, self.ymax)


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
        with open(os.path.join(os.path.dirname(os.path.realpath('__file__')), filename), "r", encoding="utf8") as f:
            lines = f.readlines()
            return {value: key for (key, value) in enumerate(list(map(lambda x: x.strip(), lines)))}



class Transformer(object):
    def __init__(self, xml_dir, out_dir, class_file):
        self.xml_dir = xml_dir
        self.out_dir = out_dir
        self.class_file = class_file

    def transform(self):
        reader = Reader(xml_dir=self.xml_dir)
        xml_files = reader.get_xml_files()
        classes = reader.get_classes(self.class_file)
        object_mapper = ObjectMapper()
        annotations = object_mapper.bind_files(xml_files, xml_dir=self.xml_dir)
        self.write_to_txt(annotations, classes)

    def write_to_txt(self, annotations, classes):
        for annotation in annotations:
            output_path = os.path.join(self.out_dir, self.darknet_filename_format(annotation.filename))
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w+") as f:
                f.write(self.to_darknet_format(annotation, classes))

    def to_darknet_format(self, annotation, classes):
        result = []
        for obj in annotation.objects:
            if obj.name not in classes:
                print("Please, add '%s' to classes.txt file." % obj.name)
                exit()
            x, y, width, height = self.get_object_params(obj, annotation.size)
            result.append("%d %.6f %.6f %.6f %.6f" % (classes[obj.name], x, y, width, height))
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

def convert(xml_dir, out_dir, class_file='classes.txt'):
    """

    @param xml_dir:
    @param out_dir:
    @param class_file: txt file contains all labels, each label per line
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

    if not os.access(class_file, os.F_OK):
        print("%s file is missing." % class_file)
        sys.exit()

    if not os.access(class_file, os.R_OK):
        print("%s file is not readable." % class_file)
        sys.exit()

    transformer = Transformer(xml_dir=xml_dir, out_dir=out_dir, class_file=class_file)
    transformer.transform()