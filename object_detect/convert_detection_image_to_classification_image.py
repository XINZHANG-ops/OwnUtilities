import os
import numpy as np
import math
from tqdm import tqdm
# image read libraries
from PIL import Image
import matplotlib.pyplot as plt
import shutil


class obj_to_cls:
    def __init__(self, images_dir, labels_dir=None, filter_diff=True):
        """

        @param images_dir: .jpg format images
        @param labels_dir: txt format labels
        @param filter_diff: if filter image and label files with name doesn't match
        """
        image_path = []
        label_path = []
        image_names = set()
        label_names = set()

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith(".jpg"):
                    if 'checkpoint' in file:
                        pass
                    else:
                        file_path = os.path.join(root, file)
                        image_names.add(os.path.splitext(file_path.split(os.sep)[-1])[0])
                        image_path.append(file_path)

        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                if file.endswith(".txt"):
                    if 'checkpoint' in file:
                        pass
                    else:
                        file_path = os.path.join(root, file)
                        label_names.add(os.path.splitext(file_path.split(os.sep)[-1])[0])
                        label_path.append(file_path)

        common_names = image_names.intersection(label_names)

        if filter_diff:
            image_path = [i for i in image_path if os.path.splitext(i.split(os.sep)[-1])[0] in common_names]
            label_path = [i for i in label_path if os.path.splitext(i.split(os.sep)[-1])[0] in common_names]

        image_path = sorted(image_path)
        label_path = sorted(label_path)

        self.image_path = image_path
        self.label_path = label_path
        self.image_label_path_pair = list(zip(image_path, label_path))

    @staticmethod
    def create_folder(path, sub_dirs):
        for sub in sub_dirs:
            create_dir = os.path.join(path, sub)
            if os.path.exists(create_dir):
                shutil.rmtree(create_dir)
                os.makedirs(create_dir)
            else:
                os.makedirs(create_dir)

    @staticmethod
    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + " (" + str(counter) + ")" + extension
            counter += 1
        return path

    def test_if_unique_data_pairs(self):
        # TEST if any image and label doesn't match
        for ip, lp in self.image_label_path_pair:
            ip_name = '.'.join(ip.split('/')[-1].split('.')[:-1])
            lp_name = '.'.join(lp.split('/')[-1].split('.')[:-1])
            if ip_name == lp_name:
                pass
            else:
                print(ip_name, lp_name)
                print("name doesn't match")

    def convert_to_classification_data(self, save_path):
        """
        This will ignore the train test structure of original directory, and create a new directory contains each category as a subdirectory
        @param save_path:
        @return:
        """
        import nltk
        from tqdm import tqdm
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            os.mkdir(save_path)

        current_labels = nltk.defaultdict(int)

        for img_dir, label_dir in tqdm(self.image_label_path_pair):
            ip_name = img_dir.split('/')[-1].split('.')[0]
            img = Image.open(img_dir)
            # plt.figure(figsize=(15, 8))
            # plt.imshow(img)
            img_array = np.array(img)  # convert to np array

            # Note that some pictures doesn't have channel
            try:
                hight, width, channel = img_array.shape
            except:
                hight, width = img_array.shape
            with open(label_dir) as f:
                lines = f.read().splitlines()
            for l in lines:
                line_split = l.split(' ')
                label = int(line_split[0])
                current_labels[label] += 1
                x, y, w, h = tuple([float(i) for i in line_split[1:5]])

                top_left_x = math.floor(x * width - w * width / 2)
                top_left_y = math.floor(y * hight - h * hight / 2)
                bot_right_x = math.floor(x * width + w * width / 2)
                bot_right_y = math.floor(y * hight + h * hight / 2)

                img2 = img.crop((top_left_x, top_left_y, bot_right_x, bot_right_y))
                # plt.figure(figsize=(15, 8))
                # plt.imshow(img2)

                save_dir = os.path.join(save_path, str(label), str(current_labels[label]) + '.jpg')

                try:
                    img2.save(save_dir)
                except:
                    os.mkdir(os.path.join(save_path, str(label)))
                    img2.save(save_dir)

    def convert_to_classification_data_same_structure(self, new_dir):
        """
        This function will remain the original directory structure and if one image has multiple object
        the save name will be added (1) or (2) etc
        @param self:
        @param new_dir:
        @return:
        """
        all_image_path = []
        for ip in self.image_path:
            all_image_path.append(os.path.join(*ip.split(os.sep)[1:-1]))

        all_label_path = []
        for lp in self.label_path:
            all_label_path.append(os.path.join(*lp.split(os.sep)[1:-1]))

        all_image_path = list(set(all_image_path))
        all_label_path = list(set(all_label_path))
        obj_to_cls.create_folder(new_dir, all_image_path)

        for im_p, lb_p in tqdm(self.image_label_path_pair):
            img = Image.open(im_p)
            img_array = np.array(img)  # convert to np array

            # Note that some pictures doesn't have channel
            try:
                hight, width, channel = img_array.shape
            except:
                hight, width = img_array.shape
            with open(lb_p) as f:
                lines = f.read().splitlines()
            for l in lines:
                line_split = l.split(' ')
                label = int(line_split[0])
                x, y, w, h = tuple([float(i) for i in line_split[1:5]])

                top_left_x = math.floor(x * width - w * width / 2)
                top_left_y = math.floor(y * hight - h * hight / 2)
                bot_right_x = math.floor(x * width + w * width / 2)
                bot_right_y = math.floor(y * hight + h * hight / 2)

                img2 = img.crop((top_left_x, top_left_y, bot_right_x, bot_right_y))

                save_dir = obj_to_cls.uniquify(os.path.join(new_dir, os.path.join(*im_p.split(os.sep)[1:])))
                img2.save(save_dir)




def demo():
    # from object_detect.convert_detection_image_to_classification_image import obj_to_cls
    print('The function will check all files end with .jpg and .txt')
    obj2cls = obj_to_cls('coco128', 'coco128')
    print('check if image and label are 1 to 1')
    obj2cls.test_if_unique_data_pairs()
    obj2cls.convert_to_classification_data('coco128_classification')
    obj2cls.convert_to_classification_data_same_structure('coco128_classification2')