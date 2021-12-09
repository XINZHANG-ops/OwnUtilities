import os
import numpy as np
import math

# image read libraries
from PIL import Image
import matplotlib.pyplot as plt
import shutil


class obj_to_cls:
    def __init__(self, images_dir, labels_dir):
        """

        @param images_dir: .jpg format images
        @param labels_dir: txt format labels
        """
        image_path = []
        label_path = []

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith(".jpg"):
                    if 'checkpoint' in file:
                        pass
                    else:
                        image_path.append(os.path.join(root, file))
        image_path = sorted(image_path)

        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                if file.endswith(".txt"):
                    if 'checkpoint' in file:
                        pass
                    else:
                        label_path.append(os.path.join(root, file))

        label_path = sorted(label_path)

        self.image_path = image_path
        self.label_path = label_path
        self.image_label_path_pair = list(zip(image_path, label_path))

    def test_if_unique_data_pairs(self):
        # TEST if any image and label doesn't match
        for ip, lp in self.image_label_path_pair:
            ip_name = ip.split('/')[-1].split('.')[0]
            lp_name = lp.split('/')[-1].split('.')[0]
            if ip_name == lp_name:
                pass
            else:
                print(ip_name, lp_name)
                print("name doesn't match")

    def convert_to_classification_data(self, save_path):
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
                x, y, w, h = tuple([float(i) for i in line_split[1:]])

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



def demo():
    print('The function will check all files end with .jpg and .txt')
    obj2cls = obj_to_cls('coco128', 'coco128')
    print('check if image and label are 1 to 1')
    obj2cls.test_if_unique_data_pairs()
    obj2cls.convert_to_classification_data('coco128_classification')