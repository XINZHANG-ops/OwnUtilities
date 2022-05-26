import os
import numpy as np
import math
from tqdm import tqdm
import shutil
# image read libraries
import cv2


def plot_box(images_dir, labels_dir, out_put_dir, thickness=3, filter_diff=True):
    """
    plot the box with labels on the images

    @param images_dir:
    @param labels_dir:
    @param out_put_dir:
    @param thickness:
    @param filter_diff:
    @return:
    """
    if os.path.exists(out_put_dir):
        shutil.rmtree(out_put_dir)
        os.makedirs(out_put_dir)
    else:
        os.makedirs(out_put_dir)

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
        image_path = [
            i for i in image_path if os.path.splitext(i.split(os.sep)[-1])[0] in common_names
        ]
        label_path = [
            i for i in label_path if os.path.splitext(i.split(os.sep)[-1])[0] in common_names
        ]

    image_path = sorted(image_path)
    label_path = sorted(label_path)

    labels_colors = dict()

    image_label_path_pair = list(zip(image_path, label_path))

    for img_p, lab_p in tqdm(image_label_path_pair):
        img = cv2.imread(img_p)
        img_array = np.array(img)

        # Note that some pictures doesn't have channel
        try:
            hight, width, channel = img_array.shape
        except:
            hight, width = img_array.shape
        with open(lab_p) as f:
            lines = f.read().splitlines()

        for l in lines:
            line_split = l.split(' ')
            label = int(line_split[0])
            if label in labels_colors:
                color = labels_colors[label]
            else:
                color = list(np.random.choice(range(256), size=3))
                color = (int(color[0]), int(color[1]), int(color[2]))
                labels_colors[label] = color

            x, y, w, h = tuple([float(i) for i in line_split[1:5]])

            # take max or min to make sure the crop is within the image
            top_left_x = max(math.floor(x * width - w * width / 2), 0)
            top_left_y = max(math.floor(y * hight - h * hight / 2), 0)
            bot_right_x = min(math.floor(x * width + w * width / 2), width)
            bot_right_y = min(math.floor(y * hight + h * hight / 2), hight)

            img = cv2.rectangle(
                img, (top_left_x, top_left_y), (bot_right_x, bot_right_y), color, thickness
            )
            """
            cv2.putText(img,'Hello World!', 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
                    """
            cv2.putText(
                img, f'{label}', (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                thickness, 2
            )
        image_name = ''.join(os.path.splitext(img_p.split(os.sep)[-1]))
        cv2.imwrite(os.path.join(out_put_dir, image_name), img)
