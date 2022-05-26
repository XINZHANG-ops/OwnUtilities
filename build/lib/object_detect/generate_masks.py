import torch
import os
import numpy as np
import math
from tqdm import tqdm
import random
# image read libraries
from PIL import Image
import matplotlib.pyplot as plt
import shutil


def create_mask_for_object(
    img_dir,
    label_dir,
    save_dir,
    expand_ratio_width=0,
    expand_ratio_hight=0,
    img_type='jpg',
    save_type='png'
):
    """
    This function will make where there is a box as mask
    @param img_dir:
    @param label_dir:
    @param save_dir:
    @param expand_ratio_width:
    @param expand_ratio_hight:
    @param img_type:
    @param save_type:
    @return:
    """
    image_path = []
    label_path = []
    os.makedirs(save_dir, exist_ok=True)

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(f".{img_type}"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)

                    image_path.append(file_path)

    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)
                    label_path.append(file_path)

    image_path = sorted(image_path)
    label_path = sorted(label_path)

    image_label_path_pair = list(zip(image_path, label_path))
    epsilon = 1e-6
    for im_p, lb_p in tqdm(image_label_path_pair):
        img = Image.open(im_p)
        img_array = np.array(img)  # convert to np array
        # Note that some pictures doesn't have channel
        try:
            hight, width, channel = img_array.shape
        except:
            hight, width = img_array.shape
        mask_img = np.zeros(img_array.shape, dtype=np.uint8)

        with open(lb_p) as f:
            lines = f.read().splitlines()
        for l in lines:
            line_split = l.split(' ')
            label = int(line_split[0])
            x, y, w, h = tuple([float(i) + epsilon for i in line_split[1:5]])
            # change crop ratio
            w = (1 + expand_ratio_width) * w
            h = (1 + expand_ratio_hight) * h

            # take max or min to make sure the crop is within the image
            top_left_x = max(math.floor(x * width - w * width / 2), 0)
            top_left_y = max(math.floor(y * hight - h * hight / 2), 0)
            bot_right_x = min(math.floor(x * width + w * width / 2), width)
            bot_right_y = min(math.floor(y * hight + h * hight / 2), hight)
            mask_img[top_left_y:bot_right_y, top_left_x:bot_right_x, :] = np.uint8(255)
        img_name = os.path.splitext(im_p.split(os.sep)[-1])[0]
        save_path = os.path.join(save_dir, img_name + f'.{save_type}')
        mask_img = Image.fromarray(mask_img)
        mask_img.save(save_path)


def create_rand_mask(
    img_dir,
    label_dir,
    save_dir,
    save_txt_dir,
    num_per_img=2,
    width_minmax=(0.05, 0.06),
    hight_minmax=(0.05, 0.06),
    img_type='jpg',
    save_type='png'
):
    """
    This function will generate num_per_img masks that are not intersect with boxes
    @param img_dir:
    @param label_dir:
    @param save_dir:
    @param num_per_img:
    @param width_minmax: this gives the random range of with of the mask, in terms of the percentage of the image width
    @param hight_minmax: this gives the random range of hight of the mask, in terms of the percentage of the image hight
    @param img_type:
    @param save_type:
    @return:
    """
    image_path = []
    label_path = []
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_txt_dir, exist_ok=True)

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(f".{img_type}"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)

                    image_path.append(file_path)

    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)
                    label_path.append(file_path)

    image_path = sorted(image_path)
    label_path = sorted(label_path)

    image_label_path_pair = list(zip(image_path, label_path))
    epsilon = 1e-6
    center_x_range = [width_minmax[0] / 2, 1 - width_minmax[0] / 2]
    center_y_range = [hight_minmax[0] / 2, 1 - hight_minmax[0] / 2]
    for im_p, lb_p in tqdm(image_label_path_pair):
        img = Image.open(im_p)
        img_array = np.array(img)  # convert to np array
        # Note that some pictures doesn't have channel
        try:
            hight, width, channel = img_array.shape
        except:
            hight, width = img_array.shape
        mask_img = np.zeros(img_array.shape, dtype=np.uint8)
        with open(lb_p) as f:
            lines = f.read().splitlines()

        result = []
        for _ in range(num_per_img):
            while True:
                x_center = random.uniform(center_x_range[0], center_x_range[1])
                y_center = random.uniform(center_y_range[0], center_y_range[1])
                width_ran = random.uniform(width_minmax[0], width_minmax[1])
                hight_ran = random.uniform(hight_minmax[0], hight_minmax[1])

                top_left_x_ran = x_center - width_ran / 2
                bot_right_x_ran = x_center + width_ran / 2
                top_left_y_ran = y_center - hight_ran / 2
                bot_right_y_ran = y_center + hight_ran / 2

                all_intersection = 0.0

                for l in lines:
                    line_split = l.split(' ')
                    x, y, w, h = tuple([float(i) + epsilon for i in line_split[1:5]])

                    # take max or min to make sure the crop is within the image
                    top_left_x = x - w / 2
                    top_left_y = y - h / 2
                    bot_right_x = x + w / 2
                    bot_right_y = y + h / 2

                    x1 = torch.max(torch.tensor(top_left_x_ran), torch.tensor(top_left_x))
                    y1 = torch.max(torch.tensor(top_left_y_ran), torch.tensor(top_left_y))
                    x2 = torch.min(torch.tensor(bot_right_x_ran), torch.tensor(bot_right_x))
                    y2 = torch.min(torch.tensor(bot_right_y_ran), torch.tensor(bot_right_y))

                    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
                    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
                    intersection = intersection.item()
                    all_intersection += intersection
                if all_intersection == 0.0:
                    lines.append(f'label {x_center} {y_center} {width_ran} {hight_ran}')
                    top_left_x = max(math.floor(x_center * width - width_ran * width / 2), 0)
                    top_left_y = max(math.floor(y_center * hight - hight_ran * hight / 2), 0)
                    bot_right_x = min(math.floor(x_center * width + width_ran * width / 2), width)
                    bot_right_y = min(math.floor(y_center * hight + hight_ran * hight / 2), hight)
                    mask_img[top_left_y:bot_right_y, top_left_x:bot_right_x, :] = np.uint8(255)
                    result.append(
                        "%d %.6f %.6f %.6f %.6f" % (0, x_center, y_center, width_ran, hight_ran)
                    )
                    break

        img_name = os.path.splitext(im_p.split(os.sep)[-1])[0]
        result = "\n".join(result)
        output_path = os.path.join(save_txt_dir, img_name + '.txt')
        with open(output_path, "w+") as f:
            f.write(result)

        save_path = os.path.join(save_dir, img_name + f'.{save_type}')
        mask_img = Image.fromarray(mask_img)
        mask_img.save(save_path)


def pick_make_bg_img(
    img_dir, label_dir, out_dir, target_label, max_width=0.1, max_height=0.1, max_obj_num=10
):
    """
    This function will select wanted images as background image candidates
    @param img_dir:
    @param label_dir:
    @param out_dir:
    @param target_label: the image must have this label at least once

    # note next 3 paramter exist since our Inpainting model is not perfect, we need to apply some restrictions on
    # candidates
    @param max_width: max width can exist in this image for objects, between 0 and 1
    @param max_height: max height can exist in this image for objects, between 0 and 1
    @param max_obj_num: max number of object can exit in this image
    @return:
    """
    image_path = []
    label_path = []

    if os.path.exists(out_dir):
        pass
    else:
        os.makedirs(out_dir)

    if os.path.exists(os.path.join(out_dir, 'images')):
        pass
    else:
        os.makedirs(os.path.join(out_dir, 'images'))

    if os.path.exists(os.path.join(out_dir, 'labels')):
        pass
    else:
        os.makedirs(os.path.join(out_dir, 'labels'))

    if os.path.exists(os.path.join(out_dir, 'empty_labels')):
        pass
    else:
        os.makedirs(os.path.join(out_dir, 'empty_labels'))

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)
                    image_path.append(file_path)

    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".txt"):
                if 'checkpoint' in file:
                    pass
                else:
                    file_path = os.path.join(root, file)
                    label_path.append(file_path)

    image_path = sorted(image_path)
    label_path = sorted(label_path)
    image_label_path_pair = list(zip(image_path, label_path))

    epsilon = 1e-6
    for im_p, lb_p in tqdm(image_label_path_pair):
        im_name = os.path.splitext(im_p.split(os.sep)[-1])[0]
        with open(lb_p) as f:
            lines = f.read().splitlines()
        all_targets = set()
        w_max = 0
        h_max = 0
        obj_count = 0
        for l in lines:
            line_split = l.split(' ')
            label = int(line_split[0])
            all_targets.add(label)
            x, y, w, h = tuple([float(i) + epsilon for i in line_split[1:5]])
            if w > w_max:
                w_max = w
            if h > h_max:
                h_max = h
            obj_count += 1

        if target_label in all_targets and w_max <= max_width and h_max <= max_height and obj_count <= max_obj_num:
            img_out_path = os.path.join(out_dir, 'images', im_name + '_bg.jpg')
            label_out_path = os.path.join(out_dir, 'labels', im_name + '_bg.txt')
            empty_label_out_path = os.path.join(out_dir, 'empty_labels', im_name + '_bg.txt')
            shutil.copy2(lb_p, label_out_path)
            shutil.copy2(im_p, img_out_path)
            with open(empty_label_out_path, mode='a'):
                pass
