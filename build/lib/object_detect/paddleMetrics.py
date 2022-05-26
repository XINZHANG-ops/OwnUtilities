from collections import Counter
import nltk
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import declxml as xml
import json

# requirement
"""
declxml==0.9.1
"""
"""
xml and txt are two formats of object detection labels, they are same labels just in different format

different models take different format as input

"""


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


class Transformer(object):
    def __init__(self, xml_dir):
        self.xml_dir = xml_dir
        self.label_map = dict()
        self.all_classes = list()
        self.label_map_exist = False

    def transform(self):
        reader = Reader(xml_dir=self.xml_dir)
        xml_files = reader.get_xml_files()

        object_mapper = ObjectMapper()
        annotations = object_mapper.bind_files(xml_files, xml_dir=self.xml_dir)
        if len(annotations) > 0:
            return self.get_boxes(annotations)

    def get_boxes(self, annotations):
        # each annotation is all boxes in an image
        all_boxes = []
        for annotation in annotations:
            file_name = self.darknet_filename_format(annotation.filename)
            all_boxes.extend(self.to_darknet_format(annotation, file_name))
        return all_boxes

    def to_darknet_format(self, annotation, image_name):
        result = []
        for obj in annotation.objects:
            label_name = obj.name
            x, y, w, h = self.get_object_params(obj)
            box = [image_name, label_name, 1, x, y, w, h]
            result.append(box)
        return result

    @staticmethod
    def get_object_params(obj):
        box = obj.box
        x = box.xmin + 0.5 * (box.xmax - box.xmin)
        y = box.ymin + 0.5 * (box.ymax - box.ymin)

        width = box.xmax - box.xmin
        height = box.ymax - box.ymin

        return int(x), int(y), int(width), int(height)

    @staticmethod
    def darknet_filename_format(filename):
        pre, ext = os.path.splitext(filename)
        return pre


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    boxes_preds = torch.tensor([9.26342e+02, 3.73585e+02, 9.73841e+02, 4.08305e+02])
    boxes_labels = torch.tensor([9.26122e+02, 3.77460e+02, 9.71399e+02, 4.06312e+02])

    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":

        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", interpolate_pr=True
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        interpolate_pr: if to use interpolated pr curve, only effect plot not score
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    all_classes = list(set([l[1] for l in true_boxes]))

    label_wise_precision = nltk.defaultdict(float)
    label_wise_recall = nltk.defaultdict(float)
    label_wise_ap = nltk.defaultdict(float)
    label_wise_pr_curve = dict()

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in tqdm(all_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection (same image)
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))

        if interpolate_pr:
            precisions = precisions.tolist()
            for ipidx, _ in enumerate(precisions):
                precisions[ipidx] = max(precisions[ipidx:])
        precisions = torch.tensor(precisions)

        recalls = torch.cat((torch.tensor([0]), recalls))
        label_wise_pr_curve[c] = {
            'precisions': precisions.tolist() + [0],
            'recalls': recalls.tolist() + [recalls.tolist()[-1]]
        }
        # torch.trapz for numerical integration
        if precisions.shape[0] == 1:  # this means we never predict for this label
            label_wise_precision[c] = 0
        else:
            label_wise_precision[c] = precisions.numpy()[-1]
        label_wise_recall[c] = recalls.numpy()[-1]
        # Note that trapz naturally calculated interpolated curve area
        label_ap = float(torch.trapz(precisions, recalls))
        # average_precisions.append(label_ap)
        label_wise_ap[c] = label_ap

    return dict(label_wise_precision), dict(label_wise_recall
                                            ), dict(label_wise_ap), label_wise_pr_curve


class label_metrics:
    def __init__(self, pred_json_path, truth_xml_dir, conf_threshold=0.25):
        xml_transformer = Transformer(truth_xml_dir)
        true_boxes = xml_transformer.transform()

        with open(pred_json_path) as json_file:
            predictions = json.load(json_file)

        pred_boxes = []
        file_names = [i.split('.')[0] for i in predictions['names']]
        for idx, img_pred in enumerate(predictions['results']):
            name = file_names[idx]
            for box_pred in img_pred:
                bbox = box_pred['bbox']
                conf = box_pred['score']
                label_name = box_pred['category']
                x = bbox[0]
                y = bbox[1]
                box_w = bbox[2]
                box_h = bbox[3]
                x1, y1, w, h = x + box_w / 2, y + box_h / 2, int(box_w), int(box_h)
                if conf >= conf_threshold:
                    pred_boxes.append([name, label_name, conf, x1, y1, w, h])

        self.pred_boxes = pred_boxes
        self.true_boxes = true_boxes

    def get_metrics(
        self, iou_thresholds=[0.5, 0.7, 0.9], interpolate_pr=True, output_dir='eval_results'
    ):
        def uniquify(path):
            filename, extension = os.path.splitext(path)
            counter = 1

            while os.path.exists(path):
                path = filename + " (" + str(counter) + ")" + extension
                counter += 1
            return path

        output_dir = uniquify(output_dir)
        os.mkdir(output_dir)

        table_result = []
        column_names = []

        label_all_precisions = nltk.defaultdict(list)
        label_all_recalls = nltk.defaultdict(list)
        label_all_f1 = nltk.defaultdict(list)
        label_all_aps = nltk.defaultdict(list)
        label_rows = nltk.defaultdict(list)

        label_names = []

        for iou_threshold in iou_thresholds:
            print(f'processing IoU@{iou_threshold}')
            column_names.extend([
                f'P@{iou_threshold}', f'R@{iou_threshold}', f'F1@{iou_threshold}',
                f'AP@{iou_threshold}'
            ])
            label_wise_precision, label_wise_recall, label_wise_ap, label_wise_pr_curve = mean_average_precision(
                self.pred_boxes,
                self.true_boxes,
                iou_threshold=iou_threshold,
                box_format="midpoint",
                interpolate_pr=interpolate_pr
            )

            # Generate PR curve
            plt.figure(figsize=(15, 15))
            for label, prs in label_wise_pr_curve.items():
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.plot(
                    prs['recalls'],
                    prs['precisions'],
                    linewidth=2,
                    label=f'{label} AP={label_wise_ap[label]:.2f}'
                )
                plt.xlabel('recall', fontsize=30)
                plt.ylabel('precision', fontsize=30)
                plt.gca().set_aspect('equal')
            plt.plot([0, 1], [0, 1], '--', linewidth=3, color='red')
            plt.legend(fontsize=25, loc='lower right')
            plt.title(f'precision-recall IoU={iou_threshold}', fontsize=30)
            plt.savefig(os.path.join(output_dir, f'precision-recall IoU={iou_threshold}.png'))
            plt.close()

            if len(label_names) > 0:
                pass
            else:
                label_names = sorted(list(label_wise_precision.keys()))

            for label in label_names:
                label_p = label_wise_precision[label]
                label_r = label_wise_recall[label]
                label_f1 = 0.5 * label_p + 0.5 * label_r
                label_ap = label_wise_ap[label]
                label_all_precisions[label].append(label_p)
                label_all_recalls[label].append(label_r)
                label_all_f1[label].append(label_f1)
                label_all_aps[label].append(label_ap)
                label_rows[label].extend([label_p, label_r, label_f1, label_ap])

        column_names.extend([f'P@All', f'R@All', f'F1@All', f'AP@All'])
        for label in label_names:
            row = label_rows[label] + [
                sum(label_all_precisions[label]) / len(label_all_precisions[label]),
                sum(label_all_recalls[label]) / len(label_all_recalls[label]),
                sum(label_all_f1[label]) / len(label_all_f1[label]),
                sum(label_all_aps[label]) / len(label_all_aps[label])
            ]

            table_result.append(row)

        last_row = np.sum(table_result, axis=0) / len(table_result)
        table_result.append(last_row)

        df = pd.DataFrame(table_result, columns=column_names)
        df.insert(0, "label", label_names + ['All'])
        df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred',
        type=str,
        default='PPYOLOv2_prediction.json',
        help='path to prediction json',
        required=True
    )

    parser.add_argument(
        '--label',
        type=str,
        default='linmao-test/Annotations',
        help='xml labels directory',
        required=True
    )

    parser.add_argument('--conf', type=float, default=0.25, help='threshold for confidence')

    parser.add_argument(
        '--ious', '--list', nargs='+', default=[0.5, 0.7, 0.9], help='thresholds for iou'
    )

    parser.add_argument('--output', default='eval_results', help='output directory for results')

    parser.add_argument(
        '--no-interpolate', action='store_false', help='if to interpolate pr curve for plots'
    )

    args = parser.parse_args()

    prediction_path = args.pred
    label_dir = args.label
    conf_thres = args.conf
    ious = [float(i) for i in args.ious]
    output_dir = args.output
    interpolate = args.no_interpolate

    lm = label_metrics(prediction_path, label_dir, conf_threshold=conf_thres)
    lm.get_metrics(iou_thresholds=ious, interpolate_pr=interpolate, output_dir=output_dir)
