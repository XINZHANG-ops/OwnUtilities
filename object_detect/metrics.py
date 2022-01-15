import os
from collections import Counter
import nltk
from tqdm import tqdm
import torch


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


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    all_classes = list(set([l[1] for l in true_boxes]))

    label_wise_precision = nltk.defaultdict(float)
    label_wise_recall = nltk.defaultdict(float)
    label_wise_ap = nltk.defaultdict(float)

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
            # training idx as detection
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
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        if precisions.shape[0] == 1:  # this means we never predict for this label
            label_wise_precision[c] = 0
        else:
            label_wise_precision[c] = precisions.numpy()[-1]
        label_wise_recall[c] = recalls.numpy()[-1]
        label_ap = float(torch.trapz(precisions, recalls))
        # average_precisions.append(label_ap)
        label_wise_ap[c] = label_ap

    return dict(label_wise_precision), dict(label_wise_recall), dict(label_wise_ap)


class label_metrics:
    def __init__(self, pred_txt_dir, truth_txt_dir):
        pred_boxes = []
        true_boxes = []
        """
        expect pred txt sequence:
        each txt contains rows like this:

        8 0.529688 0.829167 0.078125 0.0444444 0.893442
        8 0.723828 0.816667 0.103906 0.0694444 0.905974
        8 0.95 0.778472 0.0921875 0.0708333 0.911328
        0 0.103125 0.111806 0.110937 0.118056 0.935599

        with 

        int_class x y w h conf
        """

        for root, dirs, files in os.walk(pred_txt_dir):
            for file in files:
                if file.endswith(".txt"):
                    the_path = os.path.join(root, file)
                    if 'checkpoint' not in the_path:
                        file_name = the_path.split(os.sep)[-1].split('.')[0]
                        with open(the_path) as f:
                            lines = f.read().splitlines()
                        for l in lines:
                            line_split = l.split(' ')
                            line_split.insert(0, file_name)
                            conf = line_split[-1]
                            line_split.insert(2, conf)
                            line_split = line_split[:-1]
                            line_split[1] = int(line_split[1])
                            line_split[2] = float(line_split[2])
                            line_split[3] = float(line_split[3])
                            line_split[4] = float(line_split[4])
                            line_split[5] = float(line_split[5])
                            line_split[6] = float(line_split[6])
                            pred_boxes.append(line_split)
        """
        expect true txt sequence:
        each txt contains rows like this:

        0 0.104297 0.107639 0.110156 0.120833
        8 0.724219 0.818056 0.098437 0.069444
        8 0.949219 0.784028 0.089063 0.081944
        8 0.530078 0.831250 0.075781 0.048611

        with 

        int_class x y w h
        """

        for root, dirs, files in os.walk(truth_txt_dir):
            for file in files:
                if file.endswith(".txt"):
                    the_path = os.path.join(root, file)
                    if 'checkpoint' not in the_path:
                        file_name = the_path.split(os.sep)[-1].split('.')[0]
                        with open(the_path) as f:
                            lines = f.read().splitlines()
                        for l in lines:
                            line_split = l.split(' ')
                            line_split.insert(0, file_name)
                            line_split.insert(2, 1)
                            line_split[1] = int(line_split[1])
                            line_split[2] = float(line_split[2])
                            line_split[3] = float(line_split[3])
                            line_split[4] = float(line_split[4])
                            line_split[5] = float(line_split[5])
                            line_split[6] = float(line_split[6])
                            true_boxes.append(line_split)

        self.pred_boxes = pred_boxes
        self.true_boxes = true_boxes

    def get_metrics(self, iou_threshold=0.5):
        label_wise_precision, label_wise_recall, label_wise_ap = mean_average_precision(
            self.pred_boxes, self.true_boxes, iou_threshold=iou_threshold, box_format="midpoint"
        )
        return {
            'precision': label_wise_precision,
            'recall': label_wise_recall,
            'average precision': label_wise_ap
        }

    def find_confuse_labels(
            self, iou_threshold=0.1, pred_boxes=None, true_boxes=None, box_format='midpoint'
    ):
        """
        This will give an idea of how model confused between each labels

        @param pred_boxes: result from label_map class, list of boxes, if None, use self.pred_boxes
        @param true_boxes: result from label_map class, list of boxes, if None, use self.true_boxes
        @param iou_threshold: do suggest put iou_threshold low since this is not a calculation for metrics
        @param box_format:
        @return: where precision_result means how each prediction is mapped to true labels, keys are predicted labels
                 values are count of that predict box's true label

                 where recall_result means how each ground truth is mapped to prediction labels, keys are true labels
                 values are count of that true box is predicted to
        """
        """
        the iou_threshold still needed think of the case that, there is a true box in the image, but the true box doesn't 
        overlap any prediction box, which mean it is a missed object, so we want to capture this case. If we don't have an
        iou_threshold, simply sorted by ious, then the true box actually doesn't match the sorted top predict box

        """
        if pred_boxes is None:
            pred_boxes = self.pred_boxes
        if true_boxes is None:
            true_boxes = self.true_boxes

        label_wise_confuse_recall = nltk.defaultdict(lambda: nltk.defaultdict(int))
        label_wise_confuse_precision = nltk.defaultdict(lambda: nltk.defaultdict(int))

        all_img = list(set([i[0] for i in true_boxes]))

        for img in tqdm(all_img):
            img_pred = [i for i in pred_boxes if i[0] == img]
            img_gt = [i for i in true_boxes if i[0] == img]
            for gt in img_gt:
                img_ious = []
                for det in img_pred:
                    iou = intersection_over_union(
                        torch.tensor(det[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )
                    if iou >= iou_threshold:
                        img_ious.append([iou, det])
                if img_ious:
                    match_pred_box = sorted(img_ious, key=lambda x: x[0], reverse=True)[0][1]
                    det_cls = match_pred_box[1]
                else:
                    det_cls = 'background'

                true_cls = gt[1]
                label_wise_confuse_recall[true_cls][det_cls] += 1

            for det in img_pred:
                img_ious = []
                for gt in img_gt:
                    iou = intersection_over_union(
                        torch.tensor(det[3:]),
                        torch.tensor(gt[3:]),
                        box_format=box_format,
                    )
                    if iou >= iou_threshold:
                        img_ious.append([iou, gt])
                if img_ious:
                    match_true_box = sorted(img_ious, key=lambda x: x[0], reverse=True)[0][1]
                    true_cls = match_true_box[1]
                else:
                    true_cls = 'background'

                det_cls = det[1]
                label_wise_confuse_precision[det_cls][true_cls] += 1

        recall_result = dict()
        for label, confuse_dict in label_wise_confuse_recall.items():
            recall_result[label] = [
                (k, v)
                for k, v in sorted(confuse_dict.items(), key=lambda item: item[1], reverse=True)
            ]

        precision_result = dict()
        for label, confuse_dict in label_wise_confuse_precision.items():
            precision_result[label] = [
                (k, v)
                for k, v in sorted(confuse_dict.items(), key=lambda item: item[1], reverse=True)
            ]

        return {'precision': precision_result, 'recall': recall_result}


def demo():
    pred_txt_dir = 'yolo_cls/runs/detect/exp/labels'
    truth_txt_dir = 'linmao_camera_data_real/labels/test'
    lm = label_metrics(pred_txt_dir, truth_txt_dir)
    lm.get_metrics(iou_threshold=0.5)
    lm.find_confuse_labels(iou_threshold=0.1, box_format='midpoint')
