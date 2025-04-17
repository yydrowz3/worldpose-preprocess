from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

# from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
# from lib.hrnet.lib.config import cfg, update_config
# from lib.hrnet.lib.utils.transforms import *
# from lib.hrnet.lib.utils.inference import get_final_preds
# from lib.hrnet.lib.models import pose_hrnet
from .lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from .lib.config import cfg, update_config
from .lib.utils.transforms import *
from .lib.utils.inference import get_final_preds
from .lib.models import pose_hrnet

cfg_dir = './Dete2D/lib/hrnet/experiments/'
model_dir = './Dete2D/lib/checkpoint/'

# Loading human detector model
from ..yolov3.human_detector import load_model as yolo_model
from ..yolov3.human_detector import yolo_human_det as yolo_det
# from lib.sort.sort import Sort


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args, unknown = parser.parse_known_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')
    
    return model


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    # people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        if not ret:
            continue

        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        if bboxs is None or not bboxs.any():
            print('No person detected!')
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs) 
            scores_pre = copy.deepcopy(scores) 

        # Using Sort to track people
        # people_track = people_sort.update(bboxs)
        people_track = bboxs[::-1]

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            # people_track_ = people_track[-1, :-1].reshape(1, 4)
            people_track_ = people_track[-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            # people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
            people_track_ = people_track[-num_peroson].reshape(num_peroson, 4)
            # people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores

def gen_kpts_dete(img, person_index, joints_pixel_shift, det_dim=416, num_person=1):
    # img = cv2.imread(img_path)
    args = parse_args()
    reset_config(args)
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)

    bbox, scores_detect = yolo_det(img, human_model, reso=det_dim, confidence=args.thred_score)

    x_min, y_min = np.min(joints_pixel_shift, axis=0)
    x_max, y_max = np.max(joints_pixel_shift, axis=0)
    x_min = max(0, x_min - 8)
    y_min = max(0, y_min - 8)
    x_max = min(img.shape[1], x_max + 6)
    y_max = min(img.shape[0], y_max + 6)

    if bbox is None or not bbox.any():
        print(f"On player_index: {person_index}, No person detected (Using gt box)")
        bbox = [[x_min, y_min, x_max, y_max]]
    elif bbox.shape[0] > 1:
        image_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
        bbox_center = np.array([bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2]).T
        bbox_distance = np.linalg.norm(bbox_center - image_center, axis=-1)
        bbox = bbox[np.argmin(bbox_distance)].reshape(1, 4)
        bbox = list(bbox)
    else:
        bbox = bbox[0].reshape(1, 4)
        bbox = list(bbox)


    with torch.no_grad():
        inputs, origin_img, center, scale = PreProcess(img, bbox, cfg, num_person)
        inputs = inputs[:, [2, 1, 0]]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = pose_model(inputs)
        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))


    return preds[0], maxvals[0]

def gen_kpts_dete_gt_box_only(img, joints_pixel_shift, num_person=1):
    pose_model = model_load(cfg)


    x_min, y_min = np.min(joints_pixel_shift, axis=0)
    x_max, y_max = np.max(joints_pixel_shift, axis=0)
    x_min = max(0, x_min - 8)
    y_min = max(0, y_min - 8)
    x_max = min(img.shape[1], x_max + 6)
    y_max = min(img.shape[0], y_max + 6)
    bbox = [[x_min, y_min, x_max, y_max]]

    with torch.no_grad():
        inputs, origin_img, center, scale = PreProcess(img, bbox, cfg, num_person)
        inputs = inputs[:, [2, 1, 0]]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = pose_model(inputs)
        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))


    return preds[0], maxvals[0]
