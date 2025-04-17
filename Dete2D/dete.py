import numpy as np
import os
from tqdm import tqdm
import cv2
import copy
import pickle
from lib.hrnet.gen_kpts import gen_kpts_dete, gen_kpts_dete_gt_box_only
from lib.preprocess import h36m_coco_format_simple

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 255, 0)
    rcolor = (255, 255, 0)
    thickness = 1

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(255, 0, 0), radius=1)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(255, 0, 0), radius=1)

    return img


def get_pose2d_worldpose_frame(img_path, output_dir):

    img = cv2.imread(img_path)
    keypoints, scores = gen_kpts_dete(img, det_dim=416, num_person=1)
    keypoints, scores = h36m_coco_format_simple(keypoints, scores)
    keypoints = np.concatenate((keypoints, scores), axis=-1)

    image = show2Dpose(keypoints, copy.deepcopy(img))

    file_name = img_path.split('/')[-1].split('.')[0]
    cv2.imwrite(output_dir + file_name + "_test_2D.png", image)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_data = {}
    output_data["joints_pixel_dete"] = keypoints
    with open(output_dir + file_name + "_test_2D.pkl", "wb") as f:
        pickle.dump(output_data, f)

    # output_npz = output_dir + file_name + '_keypoints.npz'
    # np.savez_compressed(output_npz, reconstruction=keypoints)

def get_pose2d_worldpose_crop(img_crop, person_index, joints_pixel_shift):
    # img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
    img = img_crop[:, :, ::-1]
    keypoints, scores = gen_kpts_dete(img, person_index, joints_pixel_shift, det_dim=416, num_person=1)
    keypoints, scores = h36m_coco_format_simple(keypoints, scores)

    flag = True
    for i in range(keypoints.shape[0]):
        if keypoints[i][0] < 0 or keypoints[i][1] < 0 or keypoints[i][0]> img.shape[1] or keypoints[i][1] > img.shape[0]:
            keypoints[i] = joints_pixel_shift[i]
            print(f"person {person_index} joint {i} out of image, replace with gt")
            flag = False

    keypoints = np.concatenate((keypoints, scores), axis=-1)

    return keypoints, flag

def get_pose2d_worldpose_crop_gt_box_only(img_crop, person_index, joints_pixel_shift):
    img = img_crop[:, :, ::-1]
    keypoints, scores = gen_kpts_dete_gt_box_only(img, joints_pixel_shift, num_person=1)
    keypoints, scores = h36m_coco_format_simple(keypoints, scores)

    flag = True
    for i in range(keypoints.shape[0]):
        if keypoints[i][0] < 0 or keypoints[i][1] < 0 or keypoints[i][0]> img.shape[1] or keypoints[i][1] > img.shape[0]:
            keypoints[i] = joints_pixel_shift[i]
            print(f"person {person_index} joint {i} out of image in gt box detection, replace with gt")
            flag = False

    keypoints = np.concatenate((keypoints, scores), axis=-1)

    return keypoints, flag



def main() -> None:
    test_img_dir_path = r"/home/z_yin/proj/smplx/examples/ARG_CRO_220001/frame_330/"
    test_output_dir = r"output/"
    os.makedirs(test_output_dir, exist_ok=True)

    dir_file_list = os.listdir(test_img_dir_path)
    for file_name in tqdm(dir_file_list):
        file_path = os.path.join(test_img_dir_path, file_name)
        get_pose2d_worldpose_frame(file_path, test_output_dir)


if __name__ == '__main__':
    main()