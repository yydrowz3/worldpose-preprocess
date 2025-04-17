import torch
import numpy as np
import sys
import cv2
import pickle
import os
import decord
import argparse
sys.path.append("./Dete2D/")
import smplx
import matplotlib.pyplot as plt
from Dete2D.dete import get_pose2d_worldpose_crop, get_pose2d_worldpose_crop_gt_box_only
from utilities import convert_smpl24_to_hm36, convert3dworld_to_camera, proj_3dworld_to_img, convert_camera_meter_to_millimeter, factor_calculate, crop_person_img

raw_video_dir = r'your path to /FIFA Skeletal Light - Camera 1 Footage/WorldPoseDataset/raw/'
pose_anno_dir = r'your path to /WorldPose/poses/'
cam_anno_dir = r'your path to /WorldPose/cameras/'

model_folder_path = r"./models/"
save_dir = r"./worldpose_preprocess_output/"

gt_invalid_cnt = 0
dete_fail_cnt = 0
fixed_res_cnt = 0


def pkl_all_joints_frame(smpl_model_folder_path: str, raw_image: np.ndarray, frame_idx: int,
                         pose_global_orient: np.ndarray, pose_body_pose: np.ndarray, pose_transl: np.ndarray, pose_betas: np.ndarray,
                         cam_k: np.ndarray, cam_K: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray):

    nan_person_idx = []
    person_id = []

    joints_world_list = []
    joints_cam_list = []
    joints_pixel_list = []
    scale_factor_list = []
    joints_pixel_concatenated_list = []
    joints_pixel_scaled_list = []
    joints_pixel_detected_list = []

    valid_flag = []
    detection_valid_flag = []
    dete_not_outofboundary_flag = []

    current_frame_res = {}

    for i in range(pose_betas.shape[0]):
        if np.isnan(pose_global_orient[i]).any() or np.isnan(pose_body_pose[i]).any() or np.isnan(pose_transl[i]).any():
            print(f"in frame {frame_idx}: person {i} was unlabelled ")
            nan_person_idx.append(i)
            continue
        smpl_model = smplx.create(model_path=smpl_model_folder_path, model_type='smpl',
                                  gender='neutral', use_face_contour=False, num_betas=10, num_expression_coeffs=10,
                                  ext='npz', use_hands=False, use_feet_keypoints=False)

        current_pose_beta = torch.from_numpy(pose_betas[i:i+1, :])
        current_pose_global_orient = torch.from_numpy(pose_global_orient[i:i+1, :])
        current_pose_body_pose = torch.from_numpy(pose_body_pose[i:i+1, :])
        current_pose_transl = torch.from_numpy(pose_transl[i:i+1, :])

        smpl_output = smpl_model(body_pose=current_pose_body_pose, global_orient=current_pose_global_orient,
                                 transl=current_pose_transl, betas=current_pose_beta,
                                 return_verts=True)

        joints = smpl_output.joints.detach().cpu().numpy().squeeze() # (45, 3) / (24, 3)
        joints = convert_smpl24_to_hm36(joints)
        joints_world_list.append(joints)
        joints_cam = convert3dworld_to_camera(joints, cam_R, cam_t)
        joints_cam_list.append(joints_cam)
        joints_pixel = proj_3dworld_to_img(joints, cam_R, cam_t, cam_K, cam_k) # (17, 2)
        joints_pixel_list.append(joints_pixel)
        scale_factor, joints_pixel_con, joints_pixel_scaled = factor_calculate(joints_cam, joints_pixel)
        scale_factor_list.append(scale_factor)
        joints_pixel_concatenated_list.append(joints_pixel_con)
        joints_pixel_scaled_list.append(joints_pixel_scaled)
        person_id.append(i)

        flag = True
        for j in range(joints_pixel.shape[0]):
            if joints_pixel[j][0] < 0 or joints_pixel[j][0] >= raw_image.shape[1] or joints_pixel[j][1] < 0 or joints_pixel[j][1] >= raw_image.shape[0]:
                flag = False
                break

        if flag == False:
            print(f"person {i} has joints out of image boundary, skip detection, using gt as detection and you should drop this person")
            joints_pixel_dete = joints_pixel.copy()
            joints_pixel_dete = np.concatenate([joints_pixel_dete, np.ones([joints_pixel_dete.shape[0], 1])], axis=-1)
            detection_valid_flag.append("False")
            dete_not_outofboundary_flag.append("False")
            global gt_invalid_cnt
            gt_invalid_cnt += 1
        else:
            cropped_person_image, start_x, start_y = crop_person_img(raw_image, joints_pixel)
            joints_pixel_shift = joints_pixel.copy()
            joints_pixel_shift[:, 0] -= start_y
            joints_pixel_shift[:, 1] -= start_x

            joints_pixel_dete_yolo_box, dete_res_flag_yolo_box = get_pose2d_worldpose_crop(cropped_person_image, i, joints_pixel_shift)
            joints_pixel_dete_yolo_box[:, 0] += start_y
            joints_pixel_dete_yolo_box[:, 1] += start_x

            joints_pixel_dete_gt_box, dete_res_flag_gt_box = get_pose2d_worldpose_crop_gt_box_only(cropped_person_image, i, joints_pixel_shift)
            joints_pixel_dete_gt_box[:, 0] += start_y
            joints_pixel_dete_gt_box[:, 1] += start_x

            difference_dis_yolo = np.linalg.norm(joints_pixel - joints_pixel_dete_yolo_box[:, :2])
            difference_dis_gt = np.linalg.norm(joints_pixel - joints_pixel_dete_gt_box[:, :2])
            if difference_dis_yolo < difference_dis_gt:
                dete_tag = "yolo"
                joints_pixel_dete = joints_pixel_dete_yolo_box
                dete_res_flag = dete_res_flag_yolo_box
                difference_dis = difference_dis_yolo
            else:
                dete_tag = "gt"
                joints_pixel_dete = joints_pixel_dete_gt_box
                dete_res_flag = dete_res_flag_gt_box
                difference_dis = difference_dis_gt
            dete_not_outofboundary_flag.append(dete_res_flag)
            if not dete_res_flag:
                global fixed_res_cnt
                fixed_res_cnt += 1
            if difference_dis > 38:
                print(f"In person {i}, using {dete_tag} results, difference distance {difference_dis} too large, set as invalid")
                detection_valid_flag.append(False)
                global dete_fail_cnt
                dete_fail_cnt += 1
            else:
                detection_valid_flag.append(True)

        valid_flag.append(flag)
        joints_pixel_detected_list.append(joints_pixel_dete)


    current_frame_res["joints_world"] = np.array(joints_world_list)
    current_frame_res["joints_cam"] = np.array(joints_cam_list)
    current_frame_res["joints_pixel"] = np.array(joints_pixel_list)
    current_frame_res["scale_factor"] = np.array(scale_factor_list)
    current_frame_res["joints_pixel_concatenated"] = np.array(joints_pixel_concatenated_list)
    current_frame_res["joints_pixel_scaled"] = np.array(joints_pixel_scaled_list)
    current_frame_res["person_id"] = np.array(person_id)
    current_frame_res["nan_person_id"] = np.array(nan_person_idx)
    current_frame_res["joints_pixel_detected"] = np.array(joints_pixel_detected_list)
    current_frame_res["frame_idx"] = frame_idx
    current_frame_res["is_valid"] = valid_flag
    current_frame_res["is_detection_valid"] = detection_valid_flag
    current_frame_res["is_dete_not_outofboundary"] = dete_not_outofboundary_flag

    return current_frame_res

def pkl_joint_file(game_name: str, pose_file_path: str, cam_file_path: str, ori_video_path: str, save_dir: str) -> None:
    vr = decord.VideoReader(ori_video_path)
    frame_len = len(vr)
    pose_data = np.load(pose_file_path, allow_pickle=True)
    cam_data = np.load(cam_file_path, allow_pickle=True)
    pose_global_orient = pose_data["global_orient"]
    pose_body_pose = pose_data["body_pose"]
    pose_transl = pose_data["transl"]
    pose_betas = pose_data["betas"]
    if frame_len != pose_global_orient.shape[1] or frame_len != pose_body_pose.shape[1] or frame_len != pose_transl.shape[1]:
        raise Exception("The length of video and pose data is not equal")
    cam_k = cam_data["k"]
    cam_K = cam_data["K"]
    cam_R = cam_data["R"]
    cam_t = cam_data["t"]
    if frame_len != cam_k.shape[0] or frame_len != cam_K.shape[0] or frame_len != cam_R.shape[0] or frame_len != cam_t.shape[0]:
        raise Exception("The length of video and camera data is not equal")

    game_res = {"game_name": game_name, "frame_length": frame_len, "joints": []}

    print(f"Now processing ({game_name})... (total frame: {frame_len})")
    for frame_idx in range(frame_len):
        print(f"Processing frame {frame_idx} ...")
        raw_image = vr[frame_idx].asnumpy()
        frame_res = pkl_all_joints_frame(smpl_model_folder_path=model_folder_path, raw_image=raw_image, frame_idx=frame_idx,
                                         pose_global_orient=pose_global_orient[:, frame_idx, :],
                                         pose_body_pose=pose_body_pose[:, frame_idx, :],
                                         pose_transl=pose_transl[:, frame_idx, :], pose_betas=pose_betas,
                                         cam_k=cam_k[frame_idx], cam_K=cam_K[frame_idx], cam_R=cam_R[frame_idx], cam_t=cam_t[frame_idx])
        game_res["joints"].append(frame_res)

    with open(os.path.join(save_dir, f"{game_name}.pkl"), "wb") as f:
        pickle.dump(game_res, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="for worldpose data preprocess (game based)")
    parser.add_argument('--game-name', required=True, type=str,  help='The name of the game.')
    parser.add_argument('--save-dir', required=False, default="./worldpose_preprocess_output/", type=str,  help='The path to camera parameters.')
    args = parser.parse_args()
    game_name = args.game_name

    pose_file = os.path.join(pose_anno_dir, f"{game_name}.npz")
    cam_file = os.path.join(cam_anno_dir, f"{game_name}.npz")
    raw_video = os.path.join(raw_video_dir, f"{game_name}.mov")


    os.makedirs(save_dir, exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    pkl_joint_file(game_name=game_name, pose_file_path=pose_file, cam_file_path=cam_file, ori_video_path=raw_video, save_dir=save_dir)

    print(f"GT invalid count: {gt_invalid_cnt}")
    print(f"Detection failed count: {dete_fail_cnt}")
    print(f"fixed result count: {fixed_res_cnt}")

if __name__ == '__main__':
    main()




















