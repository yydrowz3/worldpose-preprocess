import numpy as np
import cv2

def convert_smpl24_to_hm36(joints: np.ndarray) -> np.ndarray:
    SMPL24_TO_HM36 = [0, 2, 5, 8, 1, 4, 7, 3, 9, 12, 15, 16, 18, 20, 17, 19, 21]
    joint_HM36 = [joints[i] for i in SMPL24_TO_HM36]
    joint_HM36 = np.array(joint_HM36)

    joint_HM36[0, :] = np.mean(joints[1:3, :], axis=0, dtype=np.float32) # pelvis
    joint_HM36[7, :] = np.mean(joints[[1, 2, 16, 17], :], axis=0, dtype=np.float32) # spine

    joint_HM36[8, :] = np.mean(joints[16:18, :], axis=0, dtype=np.float32) # thorax
    joint_HM36[8, :] += (joints[15, :] - joint_HM36[8, :]) / 4

    return joint_HM36

def convert3dworld_to_camera(joints_3d: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    extrinsic_matrix = np.zeros((4, 4))
    extrinsic_matrix[:3, :3] = cam_R
    for i in range(len(cam_t)):
        extrinsic_matrix[i, 3] = cam_t[i]
    extrinsic_matrix[3, 3] = 1

    joints_3d = np.concatenate([joints_3d, np.ones((joints_3d.shape[0], 1))], axis=1)
    joints_3d_camera = joints_3d @ extrinsic_matrix.T
    return joints_3d_camera[..., :3]


def proj_3dworld_to_img(joints_3d: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray, cam_K: np.ndarray, cam_k: np.ndarray) -> np.ndarray:
    joints = []
    for j in range(joints_3d.shape[0]):
        img_pts, _ = cv2.projectPoints(joints_3d[j], cam_R, cam_t, cam_K, cam_k)
        joints.append(img_pts[0][0])
    return np.array(joints)

def convert_camera_meter_to_millimeter(joints: np.ndarray) -> np.ndarray:
    # (17, 3)
    coordinate_camera_millimeter = np.zeros_like(joints)
    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            coordinate_camera_millimeter[i][j] = joints[i][j] * 1000
    return coordinate_camera_millimeter


def factor_calculate(joints_camera: np.ndarray, joints_pixel: np.ndarray, root_idx: int = 0, ref_idx: int = 10):
    joints_camera_millimeter = convert_camera_meter_to_millimeter(joints_camera)
    joints_pixel_concatenated_with_depth = joints_camera.copy()
    joints_pixel_scaled = joints_camera.copy()

    root_joint_millimeter = joints_camera_millimeter[root_idx]
    dists_pixel = np.linalg.norm(joints_pixel[root_idx] - joints_pixel[ref_idx])
    if dists_pixel < 0.1:
        raise Exception("The pixel distance between root and ref is too small")
    dists_camera = np.linalg.norm(joints_camera_millimeter[root_idx] - joints_camera_millimeter[ref_idx])
    if dists_camera < 0.1:
        raise Exception("The camera distance between root and ref is too small")
    scale_factor = dists_camera / dists_pixel
    for j in range(joints_camera_millimeter.shape[0]):
        relative_depth_in_camera = joints_camera_millimeter[j][2] - root_joint_millimeter[2]
        joints_pixel_concatenated_with_depth[j][0] = joints_pixel[j][0]
        joints_pixel_concatenated_with_depth[j][1] = joints_pixel[j][1]
        joints_pixel_concatenated_with_depth[j][2] = relative_depth_in_camera / scale_factor
        joints_pixel_scaled[j][0] = joints_pixel[j][0] * scale_factor
        joints_pixel_scaled[j][1] = joints_pixel[j][1] * scale_factor
        joints_pixel_scaled[j][2] = relative_depth_in_camera

    return scale_factor, joints_pixel_concatenated_with_depth, joints_pixel_scaled


def crop_person_img(ori_img: np.ndarray, joints_img: np.ndarray, padding_pixel: int = 30):
    image_height, image_width = ori_img.shape[:2]
    x_coords = [int(x) for x, y in joints_img]
    y_coords = [int(y) for x, y in joints_img]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    x_min = max(0, x_min - padding_pixel)
    x_max = min(image_width, x_max + padding_pixel)
    y_min = max(0, y_min - padding_pixel)
    y_max = min(image_height, y_max + padding_pixel)
    cropped_image = ori_img[y_min:y_max, x_min:x_max]

    return cropped_image, y_min, x_min
