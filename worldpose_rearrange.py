import os
import pickle
import numpy as np
from tqdm import tqdm


def joint_file_rearrange(joint_dir, save_dir) -> None:
    file_list = os.listdir(joint_dir)
    file_list.sort()
    for file in file_list:
        with open(os.path.join(joint_dir, file), "rb") as f:
            data = pickle.load(f)
        joint_file_res = {"game_name": data["game_name"], "player": {}}
        for frame_idx in range(data["frame_length"]):
            frame_res = data["joints"][frame_idx]
            for player_idx in range(len(frame_res["person_id"])):
                if not frame_res["is_valid"][player_idx]:
                    continue
                player_id = int(frame_res["person_id"][player_idx])
                player_res = {"player_id": player_id, "joints_world": frame_res["joints_world"][player_idx],
                              "joints_cam": frame_res["joints_cam"][player_idx],
                              "joints_pixel": frame_res["joints_pixel"][player_idx],
                              "scale_factor": frame_res["scale_factor"][player_idx],
                              "joints_pixel_concatenated": frame_res["joints_pixel_concatenated"][player_idx],
                              "joints_pixel_scaled": frame_res["joints_pixel_scaled"][player_idx],
                              "joints_pixel_detected": frame_res["joints_pixel_detected"][player_idx],
                              "frame_idx": frame_idx}
                if player_id in joint_file_res["player"]:
                    joint_file_res["player"][player_id].append(player_res)
                else:
                    joint_file_res["player"][player_id] = [player_res]

        for player in joint_file_res["player"]:
            start_frame = joint_file_res["player"][player][0]["frame_idx"]
            pre_frame = joint_file_res["player"][player][0]["frame_idx"]
            for frame_number in range(len(joint_file_res["player"][player])):
                current_frame_idx = joint_file_res["player"][player][frame_number]["frame_idx"]
                if current_frame_idx == pre_frame + 1:
                    pre_frame = current_frame_idx
                else:
                    start_frame = current_frame_idx
                    pre_frame = current_frame_idx

                joint_file_res["player"][player][frame_number]["start_frame"] = start_frame


        with open(os.path.join(save_dir, file.split('.')[0] + "_player.pkl"), "wb") as fo:
            pickle.dump(joint_file_res, fo)


test_game_list = [
'ARG_CRO_221657',
'ARG_CRO_222132',
'ARG_CRO_223805',
'ARG_CRO_235121',
'ARG_FRA_180702',
'ARG_FRA_183429',
'ARG_FRA_193158',
'BRA_KOR_221807',
'BRA_KOR_232126',
'BRA_KOR_224459',
'CRO_MOR_181141',
'CRO_MOR_182854',
'CRO_MOR_184559',
'CRO_MOR_194948',
'ENG_FRA_224248',
'ENG_FRA_233519',
'ENG_FRA_232015',
'ENG_FRA_235059',
'FRA_MOR_232451',
'FRA_MOR_222923',
'MOR_POR_181952',
'MOR_POR_192030',
'MOR_POR_184724',
'NET_ARG_220032',
'NET_ARG_231259',
'NET_ARG_222749',
]


def train_test_split(joint_dir, game_result_player_dir) -> None:
    game_result_file_dir = joint_dir
    game_result_file_list = os.listdir(game_result_file_dir)
    game_name_list_train = []
    for game_file in game_result_file_list:
        game_name = game_file.split('.')[0]
        if game_name not in test_game_list:
            game_name_list_train.append(game_name)
    game_name_list_test = test_game_list

    train_dict = source_file_create(game_name_list_train, "train", game_result_player_dir)
    test_dict = source_file_create(game_name_list_test, "test", game_result_player_dir)
    pose_data_source = {}
    pose_data_source["train"] = train_dict
    pose_data_source["test"] = test_dict

    game_name_list = []
    for idx, source_name in enumerate(pose_data_source["test"]["source"]):
        game_name = source_name.split("+")[0]
        game_name_list.append(game_name)

    pose_data_source["test"]["action"] = game_name_list


    with open("./wp_hr_conf_cam_source.pkl", "wb") as fo:
        pickle.dump(pose_data_source, fo)



def source_file_create(game_name_list, tag, game_result_player_dir):
    game_result_player_file_list = os.listdir(game_result_player_dir)
    pose_data = {}

    joint3d_image = []
    joint_2d = []
    joints_25d_image = []
    factor = []
    source = []
    confidence = []

    for game_result_player_file in game_result_player_file_list:
        with open(os.path.join(game_result_player_dir, game_result_player_file), "rb") as f:
            data = pickle.load(f)
        if data["game_name"] in game_name_list:
            player_index_list = []
            for player in data["player"]:
                player_index_list.append(int(player))
            player_index_list.sort()
            for player_idx in player_index_list:
                player_res_list = data["player"][player_idx]
                for player_res in player_res_list:
                    current_frame_player_description = data["game_name"] + "+" + f"player{player_idx}" + "+" + f"start_frame{player_res['start_frame']}"
                    confidence.append(player_res["joints_pixel_detected"][..., 2])
                    joint_2d.append(player_res["joints_pixel_detected"][..., :2])
                    joint3d_image.append(player_res["joints_pixel_concatenated"])
                    joints_25d_image.append(player_res["joints_pixel_scaled"])
                    factor.append(player_res["scale_factor"])
                    source.append(current_frame_player_description)

    if tag == "test":
        pose_data["confidence"] = np.array(confidence)
        pose_data["joint_2d"] = np.array(joint_2d)
        pose_data["joint3d_image"] = np.array(joint3d_image)
        pose_data["2.5d_factor"] = np.array(factor)
        pose_data["joints_2.5d_image"] = np.array(joints_25d_image)
        pose_data["source"] = source
    elif tag == "train":
        pose_data["confidence"] = np.array(confidence)
        pose_data["joint_2d"] = np.array(joint_2d)
        pose_data["joint3d_image"] = np.array(joint3d_image)
        pose_data["source"] = source

    return pose_data





def main() -> None:
    joint_dir = r"./worldpose_preprocess_output/"
    save_dir = r"./game_result_player/"

    joint_file_rearrange(joint_dir, save_dir)

    train_test_split(joint_dir, save_dir)


if __name__ == '__main__':
    main()
