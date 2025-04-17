from tqdm import tqdm
import pickle
import numpy as np

def worldpose_add_gamename_as_action() -> None:
    with open("./wp_hr_conf_cam_source.pkl", "rb") as f:
        data = pickle.load(f)
    data_test = data["test"]
    game_list = []
    for idx, source_name in enumerate(data_test["source"]):
        game_name = source_name.split("+")[0]
        game_list.append(game_name)
    data["test"]["action"] = game_list
    with open("./wp_hr_conf_cam_source_final.pkl", "wb") as fo:
        pickle.dump(data, fo)


def no_conf_file_create() -> None:
    with open("./wp_hr_conf_cam_source_final.pkl", "rb") as f:
        data = pickle.load(f)
    del data["test"]["confidence"]
    del data["train"]["confidence"]

    train_joint2d = data["train"]["joint_2d"].copy()
    for i in tqdm(range(train_joint2d.shape[0])):
        for j in range(train_joint2d.shape[1]):
            for k in range(train_joint2d.shape[2]):
                train_joint2d[i][j][k] = np.floor(data["train"]["joint3d_image"][i][j][k])

    test_joint2d = data["test"]["joint_2d"].copy()
    for i in tqdm(range(test_joint2d.shape[0])):
        for j in range(test_joint2d.shape[1]):
            for k in range(test_joint2d.shape[2]):
                test_joint2d[i][j][k] = np.floor(data["test"]["joint3d_image"][i][j][k])

    del data["test"]["joint_2d"]
    del data["train"]["joint_2d"]

    data["train"]["joint_2d"] = train_joint2d
    data["test"]["joint_2d"] = test_joint2d

    with open("./wp_no_conf_cam_source_final.pkl", "wb") as fo:
        pickle.dump(data, fo)


def main() -> None:
    worldpose_add_gamename_as_action()

    no_conf_file_create()

if __name__ == '__main__':
    main()
