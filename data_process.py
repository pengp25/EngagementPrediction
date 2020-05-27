import csv
import numpy as np
import pandas as pd


# return sigma, range, max
def compute_feature(df: pd.DataFrame, columns):
    std_feature = []
    range_feature = []
    max_feature = []
    for column in columns:
        vector = df[column].to_numpy()
        std_feature.append(vector.std())
        max_v = vector.max()
        max_feature.append(max_v)
        range_feature.append(max_v - vector.min())

    # output
    f = open('processed_data.csv', 'a+', newline='', encoding='UTF8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(columns)
    csv_writer.writerow(std_feature)
    csv_writer.writerow(range_feature)
    csv_writer.writerow(max_feature)
    print("output finish!")


def process(file_dir: str):
    csv_file = csv.reader(open(file_dir, 'r', encoding='UTF8'))
    csv_data = [row for row in csv_file]

    columns = list(map(lambda x: x.lstrip(), csv_data[0]))    # delete left blank for every title
    # print(columns)

    # convert type: str -> float
    data = []
    for i in range(1, len(csv_data)):
        data.append(list(map(float, csv_data[i])))

    pd_data = pd.DataFrame(data, columns=columns)

    gdv, gdr, hl, hpv = [], [], [], []      # feature set No.1, 2, 6, 7 in paper
    aui = []                                # feature set No. 5

    features_in_need = [gdv, gdr, aui, hl, hpv]
    for column in columns:
        if 'gaze_0' in column or 'gaze_1' in column:
            gdv.append(column)
        elif 'gaze_angle' in column:
            gdr.append(column)
        elif 'pose_T' in column:
            hl.append(column)
        elif 'pose_R' in column:
            hpv.append(column)
        elif 'AU' in column and 'r' in column:
            aui.append(column)
    # gdv = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_0_x', 'gaze_1_y', 'gaze_1_z']
    # features_in_need.append(gdv)

    # gdr = ['gaze_angle_x', 'gaze_angle_y']
    # features_in_need.append(gdr)

    # hl = ['pose_Tx', 'pose_Ty', 'pose_Tz']      # head location
    # features_in_need.append(hl)

    # hpv = ['pose_Rx', 'pose_Ry', 'pose_Rz']     # head pose vector
    # features_in_need.append(hpv)
    print(features_in_need)

    for feature_set in features_in_need:
        compute_feature(pd_data, feature_set)


if __name__ == '__main__':
    process("example.csv")
