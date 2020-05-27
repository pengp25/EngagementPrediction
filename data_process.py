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

    features_in_need_1 = [gdv, gdr]         # feature set No.1 2
    features_in_need_2 = [hl, hpv]          # feature set No.6 7
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

    el = landmark(pd_data, 'eye')           # feat 3
    fl = landmark(pd_data, 'facial')        # feat 8

    for feature_set in features_in_need_1:          # feat 1, 2
        compute_feature(pd_data, feature_set)

    compute_feature(el, el.columns)                 # feat 3

    for feature_set in features_in_need_2:          # feat 6, 7
        compute_feature(pd_data, feature_set)

    compute_feature(fl, fl.columns)                 # feat 8


def landmark(df: pd.DataFrame, mark_kind):
    df_features = None
    if mark_kind == 'eye':
        columns = ['eye_lmk_x', 'eye_lmk_y', 'eye_lmk_X', 'eye_lmk_Y', 'eye_lmk_Z']
        df_features = pd.DataFrame(columns=columns)
        for column in columns:
            features = [column + '_' + i.__str__() for i in range(0, 56)]
            df_features[column] = df[features].mean(axis=1)
    if mark_kind == 'facial':
        columns = ['x', 'y', 'X', 'Y', 'Z']
        df_features = pd.DataFrame(columns=columns)
        for column in columns:
            features = [column + '_' + i.__str__() for i in range(0, 67)]
            df_features[column] = df[features].mean(axis=1)
    return df_features


if __name__ == '__main__':
    process("example.csv")

