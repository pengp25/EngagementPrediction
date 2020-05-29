import csv
import numpy as np
import pandas as pd
from math import floor

first_row = True


def process(data, columns):
    global first_row
    pd_data = pd.DataFrame(data, columns=columns)
    pd_output = pd.DataFrame()
    time = pd_data['timestamp'][pd_data.shape[0] -1] - pd_data['timestamp'][0]

    for data_type in ['eye', 'facial']:                     # feat No.3, 8
        l_m = landmark(pd_data, data_type)
        deviation = pd.DataFrame(l_m.std()).T
        deviation.columns = list(map(lambda x: x+"_d", deviation.columns))
        pd_range = pd.DataFrame(l_m.max() - l_m.min()).T
        pd_range.columns = list(map(lambda x: x + "_r", pd_range.columns))
        pd_output = pd.concat([pd_output, deviation, pd_range], axis=1)

    title_features = ['gaze_0', 'gaze_1', 'gaze_angle', 'pose_T', 'pose_R']
    for column in columns:
        for title_feature in title_features:
            if title_feature in column:
                deviation = pd.DataFrame([pd_data[column].std()], columns=[column + '_d'])
                pd_range = pd.DataFrame([pd_data[column].max() - pd_data[column].min()], columns=[column + '_r'])
                pd_output = pd.concat([pd_output, deviation, pd_range], axis=1)
                break

        if 'AU' in column and 'r' in column:
            pd_max = pd.DataFrame([pd_data[column].max()], columns=[column + '_m'])
            deviation = pd.DataFrame([pd_data[column].std()], columns=[column + '_d'])
            pd_range = pd.DataFrame([pd_data[column].max() - pd_data[column].min()], columns=[column + '_r'])
            pd_output = pd.concat([pd_output, pd_max, deviation, pd_range], axis=1)

        if 'AU' in column and 'c' in column:
            pd_frequent = pd.DataFrame([pd_data[column].sum()/time], columns=[column + '_f'])
            pd_output = pd.concat([pd_output, pd_frequent], axis=1)

    # print(pd_output)
    # print(pd_output.columns.size)
    if first_row:
        pd_output.to_csv('output.csv', mode='a', encoding='UTF8', index=False)
        first_row = False
    else:
        pd_output.to_csv('output.csv', mode='a', encoding='UTF8', index=False, header=False)


def landmark(df: pd.DataFrame, mark_kind):
    df_features = None
    if mark_kind == 'eye':
        columns = ['eye_lmk_x', 'eye_lmk_y', 'eye_lmk_X', 'eye_lmk_Y', 'eye_lmk_Z']
        df_features = pd.DataFrame(columns=columns)
        for column in columns:
            features = [column + '_' + i.__str__() for i in range(0, 56)]
            df_features[column] = df[features].mean(axis=1)/100
    if mark_kind == 'facial':
        columns = ['x', 'y', 'X', 'Y', 'Z']
        df_features = pd.DataFrame(columns=columns)
        for column in columns:
            features = [column + '_' + i.__str__() for i in range(0, 67)]
            df_features[column] = df[features].mean(axis=1)/100
    return df_features


def compute(file_dir: str):
    csv_file = csv.reader(open(file_dir, 'r', encoding='UTF8'))
    csv_data = [row for row in csv_file]

    columns = list(map(lambda x: x.lstrip(), csv_data[0]))    # delete left blank for every title
    # print(columns)

    # convert type: str -> float
    data = []
    for i in range(1, len(csv_data)):
        data.append(list(map(float, csv_data[i])))
    if len(data) > 550:             # first 200 frames + last 200 frames + 150 frames for computing at least
        windows = floor((len(data) - 400)/150)
        for i in range(0, 150):
            segment = data[i:i+windows]
            process(segment, columns)
            print("finished segment{}/150".format(i))
    else:
        print("too short to compute")


if __name__ == '__main__':
    compute("example.csv")

