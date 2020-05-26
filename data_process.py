import csv
import numpy as np
import pandas as pd


def get_feature(df: pd.DataFrame, columns):
    feature = []
    for column in columns:
        vector = df[column].to_numpy()
        deviation = vector.std()
        max_v = vector.max()
        variation_range = max_v - vector.min()
        feature.append([deviation, variation_range, max_v])
    return feature


def process(file_dir : str):
    csv_file = csv.reader(open(file_dir, 'r', encoding='UTF8'))
    csv_data = [row for row in csv_file]
    columns = csv_data[0]

    # convert type: str -> float
    data = []
    for i in range(1, len(csv_data)):
        data.append(list(map(float, csv_data[i])))

    pd_data = pd.DataFrame(data, columns=columns)
    print(get_feature(pd_data, columns[4:7]))
    pass


if __name__ == '__main__':
    process("example.csv")

