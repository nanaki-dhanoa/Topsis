import pandas as pd
import numpy as np
import sys
import string
import copy


def topsis_cli():
    if len(sys.argv) != 5:
        print('Incorrect number of parameters.')
        sys.exit()

    filename = sys.argv[1]
    try:
        dataframe = pd.read_csv(filename, header=0, index_col=0)
    except FileNotFoundError:
        print("Wrong file or file path")
        exit()

    dataframe2 = copy.deepcopy(dataframe)

    if len(dataframe2.columns) < 3:
        print("Input file must contain three or more columns.")
        exit()

    weights = sys.argv[2]
    weights = list(map(float, weights.split(',')))

    impacts = sys.argv[3]
    impacts = list(map(str, impacts.split(',')))

    try:
        data = dataframe2.values.astype(float)
    except ValueError:
        print("Only numeric values allowed.")
        exit()

    (rows, cols) = data.shape

    if len(weights) != cols:
        print("No of weights is unequal.")
        exit()
    if len(impacts) != cols:
        print("No of impacts is unequal.")
        exit()

    for i in impacts:
        if i != '+' and i != '-':
            print("Impacts should be '+' or '-' only.")
            exit()

    for i in string.punctuation:
        if i != ',':
            if i in weights:
                print("Weights must be separated by Comma(',').")
                exit()
            if i != '+' and i != '-' and i in impacts:
                print("Impacts must be separated by Comma(',').")
                exit()

    # converting impacts from +/- to 1/0
    encode_impacts = []
    for i in impacts:
        if i == '+':
            encode_impacts.append(1)
        else:
            encode_impacts.append(0)

    # normalizing weights between 0 and 1
    t = sum(weights)
    for i in range(cols):
        weights[i] /= t

    a = [0.0] * cols  # Will finally store the normalizing factor for each attribute

    for i in range(0, rows):
        for j in range(0, cols):
            a[j] += (data[i][j]) ** 2

    for j in range(cols):
        a[j] = (a[j]) ** 0.5

    # Normalizing the attribute vectors and scaling them with the normalized weights
    for i in range(rows):
        for j in range(cols):
            data[i][j] /= a[j]
            data[i][j] *= weights[j]

    # Computing the Ideal postive and negative solution, based on impacts of each attribute
    idp = np.amax(data, axis=0)  # MAX IN VERTICAL COLUMN
    idn = np.amin(data, axis=0)  # MIN IN EACH COLUMN
    for i in range(len(impacts)):
        if impacts[i] == '-':
            temp = idp[i]
            idp[i] = idn[i]
            idn[i] = temp

    pos_dist = list()
    neg_dist = list()

    for i in range(rows):
        t = 0
        for j in range(cols):
            t += pow((data[i][j] - idp[j]), 2)

        pos_dist.append(float(pow(t, 0.5)))

    for i in range(rows):
        t = 0
        for j in range(cols):
            t += pow((data[i][j] - idn[j]), 2)

        neg_dist.append(float(pow(t, 0.5)))

    performance_score = dict()

    for i in range(rows):
        performance_score[i + 1] = neg_dist[i] / (neg_dist[i] + pos_dist[i])

    a = list(performance_score.values())
    b = sorted(list(performance_score.values()), reverse=True)

    rank = dict()

    for i in range(len(a)):
        rank[(b.index(a[i]) + 1)] = a[i]

    rk = list(rank.keys())

    dataframe2['Topsis Score'] = a
    dataframe2['Rank'] = rk
    output = pd.DataFrame(dataframe2)

    result_name = sys.argv[4]
    output.to_csv(result_name)

    return

if __name__=="__main__":
    topsis_cli()