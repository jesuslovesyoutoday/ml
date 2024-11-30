import random
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

def visualise(x1, y1, x2, y2, point, answ, pred, r):
    fig, ax = plt.subplots(figsize=(10,10))
    crcl = plt.Circle((point[0], point[1]), r, color='yellow')
    ax.add_patch(crcl)
    ax.scatter(x1, y1, marker="^", c="red", label='0')
    ax.scatter(x2, y2, marker="s", c="blue", label='1')
    ax.scatter(point[0], point[1], marker="o", c="green", label='point')
    ax.set_title("answ = " + str(answ) + " pred = " + str(pred))
    ax.legend()
    plt.show()

def generate_tasks(fn, N, W, H):

    coords  = []
    classes = []

    X = []
    Y = []

    for i in range(N):
        x = random.randrange(0, W)
        y = random.randrange(0, H)
        cl = random.randrange(0, 2)
        X.append(x)
        Y.append(y)
        coords.append([x, y])
        classes.append(cl)

    point = coords[0]
    answ = classes[0]

    x1 = []
    x2 = []
    y1 = []
    y2 = []

    for i in range(1, N):
        if classes[i] == 0:
            x1.append(X[i])
            y1.append(Y[i])
        elif classes[i] == 1:
            x2.append(X[i])
            y2.append(Y[i])

    f = open(fn, 'w')
    f.write(" ".join(str(item) for item in x1))
    f.write("\n")
    f.write(" ".join(str(item) for item in y1))
    f.write("\n")
    f.write(" ".join(str(item) for item in x2))
    f.write("\n")
    f.write(" ".join(str(item) for item in y2))
    f.write("\n")
    f.write(" ".join(str(item) for item in (point + [answ])))
    f.close()

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def knn(x1, y1, x2, y2, point, k):
    N = len(x1) + len(x2)
    dist = [distance(point, [(x1+x2)[i], (y1+y2)[i]]) for i in range(N)]
    ind  = [i for i in range(N)]
    
    data = {ind[i]: dist[i] for i in range(N)}
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    items = list(data.keys())
        
    i = 0
    count_0 = 0
    count_1 = 0
    while (i < k):
        if(items[i] < len(x1)):
            count_0 += 1
        else:
            count_1 += 1
        i += 1
    if (count_0 > count_1):
        cl = 0
    elif (count_0 < count_1):
        cl = 1
    else:
        cl = 'cant define'

    return (cl, list(data.values())[k-1])

def accuracy(TP, TN, FP, FN):
    return ((TP + TN)/(TP + TN + FP + FN))

def precision(TP, FP):
    return (TP/(TP + FP))

def recall(TP, FN):
    return (TP/(TP + FN))

def f_1_score(prec, rec):
    return (2 * prec * rec / (prec + rec))

def best_k(acc, prec, rec, f1, n):
    data = [0]
    for i in range(1, n):
        data.append(acc[i] + prec[i] + rec[i] + f1[i])
    D = [max(acc), max(prec), max(rec), max(f1)]
    DD = [acc.index(max(acc)), prec.index(max(prec)), 
          rec.index(max(rec)), f1.index(max(f1))]
    M = max(D)
    d = [acc, prec, rec, f1]
    ret = DD[D.index(M)]
    #return ret
    #return(data.index(max(data)))
    return(acc.index(max(acc)))
    #return(prec.index(max(prec)))

def train(n):
    point = []
    data = []
    answ = []
    pred = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(n):
        f = open("train/" + str(i) + '.txt', 'r')
        data.append(f.readlines())
        x1.append([int(j) for j in data[i][0].split()])
        y1.append([int(j) for j in data[i][1].split()])
        x2.append([int(j) for j in data[i][2].split()])
        y2.append([int(j) for j in data[i][3].split()])
        point.append([int(j) for j in data[i][4].split()])
        answ.append(point[i][2])
        point[i] = point[i][:-1]

    acc = [0]
    prec = [0]
    rec = [0]
    f1 = [0]

    for k in range(1, N):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pr = []
        for i in range(n):
            predict, r = knn(x1[i], y1[i], x2[i], y2[i], point[i], k)
            pr.append(predict)
            #print("i = ", i, "k = ", k, "predict = ", 
            #                    predict, "answer = ", answ[i])
            if (answ[i] == 0 and predict == 0):
                TN += 1
            elif (answ[i] == 0 and predict == 1):
                FP += 1
            elif (answ[i] == 1 and predict == 0):
                FN += 1
            elif (answ[i] == 1 and predict == 1):
                TP += 1
        acc.append(accuracy(TP, TN, FP, FN))
        prec.append(precision(TP, FP))
        rec.append(recall(TP, FN))
        f1.append(f_1_score(prec[k], rec[k]))
        pred.append(pr)

    K = best_k(acc, prec, rec, f1, N)
    #print(K)
    #for i in range(n):
        #visualise(x1[i], y1[i], x2[i], y2[i], point[i], answ[i], pred[K+1][i])
    return K

def test(n, k):

    point = []
    data = []
    answ = []
    pred = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(n):
        f = open("test/" + str(i) + '.txt', 'r')
        data.append(f.readlines())
        x1.append([int(j) for j in data[i][0].split()])
        y1.append([int(j) for j in data[i][1].split()])
        x2.append([int(j) for j in data[i][2].split()])
        y2.append([int(j) for j in data[i][3].split()])
        point.append([int(j) for j in data[i][4].split()])
        answ.append(point[i][2])
        point[i] = point[i][:-1]

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(n):
        predict, r = knn(x1[i], y1[i], x2[i], y2[i], point[i], k)
        #print("i = ", i, "k = ", k, "predict = ", 
        #                    predict, "answer = ", answ[i])
        visualise(x1[i], y1[i], x2[i], y2[i], point[i], answ[i], predict, r)
        if (answ[i] == 0 and predict == 0):
            TN += 1
        elif (answ[i] == 0 and predict == 1):
            FP += 1
        elif (answ[i] == 1 and predict == 0):
            FN += 1
        elif (answ[i] == 1 and predict == 1):
            TP += 1
    acc = accuracy(TP, TN, FP, FN)
    prec = precision(TP, FP)
    rec = recall(TP, FN)
    f1 = f_1_score(prec, rec)

    return(acc, prec, rec, f1)


n = 10000
N = 20
W = 1000
H = 1000
for i in range(n):
    fname = "train/" + str(i) + '.txt'
    generate_tasks(fname, N, W, H)

K = train(n)

print("k after training: k = ", K, "\n")

for i in range(n):
    fname = "test/" + str(i) + '.txt'
    generate_tasks(fname, N, W, H)

#n = 1000

acc, prec, rec, f1 = test(n, K)
print("testing results:")
print("accuracy = ", acc)
print("precision = ", prec)
print("recall = ", rec)
print("f1 = ", f1)


