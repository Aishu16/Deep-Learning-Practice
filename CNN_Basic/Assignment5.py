import numpy as np
import sys
import pandas as pd
import os
from scipy import signal
import math

#############Read images############


traindir = sys.argv[1]
df = pd.read_csv(traindir + 'data.csv')  ########Load image names and labels
# print(df)
names = df['Name'].values
labels = df['Label'].values
# print(names)


traindata = np.empty((len(labels), 3, 3), dtype=np.int8)
# print(traindata)
for i in range(0, len(labels)):
    image_matrix = np.loadtxt(traindir + '/' + names[i])
    # traindata=np.append(traindata,image_matrix,axis=0)
    traindata[i] = image_matrix

traindir = sys.argv[2]
df = pd.read_csv(traindir + 'data.csv')  ########Load image names and labels
# print(df)
names = df['Name'].values
labels = df['Label'].values
# print(names)


traindata = np.empty((len(labels), 3, 3), dtype=np.int8)
# print(traindata)
for i in range(0, len(labels)):
    image_matrix = np.loadtxt(traindir + '/' + names[i])
    # traindata=np.append(traindata,image_matrix,axis=0)
    traindata[i] = image_matrix

# print(traindata)
# print(labels)


sigmoid = lambda x: 1 / (1 + np.exp(-x))

#############################################################################

##################Initialize all weights################


# c =np.random.rand(2,2)
c = np.ones((2, 2), dtype=float)
# print(c)

epochs = 10000
eta = .1
prevobj = np.inf
i = 0

####################################################################################

#########Calculate objective#################

objective = 0
# print("c1=",c)


for i in range(0, len(labels)):
    # print("traindata[i]",traindata[i])
    hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
    # print("hidden_layer",hidden_layer)
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4
    # print("output_layer=",output_layer)
    objective += (output_layer - labels[i]) ** 2
# print("objective",objective)


################################################################################################################
####### Begin Gradient Descent #######################
stop = 0.01
while (prevobj - objective > stop and i < epochs):

    # update previous objective
    prevobj = objective

    # calculate gradient update  for final layer (w)
    # dellw is the same dimension as w

    # print("c=",c)
    dellc1 = 0
    dellc2 = 0
    dellc3 = 0
    dellc4 = 0
    # f = math.sqrt(output_layer)
    for i in range(0, len(labels)):
        # print("traindata=",traindata[i])

        ####Do the convolution
        hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])

        ###Calculate gradient for c1
        sqrtf = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4 - labels[i]
        dz1dc1 = hidden_layer[0][0] * (1 - hidden_layer[0][0]) * traindata[i][0][0]
        dz2dc1 = hidden_layer[0][1] * (1 - hidden_layer[0][1]) * traindata[i][0][1]
        dz3dc1 = hidden_layer[1][0] * (1 - hidden_layer[1][0]) * traindata[i][1][0]
        dz4dc1 = hidden_layer[1][1] * (1 - hidden_layer[1][1]) * traindata[i][1][1]
        dellc1 += (sqrtf * (dz1dc1 + dz2dc1 + dz3dc1 + dz4dc1)) / 2
        # print("delc1",dellc1)

        ###Calculate gradient for c2
        dz1dc2 = hidden_layer[0][0] * (1 - hidden_layer[0][0]) * traindata[i][0][1]
        dz2dc2 = hidden_layer[0][1] * (1 - hidden_layer[0][1]) * traindata[i][0][2]
        dz3dc2 = hidden_layer[1][0] * (1 - hidden_layer[1][0]) * traindata[i][1][1]
        dz4dc2 = hidden_layer[1][1] * (1 - hidden_layer[1][1]) * traindata[i][1][2]
        dellc2 += (sqrtf * (dz1dc2 + dz2dc2 + dz3dc2 + dz4dc2)) / 2
        # print("	print(i,"",Prediction)delc2",dellc2)

        ###calculate gradient for c3
        dz1dc3 = hidden_layer[0][0] * (1 - hidden_layer[0][0]) * traindata[i][1][0]
        dz2dc3 = hidden_layer[0][1] * (1 - hidden_layer[0][1]) * traindata[i][1][1]
        dz3dc3 = hidden_layer[1][0] * (1 - hidden_layer[1][0]) * traindata[i][2][0]
        dz4dc3 = hidden_layer[1][1] * (1 - hidden_layer[1][1]) * traindata[i][2][1]
        dellc3 += (sqrtf * (dz1dc3 + dz2dc3 + dz3dc3 + dz4dc3)) / 2
        # print("delc3",dellc3)

        ###Calculate gradient for c4
        dz1dc4 = hidden_layer[0][0] * (1 - hidden_layer[0][0]) * traindata[i][1][1]
        dz2dc4 = hidden_layer[0][1] * (1 - hidden_layer[0][1]) * traindata[i][1][2]
        dz3dc4 = hidden_layer[1][0] * (1 - hidden_layer[1][0]) * traindata[i][2][1]
        dz4dc4 = hidden_layer[1][1] * (1 - hidden_layer[1][1]) * traindata[i][2][2]
        dellc4 += (sqrtf * (dz1dc4 + dz2dc4 + dz3dc4 + dz4dc4)) / 2
    # print("delc4",dellc4)

    c[0][0] -= eta * dellc1
    c[0][1] -= eta * dellc2
    c[1][0] -= eta * dellc3
    c[1][1] -= eta * dellc4
    # print("c",c)

    ##Recalculate objective

    objective = 0
    for i in range(0, len(labels)):
        hidden_layer = signal.convolve2d(traindata[i], c, mode="valid")
        for j in range(0, 2, 1):
            for k in range(0, 2, 1):
                hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
        output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4
        objective += (output_layer - labels[i]) ** 2

    print("objective", objective)

# i,"",#Do final Prediction##############
# print("c=",c)
for i in range(0, len(labels)):
    # print(traindata[i])
    hidden_layer = signal.convolve2d(traindata[i], c, mode='valid')
    # print(1-hidden_layer)
    for j in range(0, 2, 1):
        for k in range(0, 2, 1):
            hidden_layer[j][k] = sigmoid(hidden_layer[j][k])
    # print("hiden",hidden_layer)
    output_layer = (hidden_layer[0][0] + hidden_layer[0][1] + hidden_layer[1][0] + hidden_layer[1][1]) / 4
    # print("output_layer",output_layer)
    if (output_layer > 0.5):
        print(i, "1")
    else:
        print(i, "-1")
# Prediction=np.sign(output_layer)
# print(i,"",Prediction)
