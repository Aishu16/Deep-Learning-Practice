import numpy as np
import random
import sys

#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:, 1:]
trainlabels = data[:, 0]
onearray = np.ones((train.shape[0], 1))
print(onearray)
train = np.append(train, onearray, axis=1)
# np.random.shuffle(train)
# print("train=",train)
# print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:, 1:]
testlabels = data[:, 0]
onearray = np.ones((test.shape[0], 1))
test = np.append(test, onearray, axis=1)

rows = train.shape[0]
cols = train.shape[1]

# print("rows",rows)

hidden_nodes = 3
##############################
### Initialize all weights ###
w = np.random.rand(hidden_nodes)
# print("w=",w)


# check this command
# W = np.zeros((hidden_nodes, cols), dtype=float)
# W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
# print("W=",W)
epochs = 10000
eta = .01
prevobj = np.inf
i = 0
rowind = np.array([i for i in range(rows)])
###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
# print("hidden_layer=",hidden_layer)
# print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
# print("hidden_layer=",hidden_layer)
# print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
# print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
# print("obj=",obj)

# obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))

# print("Obj=",obj)
# dellW=np.ones((hidden_nodes, cols),dtype=float)
###############################
### Begin gradient descent ####
# m=4
m = int(sys.argv[3])
# m=int(sys.argv[3])
while (prevobj - obj > 0.0000001 or i < epochs):
    # while(prevobj - obj > 0):

    # Update previforous objective
    prevobj = obj
    #################Shuffle the rows##############
    # np.random.shuffle(train)
    # np.random.shuffle(trainlabels)
    np.random.shuffle(rowind)

    # Calculate gradient update for final layer (w)
    # dellw is the same dimension as w

    # print(hidden_layer[0,:].shape, w.shape)
    for k in range(0, rows, 1):
        ind = rowind[0]
        dellw = (np.dot(hidden_layer[ind, :], np.transpose(w)) - trainlabels[ind]) * hidden_layer[ind, :]
        for j in range(0, m, 1):
            ind = rowind[j]
            dellw += (np.dot(hidden_layer[ind, :], np.transpose(w)) - trainlabels[ind]) * hidden_layer[ind, :]

        # Update w
        w = w - eta * dellw
    # print("w",w)
    #	print("dellf=",de0l.0l.0.0..f)

    # Calculate gradient update for hidden layer weights (W)
    # dellW has to be of same dimension as W

    # Let's first calculate dells. After that we do dellu and dellv.
    # Here s, u, and v are the three hidden nodes
    # dells = df/dz1 * (dz1/ds1, dz1,ds2)
    ind = rowind[0]
    dells = np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[0] * (hidden_layer[ind, 0]) * (
                1 - hidden_layer[ind, 0]) * train[ind]
    for j in range(0, m, 1):
        ind = rowind[j]
        dells += np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[0] * (hidden_layer[ind, 0]) * (
                    1 - hidden_layer[ind, 0]) * train[ind]
    # print("dells",dells)
    ind = rowind[0]
    dellu = np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[1] * (hidden_layer[ind, 1]) * (
                1 - hidden_layer[ind, 1]) * train[ind]
    for j in range(0, m, 1):
        ind = rowind[j]
        dellu += np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[1] * (hidden_layer[ind, 1]) * (
                    1 - hidden_layer[ind, 1]) * train[ind]
    # print("dellu",dellu)

    ind = rowind[0]
    dellv = np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[2] * (hidden_layer[ind, 2]) * (
                1 - hidden_layer[ind, 2]) * train[ind]
    for j in range(0, m, 1):
        ind = rowind[j]
        dellv += np.sum(np.dot(hidden_layer[ind, :], w) - trainlabels[ind]) * w[2] * (hidden_layer[ind, 2]) * (
                    1 - hidden_layer[ind, 2]) * train[ind]
    # print("dellv",dellv)

    dellW = np.array([dells, dellu, dellv])

    W = W - eta * dellW
    # print("W",W)

    # Recalculate objective
    hidden_layer = np.matmul(train, np.transpose(W))
    # print("layer=",hidden_layer)

    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
    # print("hidden_layer=",hidden_layer)

    output_layer = np.matmul(hidden_layer, np.transpose(w))
    # print("output_layer=",output_layer)

    obj = np.sum(np.square(output_layer - trainlabels))
    # print("obj=",obj)

    i = i + 1
    print("Objective=", obj)

x = np.matmul(train, np.transpose(W))
predictions = np.sign(np.matmul(sigmoid(x), np.transpose(w)))
print("final prediction", predictions)
# print(w)
accuracy = np.mean(predictions - trainlabels)
print("accuracy = ", accuracy)
