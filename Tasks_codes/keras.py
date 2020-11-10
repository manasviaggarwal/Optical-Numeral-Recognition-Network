import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam


def one_hot_encoding(self,Y,nameofset):
    encoded_list = []
    if nameofset=="MNIST":
        for value in Y:
            #print("h")
            i = [0 for _ in range(10)]
            i[value] = 1
            encoded_list.append(i)
        #print(encoded_list)
        Y=np.array(encoded_list)
        return Y
    elif nameofset=="Cat-Dog":
        for value in Y:
            i = [0 for _ in range(2)]
            i[value] = 1
            encoded_list.append(i)
        Y=np.array(encoded_list)
        return Y

#this program needs outp i.e. labels vector and V matrix of train data.

outp=one_hot_encoding(np.array(outp))

trainset=np.array(V)
labelset=np.array(outp)

indetrainset = np.array(list(range(len(trainset))))
np.random.shuffle(indetrainset)
trainset = trainset[indetrainset]
labelset = labelset[indetrainset]

model = Sequential()
model.add(Dense(784*2, activation='relu', input_dim=trainset.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(labelset.shape[1], activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = Adam(lr=0.001,)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(trainset, labelset,epochs=100,batch_size=10)
