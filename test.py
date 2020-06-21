import numpy as np
from keras.models import Model

from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv3D,MaxPooling3D,AveragePooling3D,concatenate,Activation,ZeroPadding3D
from keras.optimizers import SGD
from keras.layers import add,Flatten
from keras.optimizers import Adam
import os
import random
from sklearn.model_selection import train_test_split
from scipy.stats import beta

def Conv3d_BN(x, nb_filter,kernel_size, strides=(1,1,1), padding='same',name=None):

    if name is not None:

        bn_name = name + '_bn'

        conv_name = name + '_conv'

    else:

        bn_name = None

        conv_name = None

 

    x = Conv3D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)

    x = BatchNormalization(axis=-1,name=bn_name)(x)

    return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1,1), with_conv_shortcut=False):

    x = Conv3d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')

    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')

    if with_conv_shortcut:

        shortcut = Conv3d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)

        x = add([x,shortcut])

        return x

    else:

        x = add([x,inpt])

        return x



inpt = Input(shape=(100,100,100,1))

x = ZeroPadding3D((3,3,3))(inpt)

x = Conv3d_BN(x,nb_filter=64,kernel_size=(7,7,7),strides=(2,2,2),padding='valid')

x = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2),padding='same')(x)



x = Conv_Block(x,nb_filter=64,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=64,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=64,kernel_size=(3,3,3))



x = Conv_Block(x,nb_filter=128,kernel_size=(3,3,3),strides=(2,2,2),with_conv_shortcut=True)

x = Conv_Block(x,nb_filter=128,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=128,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=128,kernel_size=(3,3,3))



x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3),strides=(2,2,2),with_conv_shortcut=True)

x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=256,kernel_size=(3,3,3))



x = Conv_Block(x,nb_filter=512,kernel_size=(3,3,3),strides=(2,2,2),with_conv_shortcut=True)

x = Conv_Block(x,nb_filter=512,kernel_size=(3,3,3))

x = Conv_Block(x,nb_filter=512,kernel_size=(3,3,3))

x = AveragePooling3D(pool_size=(4,4,4))(x)

x = Flatten()(x)

x = Dense(1,activation='sigmoid')(x)

 

model = Model(inputs=inpt,outputs=x)

train=np.load('train_mask.npy')


label = np.loadtxt('train_val.csv',delimiter=",",skiprows=1,usecols=1)

def mixup():
    for i in range(350):
        lamda=np.random.beta(0.2,0.2)
        src1=random.randint(0,349)
        X_train[i]=lamda*X_train[src1]+(1-lamda)*X_train[i]
        Y_train[i]=lamda*Y_train[src1]+(1-lamda)*Y_train[i]




sgd = SGD(lr=0.05,momentum=0.9,decay=0.000001, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.load_weights('val_resnet_warmup.h5')

for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(train, label, test_size=115)
    mixup()
    model.fit(X_train,Y_train,epochs=10,validation_data=(X_test,Y_test),batch_size=8)

X_train, X_test, Y_train, Y_test = train_test_split(train, label, test_size=115)
model.fit(X_train,Y_train,epochs=20,validation_data=(X_test,Y_test),batch_size=8)

tpath='./test'
predictions=np.zeros(117)
file_num=0
if not os.path.isdir(tpath):
    print('false')
else:
    for item in os.listdir(tpath):
        tmp=np.load(os.path.join(tpath,item))
        voxel= tmp['voxel']/255
        mask=tmp['seg']
        voxel=np.multiply(voxel,mask)
        voxel=voxel.reshape(1,100,100,100,1)
        predictions[file_num]=model.predict(voxel)
        file_num+=1

np.savetxt('submission.csv',predictions)

