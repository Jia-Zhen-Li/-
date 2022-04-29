# coding: utf-8

import tensorflow 
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist, mnist
import numpy as np

from sklearn.preprocessing import StandardScaler
from scipy.ndimage.interpolation import shift
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

import os
from PIL import Image
import cv2
#cv2.resize(np.array(img_1), (28, 28), interpolation=cv2.INTER_NEAREST)
#讀取資料夾mnist下的42000張圖片，圖片為灰階圖，所以為1通道，圖像大小28*28
def load_data():
	data_1 = np.empty((248,256,256,3),dtype="float32")
	label_1 = np.empty((248,),dtype="uint8")
	data_2 = np.empty((36,256,256,3),dtype="float32")
	label_2 = np.empty((36,),dtype="uint8")
	
	imgs_1 = os.listdir("./trainImg")
	num_1 = len(imgs_1)
	for i in range(num_1):
		img_1 = Image.open("./trainImg/"+imgs_1[i])
		arr_1 = np.asarray(img_1.resize((256,256),Image.NEAREST),dtype="float32")
		data_1[i,:,:,:] = arr_1
		label_1[i] = int(imgs_1[i].split(' ')[0])
		
	imgs_2 = os.listdir("./testImg")
	num_2 = len(imgs_2)
	for i in range(num_2):
		img_2 = Image.open("./testImg/"+imgs_2[i])
		arr_2 = np.asarray(img_2.resize((256,256),Image.NEAREST),dtype="float32")
		data_2[i,:,:,:] = arr_2
		label_2[i] = int(imgs_2[i].split(' ')[0])
	
	return (data_1,label_1), (data_2,label_2)
#from data import load_data
# Step 1. 資料準備

(x_train,y_train),(x_test,y_test)= load_data()

x_train, x_val, y_train, y_val = x_train[:212], x_train[212:], y_train[:212], y_train[212:]

width, height, channels = x_train.shape[1], x_train.shape[2], 3
x_train = x_train.reshape((x_train.shape[0], width, height, channels))
x_val = x_val.reshape((x_val.shape[0], width, height, channels))
x_test = x_test.reshape((x_test.shape[0], width, height, channels))



x_train_normalize = x_train.astype('float32') 
x_test_normalize = x_test.astype('float32') 
x_val_normalize = x_val.astype('float32') 


y_train_OneHot = tensorflow.keras.utils.to_categorical(y_train)
y_val_OneHot = tensorflow.keras.utils.to_categorical(y_val)
y_test_OneHot = tensorflow.keras.utils.to_categorical(y_test)

print(y_test_OneHot.shape)
datagen = ImageDataGenerator(rescale=1.0/255.0,featurewise_center=True,
                             featurewise_std_normalization=True,    
                             horizontal_flip=True,
                             fill_mode='nearest')
datagen.fit(x_train_normalize)

train_iterator = datagen.flow(x_train_normalize, y_train_OneHot, batch_size=32)
val_iterator = datagen.flow(x_val_normalize, y_val_OneHot, batch_size=32)
test_iterator = datagen.flow(x_test_normalize, y_test_OneHot, batch_size=32)

# Step 2. 建立模型

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

model = Sequential()

# 卷積層1與池化層1
dropout_rate = 0.25
pool_size = (2,2)
kernel_size = (3,3)
model.add(Conv2D(filters=64,kernel_size=kernel_size,
                 input_shape=(256,256,3), 
                 activation='relu', 
                 padding='same'))

model.add(Dropout(rate=dropout_rate))

model.add(MaxPooling2D(pool_size=pool_size))

# 卷積層2與池化層2

model.add(Conv2D(filters=128, kernel_size=kernel_size, 
                 activation='relu', padding='same'))

model.add(Dropout(dropout_rate))

model.add(MaxPooling2D(pool_size=pool_size))


# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())
model.add(Dropout(rate=dropout_rate))

model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=dropout_rate))


model.add(Dense(2, activation='softmax'))

print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# 載入之前訓練的模型

# try:
#     model.load_weights("./cifarCnnModel.h5")
#     print("載入模型成功!繼續訓練模型")
# except :    
print("載入模型失敗!開始訓練一個新模型")


# Step 4. 訓練模型

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
train_history = model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator),validation_data= val_iterator,validation_steps=len(val_iterator),
                    epochs=10)
#train_history=model.fit(train_iterator,validation_data= val_iterator,
#                        epochs=5, batch_size=128, verbose=1,shuffle=True)
#train_history=model.fit(x_train_normalize, y_train_OneHot,
#                        validation_split=0.2,
#                        epochs=5, batch_size=128, verbose=1,shuffle=True)          

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


# Step 6. 評估模型準確率
scores = model.evaluate_generator(test_iterator, steps=len(test_iterator),
                                  verbose=1)
#scores = model.evaluate(x_test_normalize,  y_test_OneHot, verbose=1)

scores[1]


# 進行預測

#prediction=model.predict_classes(x_test_normalize)
result=model.predict(x_test_normalize)
prediction =np.argmax(result,axis=1)
# prediction[:10]

# # 查看預測結果

# label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
#             5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
# 			
# print(label_dict)		

# import matplotlib.pyplot as plt
# def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
#     fig = plt.gcf()
#     fig.set_size_inches(12, 14)
#     if num>25: num=25 
#     for i in range(0, num):
#         ax=plt.subplot(5,5, 1+i)
#         ax.imshow(images[idx],cmap='binary')
                
#         title=str(i)+','+label_dict[labels[i][0]]
#         if len(prediction)>0:
#             title+='=>'+label_dict[prediction[i]]
            
#         ax.set_title(title,fontsize=10) 
#         ax.set_xticks([]);ax.set_yticks([])        
#         idx+=1 
#     plt.show()

# plot_images_labels_prediction(x_test,y_test, prediction,0,10)

# # 查看預測機率

# Predicted_Probability=model.predict(x_test_normalize)


# def show_Predicted_Probability(y,prediction, x_img,Predicted_Probability,i):
#     print('label:',label_dict[y[i][0]],
#           'predict:',label_dict[prediction[i]])
#     plt.figure(figsize=(2,2))
#     plt.imshow(np.reshape(x_test[i],(32, 32,3)))
#     plt.show()
#     for j in range(10):
#         print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

# show_Predicted_Probability(y_test,prediction, x_test,Predicted_Probability,0)
# show_Predicted_Probability(y_test,prediction, x_test,Predicted_Probability,3)

# # Step 8. Save Weight to h5 

# model.save_weights("./cifarCnnModel.h5")
# print("Saved model to disk")
#coding:utf-8

