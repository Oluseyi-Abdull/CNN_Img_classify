import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.test.gpu_device_name() #run to make sure tensorflow is connected to gpu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
import imghdr 
import numpy as np
import pandas as pd
import cv2  
import os  
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show


gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))

print(gpus)

#Aviod out of memory errors by setting GPU memory consumption growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

data_dir = 'images'

for image_class in os.listdir(data_dir):
    #print(image_class)
    for image in os.listdir(os.path.join(data_dir,image_class)):
        #print(image)
        image_path = os.path.join(data_dir,image_class,image)
        img =cv2.imread(image_path)
        #print(img)

img = cv2.imread(os.path.join('images','b0','b0.jpg'))
print(img.shape)

#print(plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
#plt.show()

data = tf.keras.utils.image_dataset_from_directory('images')

data = data.map(lambda x, y: (x/255,y))

scaled_iterator = data.as_numpy_iterator()

batch = scaled_iterator.next()

fig, ax = plt.subplots(ncols = 4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
    
len(data)

train_size = int(len(data)*7)
val_size = int(len(data)*2)+1
test_size = int(len(data)*1)+1

train_size + val_size + test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+ val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16,(3,3), 1, activation = 'relu', input_shape = (256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

model.summary()

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

hist = model.fit(train, epochs = 20, validation_data = val, callbacks = [tensorboard_callback])

##fig = plt.figure()
##plt.plot(hist.history['loss'],color = 'blue', label = 'loss')
##plt.plot(hist.history['val_loss'],color = 'orange', label = 'val_loss')
##fig.suptitle('Loss', fontsize = 22)
##plt.legend(loc = 'upper left')
##plt.show()

model.save(os.path.join('models','uno_recognition_model.h5'))
              
new_model = load_model(os.path.join('models','uno_recognition_model.h5'))

resize = tf.image.resize(img, (256,256))

modelnew = new_model.predict(np.expand_dims(resize/255, 0))

#print(modelnew)

#testimg = cv2.imread(os.path.join('images','r0','r0.jpg'))
#plt.imshow(cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB))
#plt.imshow(resize.numpy().astype(int))
plt.show()

##if modelnew == 0:
##    print("Predicted class is Blue Zero")
##elif modelnew == 1:
##    print("Predicted class is Blue One")
##elif modelnew == 2:
##    print("Predicted class is Blue Two")
##elif modelnew == 3:
##    print("Predicted class is Blue Three")
##elif modelnew == 4:
##    print("Predicted class is Blue Four")
##elif modelnew == 5:
##    print("Predicted class is Blue Five")
##elif modelnew == 6:
##    print("Predicted class is Blue Six")
##elif modelnew == 7:
##    print("Predicted class is Blue Seven")
##elif modelnew == 8:
##    print("Predicted class is Blue Eight")
##elif modelnew == 9:
##    print("Predicted class is Blue Nine")
##elif modelnew == 10:
##    print("Predicted class is Blue Reverse")
##elif modelnew == 11:
##    print("Predicted class is Blue Skip")
##elif modelnew == 12:
##    print("Predicted class is Blue +2")
##elif modelnew == 13:
##    print("Predicted class is Green Zero")
##elif modelnew == 14:
##    print("Predicted class is Green One")
##elif modelnew == 15:
##    print("Predicted class is Green Two")
##elif modelnew == 16:
##    print("Predicted class is Green Three")
##elif modelnew == 17:
##    print("Predicted class is Green Four")
##elif modelnew == 18:
##    print("Predicted class is Green Five")
##elif modelnew == 19:
##    print("Predicted class is Green Six")
##elif modelnew == 20:
##    print("Predicted class is Green Seven")
##elif modelnew == 21:
##    print("Predicted class is Green Eight")
##elif modelnew == 22:
##    print("Predicted class is Green Nine")
##elif modelnew == 23:
##    print("Predicted class is Green Reverse")
##elif modelnew == 24:
##    print("Predicted class is Green Skip")
##elif modelnew == 25:
##    print("Predicted class is Green +2")
##elif modelnew == 26:
##    print("Predicted class is Special Blank")
##elif modelnew == 27:
##    print("Predicted class is Special Shuffle")
##elif modelnew == 28:
##    print("Predicted class is Special +4")
##elif modelnew == 29:
##    print("Predicted class is Wild Card")
##elif modelnew == 30:
##    print("Predicted class is Red Zero")
##elif modelnew == 31:
##    print("Predicted class is Red One")
##elif modelnew == 32:
##    print("Predicted class is Red Two")
##elif modelnew == 33:
##    print("Predicted class is Red Three")
##elif modelnew == 34:
##    print("Predicted class is Red Four")
##elif modelnew == 35:
##    print("Predicted class is Red Five")
##elif modelnew == 36:
##    print("Predicted class is Red Six")
##elif modelnew == 37:
##    print("Predicted class is Red Seven")
##elif modelnew == 38:
##    print("Predicted class is Red Eight")
##elif modelnew == 39:
##    print("Predicted class is Red Nine")
##elif modelnew == 40:
##    print("Predicted class is Red Reverse")
##elif modelnew == 41:
##    print("Predicted class is Red Skip")
##elif modelnew == 42:
##    print("Predicted class is Red +2")
##elif modelnew == 43:
##    print("Predicted class is Yellow Zero")
##elif modelnew == 44:
##    print("Predicted class is Yellow One")
##elif modelnew == 45:
##    print("Predicted class is Yellow Two")
##elif modelnew == 46:
##    print("Predicted class is Yellow Three")
##elif modelnew == 47:
##    print("Predicted class is Yellow Four")
##elif modelnew == 48:
##    print("Predicted class is Yellow Five")
##elif modelnew == 49:
##    print("Predicted class is Yellow Six")
##elif modelnew == 50:
##    print("Predicted class is Yellow Seven")
##elif modelnew == 51:
##    print("Predicted class is Yellow Eight")
##elif modelnew == 52:
##    print("Predicted class is Yellow Nine")
##elif modelnew == 53:
##    print("Predicted class is Yellow Reverse")
##elif modelnew == 54:
##    print("Predicted class is Yellow Skip")
##elif modelnew == 55:
##    print("Predicted class is Yellow +2")
##else:
##    print("unidentified")
