import numpy as np
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.models import Sequential
import json
import cv2

# kích thước grid system 
cell_size = 7 
# số boundary box cần dự đoán mỗi ô vuông
box_per_cell = 2
# kích thước ảnh đầu vào
img_size = 224
# số loại nhãn
classes = {'circle':0, 'triangle':1,  'rectangle':2}
nclass = len(classes)

box_scale = 5.0
noobject_scale = 0.5
batch_size = 4
# số lần huấn luyện
epochs = 10
# learning của chúng ta
lr = 1e-3

def load():
    labels = json.load(open('/home/v000284/Documents/hw/train/labels.json'))
    # số lương ảnh
    N = len(labels[:5000])
    # matrix chứa ảnh
    X = np.zeros((N, img_size, img_size, 3), dtype='uint8')
    # matrix chứa nhãn của ảnh tương ứng
    y = np.zeros((N,cell_size, cell_size, 5+nclass))
    for idx, label in enumerate(labels[:5000]):
        img = cv2.imread("train/{}.png".format(idx))
        # normalize về khoảng [0-1]
        X[idx] = img
        for box in label['boxes']:
            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']
            # one-hot vector của nhãn object
            cl = [0]*len(classes)
            cl[classes[box['class']]] = 1
            # tâm của boundary box
            x_center, y_center, w, h = (x1+x2)/2.0, (y1+y2)/2.0, x2-x1, y2-y1
            # index của object trên ma trận ô vuông 7x7
            x_idx, y_idx = int(x_center/img_size*cell_size), int(y_center/img_size*cell_size)
            # gán nhãn vào matrix 
            y[idx, x_idx, y_idx] = 1, x_center, y_center, w, h, *cl
    
    return X, y

def iou(box_1, box_2):
    # box_1: x, y, w, h
    # box_2: x, y, w, h
    tb = min(box_1[0] + 0.5 * box_1[2], box_2[0] + 0.5 * box_2[2]) - \
        max(box_1[0] - 0.5 * box_1[2], box_2[0] - 0.5 * box_2[2])
    lr = min(box_1[1] + 0.5 * box_1[3], box_2[1] + 0.5 * box_2[3]) - \
        max(box_1[1] - 0.5 * box_1[3], box_2[1] - 0.5 * box_2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    return inter / (box_1[2] * box_1[3] + box_2[2] * box_2[3] - inter)

def loss():
    return 0

def net(inputs):
    model = Sequential()
    model.add(Conv2D(64, 7, strides=2, padding='same', activation='relu', input_shape=inputs))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(193, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 1, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(256, 1, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 1, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 1, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 1, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 1, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(512, 1, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 1, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, strides=2, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(Conv2D(1024, 3, padding='same', activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(13, activation='softmax'))
    return model

net = net((224, 224, 3))
net.summary()

X_train, y_train = load()
print(y_train[0])