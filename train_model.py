##################### LIBRARIES #######################
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torchvision
import torch
import PIL
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D, Convolution2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

####################### LOAD DATA ##########################
train_df = pd.read_csv(
    "train_test_files/split_of_60%training and 40%testing/train.txt", sep=" ", names=["image", "score"])
test_df = pd.read_csv(
    "train_test_files/split_of_60%training and 40%testing//test.txt", sep=" ", names=["image", "score"])
train_df['purpose'] = 'train'
test_df['purpose'] = 'test'

np.random.seed(17)
val_set_size = int((40*train_df.shape[0])/100)
val_idx = np.random.choice(train_df.shape[0], size=val_set_size)
train_df.loc[val_idx, 'purpose'] = 'validation'
df = pd.concat([train_df, test_df])
del train_df, test_df

p = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
count = 0


def retrievePixels(img_name):
    global count
    path = f"/content/drive/My Drive/data/Images/{img_name}"
    img = PIL.Image.open(path)
    img = p(img)
    img = img.numpy()
    x = img.reshape(1, -1)[0]
    print(count)
    count = count+1

    return x


df['pixels'] = df['image'].apply(retrievePixels)
features = []
pixels = df['pixels'].values
for i in range(0, pixels.shape[0]):
    features.append(pixels[i])

features = np.array(features)
features = features.reshape(features.shape[0], 224, 224, 3)
features = features / 255  # normalize inputs within [0, 1]


####################### Make Model ##########################

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
# LOAD A PRETRAINED MODEL FOR BETTER PERFORMANCEE ########################3
model.load_weights('vgg_face_weights.h5')

num_of_classes = 1  # this is a regression problem

# freeze all layers of VGG-Face except last 7 one
for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Flatten()(model.layers[-4].output)
base_model_output = Dense(num_of_classes)(base_model_output)

beauty_model = Model(inputs=model.input, outputs=base_model_output)

######################### TRain#############################
beauty_model.compile(loss='mean_squared_error',
                     optimizer=keras.optimizers.Adam())

checkpointer = ModelCheckpoint(
    filepath='beauty_model.hdf5', monitor="val_loss", verbose=1, save_best_only=True, mode='auto'
)

earlyStop = EarlyStopping(monitor='val_loss', patience=20)
train_idx = df[(df['purpose'] == 'train')].index
val_idx = df[(df['purpose'] == 'validation')].index
test_idx = df[(df['purpose'] == 'test')].index

score = beauty_model.fit(
    features[train_idx], df.iloc[train_idx].score, epochs=5000, validation_data=(features[val_idx], df.iloc[val_idx].score), callbacks=[checkpointer, earlyStop]
)
beauty_model.load_weights("beauty_model.hdf5")
beauty_model.save("model_.h5")
