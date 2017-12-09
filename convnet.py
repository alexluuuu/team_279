import numpy as np
import pandas as pd
import collections
import string
import matplotlib.pyplot as plt

from sklearn import metrics

from utils.DatasetPrep import GatherImageSet
from skimage import io
from skimage.transform import rescale, resize

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

image_dir = 'sample_dataset/images/'
truth_file = 'sample_dataset/ISIC-2017_Training_Part3_GroundTruth.csv'

epochs = 50
batch_size = 64
scale = 0.1
H, W = 224, 224

num_images, image_names = GatherImageSet(image_dir)
images = []
image_ids = []

for i in range(num_images):
    img = resize(io.imread(image_dir+image_names[i]), (H, W))
    images.append(img)
    image_ids.append(image_names[i])
    if i % 100 == 0: print i

df = pd.DataFrame(data = {'image_id': image_ids, 'image': images})
truth = pd.read_csv(truth_file)
df = df.join(truth, rsuffix = '_other')
    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

image_set = np.zeros((len(df['image']), H, W, 3))
for i in range(len(df['image'])):
    image_set[i] = df['image'][i]

labels = df['melanoma']

training_examples = int(len(labels) * 0.75)

X_train = image_set[:training_examples]
Y_train = labels[:training_examples]
X_test = image_set[training_examples:]
Y_test = labels[training_examples:]

train_generator = train_datagen.flow(X_train, Y_train)
test_generator = test_datagen.flow(X_test, Y_test)

# dimensions of our images.
#img_width, img_height = max_W, max_H
img_width, img_height = H, W

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])
#try SGD with momentum?

hist = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=800 // batch_size)

model.save_weights('first_try.h5')

history = hist.history
train_acc = history['acc'][-1]
val_acc = history['val_acc'][-1]

print "Train Accuracy\t\t{:.3f}".format(train_acc)
print "Validation Accuracy\t{:.3f}".format(val_acc)
print history
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.show()