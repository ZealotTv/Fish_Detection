import os
import platform
import numpy as np
import pandas as pd
import random as python_random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import datetime
import cv2
import time
import imageio
import imgaug.augmenters as iaa
import imgaug as ia
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Add
from tensorflow.keras.layers import InputLayer, Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Activation, MaxPool2D, ZeroPadding2D
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from keras import activations
from keras import regularizers


tf.random.set_seed(73)
TPU_INIT = False

DataDir = './Fish_Dataset'

ia.seed(73)

IMG_SIZE = 224
ColorCh = 3

CATEGORIES = []
for list_ in os.listdir(DataDir):
    if not '.' in list_:
        CATEGORIES.append(list_)
        
def isValid(text):
    supported_types = ['.png', '.jpg', '.jpeg']
    for img_type in supported_types:
        if img_type in text:
            return True
        else:
            return False
def prepareData(Dir, split_ratio):
    X = []
    y = []
    Frame = []
    
    flip = iaa.Fliplr(1.0)
    zoom = iaa.Affine(scale=1)
    random_brightness = iaa.Multiply((1, 1.2))
    rotate = iaa.Affine(rotate=(-20, 20))
    
    for i, category in enumerate(CATEGORIES):
        path = os.path.join(Dir, category, (category))        
        if not os.path.isdir(path):
            pass
        
        else:
            class_num = CATEGORIES.index(category)
            limit = 500 # images from each class
            img_list = os.listdir(path)[0:limit]
            random.shuffle(img_list)
            
            for img in tqdm(img_list):
                if isValid(img):
                    orig_img = cv2.imread(os.path.join(path,img) , cv2.IMREAD_COLOR)
                    image_aug = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE), 
                                           interpolation = cv2.INTER_CUBIC)
                    
                    image_aug = flip(image = image_aug)
                    image_aug = random_brightness(image = image_aug)
                    image_aug = zoom(image = image_aug)
                    image_aug = rotate(image = image_aug)
                    
                    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                    X.append(image_aug)
                    y.append(class_num)
                else:
                    pass
 
    if len(X) > 0:
        train_len = int(len(X) * split_ratio)
        
        features = pd.DataFrame((np.array(X)).reshape(-1, IMG_SIZE * IMG_SIZE * ColorCh))
        labels = pd.DataFrame({'label': y})
        
        Frame = pd.concat([features, labels], axis=1).sample(frac = 1, random_state=73)     
        train_df, test_df = Frame[train_len:], Frame[:train_len]
        
        return train_df, test_df
      
      
train_df, test_df = prepareData(DataDir, split_ratio=0.2)

label_count_train = pd.Series(train_df['label'].values.ravel()).value_counts()
n_classes = len(label_count_train)

label_count_test = pd.Series(test_df['label'].values.ravel()).value_counts()

X_train = train_df.drop(["label"],axis = 1).to_numpy().reshape(-1,IMG_SIZE,IMG_SIZE,ColorCh).astype(np.float32) / 255.0
y_train = train_df['label']

X_test = test_df.drop(["label"],axis = 1).to_numpy().reshape(-1,IMG_SIZE,IMG_SIZE,ColorCh).astype(np.float32) / 255.0
y_test = test_df['label']

input_shape = X_train.shape[1:]

kernel_regularizer = regularizers.l2(0.0001)
final_activation = 'softmax'
entropy = 'sparse_categorical_crossentropy'

def vgg_block(num_convs, num_channels):
    block = tf.keras.models.Sequential()
    for _ in range(num_convs):
        block.add(Conv2D(filters=num_channels, kernel_size = (3,3), padding="same"))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        
    block.add(MaxPooling2D(pool_size=2, strides=2))
    return block

def VGG_MODEL(n_layers):
    supported_layers = [11, 13, 16, 19]
    
    if not n_layers in supported_layers:
        print('not supported')
        return
    
    if n_layers == 11:
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    if n_layers == 13:
        conv_arch = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
        
    if n_layers == 16:
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
    
    if n_layers == 19:
        conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
    
    
    vgg_model = Sequential()
    vgg_model.add(Input(shape=input_shape))
    
    for (num_convs, num_channels) in conv_arch:
        vgg_model.add(vgg_block(num_convs, num_channels))
    
    vgg_model.add(Flatten())
    
    vgg_model.add(Dense(units=4096))
    vgg_model.add(BatchNormalization())
    vgg_model.add(Activation('relu'))
    vgg_model.add(Dropout(0.5, seed=73))
    
    vgg_model.add(Dense(units=4096))
    vgg_model.add(BatchNormalization())
    vgg_model.add(Activation('relu'))
    vgg_model.add(Dropout(0.5, seed=73))
    
    vgg_model.add(Dense(units=n_classes, activation=final_activation))
    
    return vgg_model
  
def get_best_epoch(test_loss, history):
    for key, item in enumerate(history.history.items()):
        (name, arr) = item
        if name == 'val_loss':
            for i in range(len(arr)):
                if round(test_loss, 2) == round(arr[i], 2):
                    return i
        

def model_summary(model, history):
    print('---'*30)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    if history:
        index = get_best_epoch(test_loss, history)
        print('Best Epochs: ', index)

        train_accuracy = history.history['accuracy'][index]
        train_loss = history.history['loss'][index]

        print('Accuracy on train:',train_accuracy,'\tLoss on train:',train_loss)
        print('Accuracy on test:',test_accuracy,'\tLoss on test:',test_loss)
        print_graph('loss', index, history)
        print_graph('accuracy', index, history)
        print('---'*30)
EPOCHS = 50
patience = 5

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
batch_size = 16

if TPU_INIT:
    max_lr = max_lr * tpu_strategy.num_replicas_in_sync
    batch_size = batch_size * tpu_strategy.num_replicas_in_sync


rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

    
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ((logs.get('accuracy')>=0.999) | (logs.get('loss') <= 0.01)):
            print("\nLimits Reached cancelling training!")
            self.model.stop_training = True

            
            
end_callback = myCallback()

lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min')

lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)

early_stopping = EarlyStopping(patience = patience, monitor='val_loss',
                                 mode='min', restore_best_weights=True, 
                                 verbose = 1, min_delta = .00075)


checkpoint_filepath = 'Fish_Weights.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        verbose = 1,
                                        save_best_only=True)
log_dir="logs/fit/" + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
tensorboard_callback = TensorBoard(log_dir = log_dir, write_graph=True, histogram_freq=1)

callbacks = [end_callback, lr_callback, model_checkpoints, tensorboard_callback, early_stopping, lr_plat]
    
if TPU_INIT:
        callbacks = [end_callback, lr_callback, model_checkpoints, early_stopping, lr_plat]
def CompileModel(model):
    model.summary()
    model.compile(optimizer='adam', loss=entropy,metrics=['accuracy'])
    return model


def FitModel(model, name):

    history = model.fit(X_train,y_train,
                              epochs=EPOCHS,
                              callbacks=callbacks,
                              batch_size = batch_size,
                              validation_data = (X_test, y_test),
                              )
    
    model.load_weights(checkpoint_filepath)

    final_accuracy_avg = np.mean(history.history["val_accuracy"][-5:])

    final_loss = history.history["val_loss"][-1]
  
    group = {history: 'history', name: 'name', model: 'model', final_accuracy_avg:'acc', final_loss:'loss'}

    clear_output()
    model.summary()

    print('\n')
    print('---'*30)
    print(name,' Model')
    print('Total Epochs :', len(history.history['loss']))
    print('Accuracy on train:',history.history['accuracy'][-1],'\tLoss on train:',history.history['loss'][-1])
    print('Accuracy on val:', final_accuracy_avg ,'\tLoss on val:', final_loss)
    print('---'*30)

    return model, history
skip_training = False

mode = 'proMode'

def BuildModel():
    if TPU_INIT:
        with tpu_strategy.scope():
            prepared_model = VGG_MODEL(16)
            compiled_model = CompileModel(prepared_model)
    else:
        prepared_model = VGG_MODEL(16)
        compiled_model = CompileModel(prepared_model)

    return compiled_model

compiled_model = BuildModel()
model, history = FitModel(compiled_model, 'vgg')
model_summary(model, history)

model.save('fish_model.h5')
