import os
import random
import timeit
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import numpy as np 
import tensorflow.keras
import tensorflow as tf
from keras_unet_collection import models
from Unet import *


class DataGen(tensorflow.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=(256, 256)):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def load(self, name):
        image_path = os.path.join(self.path, "image/", name)
        mask_path = os.path.join(self.path, "label/", name)
        #print(name, image_path)
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
        except:
            print('wrong',image_path)
        
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)
        

        image = image / 255.0
        mask = mask / 255.0
        return image, mask
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            files = self.ids[index * self.batch_size:]
        else:
            files = self.ids[index * self.batch_size:(index + 1) * self.batch_size]

        print(files)
        # print(len(files))
        images = []
        masks  = []
        for name in files:
            image, mask = self.load(name)
            images.append(image)
            masks.append(mask)
            
        images = np.array(images)
        masks  = np.array(masks)
        return images, masks
    
    def on_epoch_end(self):
        pass

def check_image(train_ids, train_path, batch_size, image_size):
    gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
    x, y = gen.__getitem__(0)
    print(x.shape, y.shape)
    r = random.randint(0, len(x) - 1)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(x[r])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[r] * 255, image_size), cmap='gray')
    plt.show()

def postproccessing(image):
    kernel = np.ones((5,5), np.uint8)

    # opening
    erosion1 = cv2.erode(image, kernel, iterations = 1)
    dilation1 = cv2.dilate(erosion1, kernel, iterations = 1)

    # closing
    dilation2 = cv2.dilate(dilation1, kernel, iterations = 1)
    erosion2 = cv2.erode(dilation2, kernel, iterations = 1)

    return erosion2

def calculate_jaccard_index(folder_name):
    index = os.listdir('results/' + folder_name + '/image/')
    iou = []
    for i in index:
        img_true = cv2.imread('./results/' + folder_name + '/label/' + i, 0)
        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        img_pred = cv2.imread('./results/' + folder_name + '/results/' + i, 0)
        img_pred[img_pred < 128] = 0
        img_pred[img_pred >= 128] = 1
        img_true = np.array(img_true).ravel()
        img_pred = np.array(img_pred).ravel()
        ji = jaccard_score(img_true, img_pred)
        iou.append(ji)

        if i == '425.png':
            print(folder_name + ': ' + str(ji))

    return iou

'''save the model train acc loss fig.'''
def save_figure(history, feature):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coefficient')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('figures/' + feature + '_dice.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('figures/' + feature + '_loss.png')
    plt.clf()

'''test image and save result'''
def save_results(model, image_size, folder_name):
    if not os.path.exists('results/' + folder_name):
        os.mkdir('./results/' + folder_name)
        os.mkdir('./results/' + folder_name + '/image')
        os.mkdir('./results/' + folder_name + '/label')
        os.mkdir('./results/' + folder_name + '/results')

    inference_time = []
    test_ids = os.listdir('c:/Users/CCC/Desktop/tumour_project/model/data/image/')
    for i in range(len(test_ids)):
        x = cv2.imread('c:/Users/CCC/Desktop/tumour_project/model/test/image/' + test_ids[i])
        cv2.imwrite('./results/' + folder_name + '/image/' + test_ids[i], x)
        y = cv2.imread('c:/Users/CCC/Desktop/tumour_project/model/test/label/' + test_ids[i])
        cv2.imwrite('./results/' + folder_name + '/label/' + test_ids[i], y)
        size = np.shape(x)

        x = cv2.resize(x, (256, 256))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        start = timeit.default_timer()
        results = model.predict(x)
        results = results >= 0.5
        stop = timeit.default_timer()
        inference_time.append(round(stop - start, 2))
        results = np.reshape(results * 255, image_size)
        results = np.stack((results,) * 3, -1)
        results = results.astype(np.uint8)
        results = cv2.cvtColor(results, cv2.COLOR_BGR2GRAY)
        results = postproccessing(results)
        results = cv2.resize(results, (size[1], size[0]))
        cv2.imwrite('./results/' + folder_name + '/results/' + test_ids[i], results)

    return round(sum(inference_time) / len(inference_time), 2)

def train(model, feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps):
    reduce_lr_loss = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=7, verbose=1, epsilon=1e-4)
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath='models/' + feature + '.h5', monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True)

    start = timeit.default_timer()
    history = model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=[checkpoint, reduce_lr_loss])
    stop = timeit.default_timer()
    train_time = round(stop - start, 2)
    save_figure(history, feature)

    model.load_weights('models/' + feature + '.h5')
    inference_time = save_results(model, image_size, feature)
    iou = calculate_jaccard_index(feature)
    return train_time, inference_time, iou

'''train four model'''
def run(image_size, batch_size, epochs, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count):
    feature = '_' + str(image_size) + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(filters[0]) + '_' + str(count)

    myBCDUNet = BCDUNet(image_size, learning_rate)
    model1 = myBCDUNet.build_model(filters)
    #model1.summary()
    training_time1, inference_time1, iou1 = train(model1, 'BCDUNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    myFRUNet = FRUNet(image_size, learning_rate)
    model2 = myFRUNet.build_model(filters)
    #model2.summary()
    training_time2, inference_time2, iou2 = train(model2, 'FRUNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    myUNetPlusPlus = UNetPlusPlus(image_size, learning_rate)
    model3 = myUNetPlusPlus.build_model(filters)
    #model3.summary()
    training_time3, inference_time3, iou3 = train(model3, 'UNet++' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    myUNet = UNet(image_size, learning_rate)
    model4 = myUNet.build_model(filters)
    #model4.summary()
    training_time4, inference_time4, iou4 = train(model4, 'UNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    '''
    f = open('summary.txt', 'a')
    lines = ['----------Summary' + feature + '----------\n',
             'Accuracy(Jaccard index)\n',
             'BCDUNet: ' + str(round(sum(iou1) / len(iou1), 4)) + ' +- ' + str(round(np.var(iou1), 4)) + '\n',
             'FRUNet: '  + str(round(sum(iou2) / len(iou2), 4)) + ' +- ' + str(round(np.var(iou2), 4)) + '\n',
             'UNet++: '  + str(round(sum(iou3) / len(iou3), 4)) + ' +- ' + str(round(np.var(iou3), 4)) + '\n',
             'UNet: '    + str(round(sum(iou4) / len(iou4), 4)) + ' +- ' + str(round(np.var(iou4), 4)) + '\n\n',
             'Training Time\n',
             'BCDUNet: ' + str(training_time1) + '\n',
             'FRUNet: '  + str(training_time2) + '\n',
             'UNet++: '  + str(training_time3) + '\n',
             'UNet: '    + str(training_time4) + '\n\n',
             'Inference Time\n',
             'BCDUNet: ' + str(inference_time1) + '\n',
             'FRUNet: '  + str(inference_time2) + '\n',
             'UNet++: '  + str(inference_time3) + '\n',
             'UNet: '    + str(inference_time4) + '\n',
             '------------------------------------------\n\n']
    f.writelines(lines)
    f.close()
    '''

if __name__ == '__main__':
    seed = 55688
    random.seed = seed
    np.random.seed = seed
    tf.seed = seed

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('figures'):
        os.mkdir('figures')

    dataset_path = 'C:/Users/Jeff/Desktop/tumour_project/model/data/'
    train_path = 'C:/Users/Jeff/Desktop/tumour_project/model/data/train/'
    valid_path = 'C:/Users/Jeff/Desktop/tumour_project/model/data/valid/'
    train_ids = os.listdir(train_path + 'image/')
    valid_ids = os.listdir(valid_path + 'image/')

    # #image_size = (256, 512)
    # image_size = (256, 256)
    # epoch = 50
    # filters = [16, 32, 64, 128, 256]
    # learning_rate = 0.0001

    '''check if image is ok.'''
    # dataset_path = 'C:/Users/CCC/Desktop/tumour_project/model/data/'
    # train_path = dataset_path
    # train_ids = os.listdir(train_path + 'image/')
    
    batch_size = 8
    image_size = (256, 256)
    epoch = 5
    filters = [16, 32, 64, 128, 256]
    learning_rate = 0.0001
    val_data_size = 67
    valid_ids = train_ids[:val_data_size]
    train_ids = train_ids[val_data_size:]
    check_image(train_ids, train_path, batch_size, image_size)



    '''train model with image augmentation and different batch size.'''
    batch_size = 8
    check_image(train_ids, train_path, batch_size, image_size)
    time = 5
    count = 3
    while count < time:
        # batch_size = 2
        # train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
        # train_steps = len(train_ids)//batch_size
        # valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
        # valid_steps = len(valid_ids)//batch_size
        # run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

        # batch_size = 4
        # train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
        # train_steps = len(train_ids)//batch_size
        # valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
        # valid_steps = len(valid_ids)//batch_size
        # run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

        batch_size = 8
        train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
        train_steps = len(train_ids)//batch_size
        valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
        valid_steps = len(valid_ids)//batch_size
        run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

        count += 1