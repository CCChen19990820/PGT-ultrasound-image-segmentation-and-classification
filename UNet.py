import numpy as np
import tensorflow as tf
import tensorflow.keras

class UNet:
    def __init__(self, image_size, learning_rate):
        self.input_size = (image_size[0], image_size[1], 3)
        self.learning_rate = learning_rate

    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = tensorflow.keras.backend.flatten(y_true)
        y_pred_f = tensorflow.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def standard_unit(self, inputs, filters):
        conv1 = tensorflow.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        bn1 = tensorflow.keras.layers.BatchNormalization()(conv1)
        conv2 = tensorflow.keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
        bn2 = tensorflow.keras.layers.BatchNormalization()(conv2)
        return bn2

    def upsampling_block(self, inputs, filters):
        upsampling = tensorflow.keras.layers.UpSampling2D((2, 2))(inputs)
        conv = tensorflow.keras.layers.Conv2D(filters, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(upsampling)
        bn = tensorflow.keras.layers.BatchNormalization()(conv)
        return bn

    def build_model(self, filters):
        inputs = tensorflow.keras.layers.Input(self.input_size)

        # encode
        conv1 = self.standard_unit(inputs, filters[0])
        pool1 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pool1, filters[1])
        pool2 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pool2, filters[2])
        pool3 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pool3, filters[3])
        pool4 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        conv5 = self.standard_unit(pool4, filters[4])

        # decode
        up1 = self.upsampling_block(conv5, filters[3])
        concat1 = tensorflow.keras.layers.Concatenate()([conv4, up1])
        conv6 = self.standard_unit(concat1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        concat2 = tensorflow.keras.layers.Concatenate()([conv3, up2])
        conv7 = self.standard_unit(concat2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        concat3 = tensorflow.keras.layers.Concatenate()([conv2, up3])
        conv8 = self.standard_unit(concat3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        concat4 = tensorflow.keras.layers.Concatenate()([conv1, up4])
        conv9 = self.standard_unit(concat4, filters[0])

        outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)
        model = tensorflow.keras.models.Model(inputs, outputs)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        return model

class UNetPlusPlus(UNet):
    def build_model(self, filters):
        inputs = tensorflow.keras.layers.Input(self.input_size)
        # encode
        conv1_1 = self.standard_unit(inputs, filters[0])
        pool1 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)
        conv2_1 = self.standard_unit(pool1, filters[1])
        pool2 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)
        conv3_1 = self.standard_unit(pool2, filters[2])
        pool3 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)
        conv4_1 = self.standard_unit(pool3, filters[3])
        pool4 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)
        conv5_1 = self.standard_unit(pool4, filters[4])

        # skip connection
        up1_2 = self.upsampling_block(conv2_1, filters[1])
        concat1 = tensorflow.keras.layers.Concatenate()([up1_2, conv1_1])
        conv1_2 = self.standard_unit(concat1, filters[0])
        up2_2 = self.upsampling_block(conv3_1, filters[2])
        concat2 = tensorflow.keras.layers.Concatenate()([up2_2, conv2_1])
        conv2_2 = self.standard_unit(concat2, filters[1])
        up1_3 = self.upsampling_block(conv2_2, filters[1])
        concat3 = tensorflow.keras.layers.Concatenate()([up1_3, conv1_1, conv1_2])
        conv1_3 = self.standard_unit(concat3, filters[0])
        up3_2 = self.upsampling_block(conv4_1, filters[3])
        concat4 = tensorflow.keras.layers.Concatenate()([up3_2, conv3_1])
        conv3_2 = self.standard_unit(concat4, filters[2])
        up2_3 = self.upsampling_block(conv3_2, filters[2])
        concat5 = tensorflow.keras.layers.Concatenate()([up2_3, conv2_1, conv2_2])
        conv2_3 = self.standard_unit(concat5, filters[1])
        up1_4 = self.upsampling_block(conv2_3, filters[1])
        concat6 = tensorflow.keras.layers.Concatenate()([up1_4, conv1_1, conv1_2, conv1_3])
        conv1_4 = self.standard_unit(concat6, filters[0])

        # decode
        up4_2 = self.upsampling_block(conv5_1, filters[3])
        concat7 = tensorflow.keras.layers.Concatenate()([up4_2, conv4_1])
        conv4_2 = self.standard_unit(concat7, filters[3])
        up3_3 = self.upsampling_block(conv4_2, filters[2])
        concat8 = tensorflow.keras.layers.Concatenate()([up3_3, conv3_1, conv3_2])
        conv3_3 = self.standard_unit(concat8, filters[2])
        up2_4 = self.upsampling_block(conv3_3, filters[1])
        concat9 = tensorflow.keras.layers.Concatenate()([up2_4, conv2_1, conv2_2, conv2_3])
        conv2_4 = self.standard_unit(concat9, filters[1])
        up1_5 = self.upsampling_block(conv2_4, filters[0])
        concat10 = tensorflow.keras.layers.Concatenate()([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
        conv1_5 = self.standard_unit(concat10, filters[0])

        # deep supervision
        '''
        output1 = tensorflow.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(conv1_2)
        output2 = tensorflow.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(conv1_3)
        output3 = tensorflow.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(conv1_4)
        '''

        output4 = tensorflow.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(conv1_5)
        model = tensorflow.keras.models.Model(inputs, output4)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        return model

class FRUNet(UNet):
    def residual_block(self, inputs, filters, strides=1, is_first=False):
        # feature extraction
        if not is_first:
            bn1 = tensorflow.keras.layers.BatchNormalization()(inputs)
            relu1 = tensorflow.keras.layers.Activation("relu")(bn1)
            conv1 = tensorflow.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(relu1)
        else:
            conv1 = tensorflow.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(inputs)
        bn2 = tensorflow.keras.layers.BatchNormalization()(conv1)
        relu2 = tensorflow.keras.layers.Activation("relu")(bn2)
        conv2 = tensorflow.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1)(relu2)

        # shortcut
        shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(inputs)
        bn3 = tensorflow.keras.layers.BatchNormalization()(shortcut)

        # addition
        addition = tensorflow.keras.layers.Add()([conv2, bn3])
        return addition

    def build_model(self, filters):
        inputs = tensorflow.keras.layers.Input(self.input_size)

        # encode
        residual1 = self.residual_block(inputs, filters[0], 1, True)
        residual2 = self.residual_block(residual1, filters[1], 2)
        residual3 = self.residual_block(residual2, filters[2], 2)
        residual4 = self.residual_block(residual3, filters[3], 2)
        residual5 = self.residual_block(residual4, filters[4], 2)

        # decode
        up1 = self.upsampling_block(residual5, filters[3])
        concat1 = tensorflow.keras.layers.Concatenate()([up1, residual4])
        residual6 = self.residual_block(concat1, filters[3])
        up2 = self.upsampling_block(residual6, filters[2])
        concat2 = tensorflow.keras.layers.Concatenate()([up2, residual3])
        residual7 = self.residual_block(concat2, filters[2])
        up3 = self.upsampling_block(residual7, filters[1])
        concat3 = tensorflow.keras.layers.Concatenate()([up3, residual2])
        residual8 = self.residual_block(concat3, filters[1])
        up4 = self.upsampling_block(residual8, filters[0])
        concat4 = tensorflow.keras.layers.Concatenate()([up4, residual1])
        residual9 = self.residual_block(concat4, filters[0])
        
        outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(residual9)
        model = tensorflow.keras.models.Model(inputs, outputs)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        return model

class BCDUNet(UNet):
    def BConvLSTM(self, in1, in2, d, fi, fo):
        x1 = tensorflow.keras.layers.Reshape(target_shape=(1, np.int32(self.input_size[0]/d), np.int32(self.input_size[1]/d), fi))(in1)
        x2 = tensorflow.keras.layers.Reshape(target_shape=(1, np.int32(self.input_size[0]/d), np.int32(self.input_size[1]/d), fi))(in2)
        merge = tensorflow.keras.layers.concatenate([x1,x2], axis=1) 
        merge = tensorflow.keras.layers.ConvLSTM2D(fo, (3, 3), padding='same', return_sequences=False, go_backwards=True,kernel_initializer='he_normal')(merge)
        return merge

    def build_model(self, filters):
        inputs = tensorflow.keras.layers.Input(self.input_size)

        # encode
        conv1 = self.standard_unit(inputs, filters[0])
        pool1 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pool1, filters[1])
        pool2 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pool2, filters[2])
        pool3 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pool3, filters[3])
        pool4 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        # D1
        conv5_1 = self.standard_unit(pool4, filters[4])
        # D2
        conv5_2 = self.standard_unit(conv5_1, filters[4])
        # D3
        merge_dense = tensorflow.keras.layers.concatenate([conv5_2, conv5_1], axis=3)
        conv5_3 = self.standard_unit(merge_dense, filters[4])
        
        # decode
        up1 = self.upsampling_block(conv5_3, filters[3])
        LSTM1 = self.BConvLSTM(conv4, up1, 8, filters[3], filters[2])
        conv6 = self.standard_unit(LSTM1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        LSTM2 = self.BConvLSTM(conv3, up2, 4, filters[2], filters[1])
        conv7 = self.standard_unit(LSTM2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        LSTM3 = self.BConvLSTM(conv2, up3, 2, filters[1], filters[0])
        conv8 = self.standard_unit(LSTM3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        LSTM4 = self.BConvLSTM(conv1, up4, 1, filters[0], int(filters[0]/2))
        conv9 = self.standard_unit(LSTM4, filters[0])  

        outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv9)
        model = tensorflow.keras.models.Model(inputs, outputs)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        return model