import numpy as np
from keras.models import *
from keras.layers import *
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras import backend as keras
import keras
# segnet
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from matplotlib import pyplot as plt
import excel
import estimate
import cv2
import postprocessing
import os

import loss

class model():
    def __init__(self, model, name, size= (256,256,4)):
        self.heights= size[0]
        self.widths= size[1]
        self.channels= size[2]
        self.shape = size
        self.name = name
        self.model = model

    def train(self, x_train, y_train, epochs= 100, batch_size= 10, validation_ratio= 0.125):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # if x_train.ndim == 3:
        #     x_train = np.expand_dims(x_train, axis=-1)
        # if y_train.ndim == 3:
        #     y_train = np.expand_dims(y_train, axis=-1)

        history = self.model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[saveModel, checkBestPoint ,reduce_lr])
        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        # print(history.history.keys())
        # fig = plt.figure()
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')

        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='lower left')
        # fig.savefig('.\\result\performance\\' + self.name + '.png')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()

    def test(self, x_test, y_test, data_start, batch_size= 10, threshold= 0.5, save_path = None, land_thre = 0.5):
        def mkdir(path):
            # 去除首位空格
            path = path.strip()
            # 去除尾部 \ 符號
            path = path.rstrip("\\")

            # 判斷路徑是否存在
            # 存在     True
            # 不存在   False
            isExists = os.path.exists(path)

            # 判斷結果
            if not isExists:
                # 如果不存在則建立目錄
                print("Building the file.")
                # 建立目錄操作函式
                os.makedirs(path)
                return True
            else:
                # 如果目錄存在則不建立，並提示目錄已存在
                print("File is existing.")
                return False

        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)
        if save_path == None:
            save_path = self.name

        y_predict = self.model.predict(x_test, batch_size=batch_size)

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size=(self.heights, self.widths, 1),
                                                  threshold= threshold)
        print("Estimate.")
        iou = estimate.IOU(y_test, y_output, self.widths, len(y_test))
        (precision, recall, F1) = estimate.F1_estimate(y_test, y_output, self.widths, len(y_test))
        avr_iou = np.sum(iou) / len(y_test)
        avr_precision = np.sum(precision) / len(y_test)
        avr_recall = np.sum(recall) / len(y_test)
        avr_F1 = np.sum(F1) / len(y_test)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(y_output)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("e1", "precision", vertical=True)
        ex_iou.write_excel("e2", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        ex_iou.write_excel("g1", "recall", vertical=True)
        ex_iou.write_excel("g2", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        ex_iou.write_excel("i1", "F1", vertical=True)
        ex_iou.write_excel("i2", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

class Judgement_model(object):
    def __init__(self, model, name, input_shape= (256,256,4), classes = 2):
        self.heights= input_shape[0]
        self.widths= input_shape[1]
        self.channels= input_shape[2]
        self.shape = input_shape
        self.classes = classes
        self.name = name
        self.model = model

    def GT_data_transfer(self, excel_file, total_num, extract_index= "iou", threshold= 0.9):
        file = excel.Excel(file_path=excel_file)
        print("Excel file {} is opened.".format(excel_file))

        if extract_index == "iou" or extract_index == "IoU":
            index = file.read_excel(start="c3:c" + str(2 + total_num))
        elif extract_index == "precision" or extract_index == "Precision":
            index = file.read_excel(start="e2:e" + str(1 + total_num))
        elif extract_index == "recall" or extract_index == "Recall":
            index = file.read_excel(start="g2:g" + str(1 + total_num))
        elif extract_index == "f1" or extract_index == "F1":
            index = file.read_excel(start="i2:i" + str(1 + total_num))
        else:
            print("extrac_index：{}. ERROR".format(extract_index))
            raise ValueError
        file.close_excel()

        index = np.array(index, dtype=np.float32)

        print("Index is {}.".format(extract_index))
        print("Index shape {}".format(index.shape))
        print("Index dtype {}".format(index.dtype))
        low_result = np.ones(index.shape) * (index <= threshold)
        high_result = np.ones(index.shape) * (index > threshold)

        output_y = np.zeros((len(index), 2))
        output_y[:, 0] = high_result
        output_y[:, 1] = low_result

        return output_y

    def train(self, x, y, epochs= 100, batch_size= 10, validation_ratio= 0.125):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # history = self.model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[saveModel, checkBestPoint, callbacks_list[0], callbacks_list[1]])
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_ratio,
                                 callbacks=[saveModel, checkBestPoint, reduce_lr])

        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()

    def test(self, x, y, data_start, batch_size= 10, save_path = None):
        if save_path == None:
            save_path = self.name

        predict_y = self.model.predict(x, batch_size=batch_size)

        print("Postprocessing.")
        self.postprocessing(test_y= y, predict_y= predict_y, save_path= save_path, data_start= data_start)

    def postprocessing(self, test_y, predict_y, save_path, data_start, threshold = 0.5):
        print("Check the threshold.")
        output_y = np.ones(predict_y[:, 0].shape) * (predict_y[:, 0] > threshold)

        print("Estimate.")
        correct_y = np.ones(output_y.shape) * (output_y == test_y[:, 0])
        acc = np.sum(correct_y) / len(test_y)
        print("Average Acc：{}".format(acc))

        ex_acc = excel.Excel()
        ex_acc.write_excel("a1", save_path)
        ex_acc.write_excel("a2", "Order")
        for index in range(len(test_y)):
            ex_acc.write_excel("a" + str(index + 3), str(data_start + index))
        ex_acc.write_excel("b2", "Value of GT")
        ex_acc.write_excel("b3", test_y[:, 0], vertical= True)
        ex_acc.write_excel("d2", "Value of prediction")
        ex_acc.write_excel("d3", output_y, vertical=True)
        ex_acc.write_excel("f2", "Value of correct")
        ex_acc.write_excel("f3", correct_y, vertical=True)
        ex_acc.write_excel("h2", "Value of Acc")
        ex_acc.write_excel("h3", acc)

        ex_acc.save_excel(file_name=".\\result\data\\" + save_path + "_acc.xlsx")
        ex_acc.close_excel()
        print(correct_y.shape)
        print(output_y.shape)
        print(test_y[:, 0].shape)
        print(acc.shape)

class SI_mark_processing(object):
    def __init__(self, model, name, input_shape= (256,256,4), pipelines = 2):
        self.heights= input_shape[0]
        self.widths= input_shape[1]
        self.channels= input_shape[2]
        self.shape = input_shape
        self.pipelines = pipelines
        self.name = name
        self.pipe_model_struct = model[0]
        self.judgement_model_struct = model[1]

    def predict(self, x, y, judge_weights_path, pipe_weights_path, save_path, data_start,judge_threshold = 0.5):
        if len(pipe_weights_path) != self.pipelines:
            print("Pipelines num is different.")
            raise ValueError

        print("Judge processing.")
        dist_data = self.judge(x, judge_weights_path, threshold= judge_threshold)

        print("Finish judge processing.")

        print("Data distribution.")
        high_x = x
        high_y = y
        high_x = np.delete(high_x, np.where(dist_data == 0), axis=0)
        high_y = np.delete(high_y, np.where(dist_data == 0), axis=0)
        low_x = x
        low_y = y
        low_x = np.delete(low_x, np.where(dist_data == 1), axis=0)
        low_y = np.delete(low_y, np.where(dist_data == 1), axis=0)
        print("high_x shape:{}\thigh_y shape:{}".format(high_x.shape, high_y.shape))
        print("low_x shape:{}\tlow_y shape:{}".format(low_x.shape, low_y.shape))
        print("End data distribution.")

        # predict_y = np.zeros(y.shape)
        print("Mark processing.")
        print("Predict high data pipelines")
        self.pipe_model_struct.load_weights(pipe_weights_path[0])
        predict_high_y = self.pipe_model_struct.predict(high_x, batch_size=3)
        print("End predict high data pipelines")

        print("Predict low data pipelines")
        self.pipe_model_struct.load_weights(pipe_weights_path[1])
        predict_low_y = self.pipe_model_struct.predict(low_x, batch_size=3)
        print("End predict low data pipelines")

        high_order = dist_data == 1
        index_high = 0
        index_low = 0
        predict_y = []
        print("Concate predict data")
        for index in range(len(y)):
            if high_order[index]:
                predict_y.append(predict_high_y[index_high])
                # predict_y[index] = predict_high_y[index_high]
                index_high += 1
                # np.delete(high_y_predict, 0, axis=0)
            else:
                predict_y.append(predict_low_y[index_low])
                # predict_y[index] = predict_low_y[index_low]
                index_low += 1
                # np.delete(low_y_predict, 0, axis=0)
        predict_y = np.array(predict_y)
        # 針對兩種預測分別進行後處理
        high_save_path = save_path + "predictHighData"
        self.postprocessing(high_y, predict_high_y, high_save_path, data_start, threshold=0.5)
        low_save_path = save_path + "predictLowData"
        self.postprocessing(low_y, predict_low_y, low_save_path, data_start, threshold=0.5)

        # 針對最終融合進行後處理
        print("End concate predict data")
        print("Postprocessing")
        self.postprocessing(y, predict_y, save_path, data_start, threshold = 0.5)
        print("End postprocessing")
        print("End mark processing.")

        return predict_y

    def judge(self, x, judge_weights_path, batch_size = 10, threshold = 0.9):
        self.judgement_model_struct.load_weights(judge_weights_path)
        predict_y = self.judgement_model_struct.predict(x, batch_size=batch_size)
        output_y = np.ones(predict_y[:, 0].shape) * (predict_y[:, 0] > threshold)
        # output_y[0]：高於門檻值
        # output_y[1]：低於等於門檻值
        return output_y

    def postprocessing(self, test_y, predict_y, save_path, data_start, threshold = 0.5):
        def mkdir(path):
            # 去除首位空格
            path = path.strip()
            # 去除尾部 \ 符號
            path = path.rstrip("\\")

            # 判斷路徑是否存在
            # 存在     True
            # 不存在   False
            isExists = os.path.exists(path)

            # 判斷結果
            if not isExists:
                # 如果不存在則建立目錄
                print("Building the file.")
                # 建立目錄操作函式
                os.makedirs(path)
                return True
            else:
                # 如果目錄存在則不建立，並提示目錄已存在
                print("File is existing.")
                return False

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(predict_y,
                                                  size=(self.heights, self.widths, 1),
                                                  threshold= threshold)

        print("Estimate.")
        iou = estimate.IOU(test_y, y_output, self.widths, len(test_y))
        (precision, recall, F1) = estimate.F1_estimate(test_y, y_output, self.widths, len(test_y))
        avr_iou = np.sum(iou) / len(test_y)
        avr_precision = np.sum(precision) / len(test_y)
        avr_recall = np.sum(recall) / len(test_y)
        avr_F1 = np.sum(F1) / len(test_y)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(test_y)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("a2", "image num")
        for index in range(len(test_y)):
            ex_iou.write_excel("a" + str(index + 3), str(data_start + index))
        ex_iou.write_excel("f1", "precision", vertical=True)
        ex_iou.write_excel("f2", precision, vertical=True)
        ex_iou.write_excel("g1", "avr_precision", vertical=True)
        ex_iou.write_excel("g2", avr_precision, vertical=True)
        ex_iou.write_excel("h1", "recall", vertical=True)
        ex_iou.write_excel("h2", recall, vertical=True)
        ex_iou.write_excel("i1", "avr_recall", vertical=True)
        ex_iou.write_excel("i2", avr_recall, vertical=True)
        ex_iou.write_excel("j1", "F1", vertical=True)
        ex_iou.write_excel("j2", F1, vertical=True)
        ex_iou.write_excel("k1", "avr_F1", vertical=True)
        ex_iou.write_excel("k2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()


def Fusion_net(activation= 'sigmoid', size = (256, 256, 1)):
    # first input
    input_1 = Input(size)
    conv_11 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input_1)
    conv_12 = Conv2D(64, 3, activation='relu', padding='same')(conv_11)

    # second input
    input_2 = Input(size)
    conv_21 = Conv2D(64, 3, activation='relu', padding='same')(input_2)
    conv_22 = Conv2D(64, 3, activation='relu', padding='same')(conv_21)

    # fusion part
    con = concatenate([conv_12, conv_22])
    fusion_1 = Conv2D(64, 3, activation='relu', padding='same')(con)
    fusion_2 = Conv2D(64, 3, activation='relu', padding='same')(fusion_1)
    output = Conv2D(1, 1, activation=activation, padding= "same")(fusion_2)

    model = Model(inputs= [input_1, input_2], outputs= output)
    return model

def Fusion_net_twoActivation(size = (256, 256, 1)):
    # first input
    input_1 = Input(size)
    conv_11 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input_1)
    conv_12 = Conv2D(64, 3, activation='relu', padding='same')(conv_11)

    # second input
    input_2 = Input(size)
    conv_21 = Conv2D(64, 3, activation='relu', padding='same')(input_2)
    conv_22 = Conv2D(64, 3, activation='relu', padding='same')(conv_21)

    # fusion part
    con = concatenate([conv_12, conv_22])
    fusion_1 = Conv2D(64, 3, activation='relu', padding='same')(con)
    fusion_2 = Conv2D(64, 3, activation='relu', padding='same')(fusion_1)
    output = Conv2D(1, 1, activation='relu', padding= "same")(fusion_2)
    output = Conv2D(1, 1, activation='sigmoid', padding="same")(output)

    model = Model(inputs= [input_1, input_2], outputs= output)
    return model

def Unet(size= ( 256, 256, 4)):
    input = Input(size)
    conv1 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2,2), None, 'same')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D((2,2), None, 'same')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2,2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv9)

    model = Model(input = input, output = conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def Unet_elu(size= (256, 256, 5)):
    input = Input(size)
    print("input.shape : ",input.shape)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input)
    conv1 = ELU(alpha=1.0)(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = ELU(alpha=1.0)(conv1)
    pool1 = MaxPooling2D((2,2), None, 'same')(conv1)

    print("pool1.shape : ",pool1.shape)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = ELU(alpha=1.0)(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = ELU(alpha=1.0)(conv2)
    pool2 = MaxPooling2D((2,2), None, 'same')(conv2)

    print("pool2.shape : ",pool2.shape)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = ELU(alpha=1.0)(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = ELU(alpha=1.0)(conv3)
    pool3 = MaxPooling2D((2,2), None, 'same')(conv3)

    print("pool3.shape : ",pool3.shape)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = ELU(alpha=1.0)(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = ELU(alpha=1.0)(conv4)
    drop4 = Dropout(0.5)(conv4)
    print("drop4.shape : ",drop4.shape)
    pool4 = MaxPooling2D((2,2), None, 'same')(drop4)
    print("pool4.shape : ",pool4.shape)

    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = ELU(alpha=1.0)(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = ELU(alpha=1.0)(conv5)
    drop5 = Dropout(0.5)(conv5)
    print("drop5.shape : ",drop5.shape)

    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = ELU(alpha=1.0)(up6)
    print("up6.shape : ",up6.shape)
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = ELU(alpha=1.0)(conv6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = ELU(alpha=1.0)(conv6)

    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = ELU(alpha=1.0)(up7)
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = ELU(alpha=1.0)(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = ELU(alpha=1.0)(conv7)

    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = ELU(alpha=1.0)(up8)
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = ELU(alpha=1.0)(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = ELU(alpha=1.0)(conv8)

    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = ELU(alpha=1.0)(up9)
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv9 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    return model

def segnet(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss.focal_loss, metrics = ['accuracy'])

    return model

def segnet_dense_inception(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    def dense_block(x, filters, kernel):
        conv1 = Convolution2D(filters, (kernel, kernel), padding='same')(x)
        conv2 = Convolution2D(filters, (kernel, kernel), padding='same')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    def Inception_model(input_layer, filters):
        tower_1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(tower_3x3)
        tower_5x5 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(tower_5x5)
        tower_max3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        tower_max3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(tower_max3x3)

        output = keras.layers.concatenate([tower_1x1, tower_3x3, tower_5x5, tower_max3x3], axis=3)
        return output

    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = dense_block(conv_1, 64, kernel)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = dense_block(pool_1, 64, kernel)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = dense_block(conv_4, 128, kernel)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = dense_block(conv_5, 256, kernel)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = dense_block(conv_5, 256, kernel)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = dense_block(conv_6, 256, kernel)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = dense_block(conv_8, 512, kernel)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = dense_block(conv_8, 512, kernel)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = dense_block(conv_9, 512, kernel)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = dense_block(pool_4, 512, kernel)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = dense_block(conv_11, 512, kernel)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = dense_block(conv_12, 512, kernel)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder
    pool_5 = Inception_model(pool_5, 128)

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = dense_block(unpool_1, 512, kernel)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = dense_block(conv_14, 512, kernel)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = dense_block(conv_15, 512, kernel)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = dense_block(unpool_2, 512, kernel)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = dense_block(conv_17, 512, kernel)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = dense_block(conv_19, 256, kernel)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = dense_block(unpool_3, 256, kernel)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = dense_block(conv_20, 256, kernel)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = dense_block(conv_22, 128, kernel)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = dense_block(unpool_4, 128, kernel)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = dense_block(conv_24, 64, kernel)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = dense_block(unpool_5, 64, kernel)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss.focal_loss, metrics = ['accuracy'])

    return model

# 雙網路 model_1, model_2 需要給要訓練的兩種網路
def two_Network(model_1, model_2, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]

    input_layer = Input(size)
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(2, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input_layer, outputs=model_connect)
    return model

def three_Network(model_1, model_2, model_3, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1] + input[:,:,:,2]

    input_layer = Input(size)
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    out3 = model_3(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2, out3])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(3, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input_layer, outputs=model_connect)
    return model

def four_Network(model_1, model_2, model_3, model_4, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1] + input[:,:,:,2] + input[:,:,:,3]

    input_layer = Input(size)
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    out3 = model_3(input_layer)
    out4 = model_4(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2, out3, out4])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(4, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input_layer, outputs=model_connect)
    return model

def two_unet(size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]

    input = Input(size)
    model1 = Unet(size)
    model2 = Unet(size)

    out1 = model1(input)
    out2 = model2(input)
    con = concatenate([out1, out2])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(2, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input, outputs=[model_connect, out1, out2])
    return model

def two_dense_unet(size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]
    input = Input(size)
    model1 = ResNet(size)
    model2 = ResNet(size)

    out1 = model1(input)
    out2 = model2(input)
    con = concatenate([out1, out2])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(2, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input, outputs=[model_connect, out1, out2])
    return model

def one_dense_one_unet(size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]
    input = Input(size)
    model1 = ResNet(size)
    model2 = Unet(size)

    out1 = model1(input)
    out2 = model2(input)
    con = concatenate([out1, out2])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(2, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input, outputs=[model_connect, out1, out2])
    return model

def RIC_Unet(size=(256,256, 5)):
    def RES_module(x, filters, kernel):
        conv1 = Convolution2D(filters, (kernel, kernel), padding='same')(x)
        conv2 = Convolution2D(filters, (kernel, kernel), padding='same')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    def DRI_module(x, f1):
        con1_1 = BatchNormalization()(x)
        con1_1 = PReLU()(con1_1)
        con1_1 = Convolution2D(f1/4, (1, 1), padding="same")(con1_1)

        con3_1 = BatchNormalization()(x)
        con3_1 = PReLU()(con3_1)
        con3_1 = Convolution2D(f1 / 4, (1, 1), padding="same")(con3_1)

        con3_3 = BatchNormalization()(x)
        con3_3 = PReLU()(con3_3)
        con3_3 = Convolution2D(f1 / 4, (1, 1), padding="same")(con3_3)
        con3_3 = BatchNormalization()(con3_3)
        con3_3 = PReLU()(con3_3)
        con3_3 = Convolution2D(f1 / 4, (3, 3), padding="same")(con3_3)
        con3_3 = keras.layers.Add()([con3_1, con3_3])

        con5_1 = BatchNormalization()(x)
        con5_1 = PReLU()(con5_1)
        con5_1 = Convolution2D(f1 / 4, (1, 1), padding="same")(con5_1)

        con5_5 = BatchNormalization()(x)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (1, 1), padding="same")(con5_5)
        con5_5 = BatchNormalization()(con5_5)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (5, 5), padding="same")(con5_5)
        con5_5 = BatchNormalization()(con5_5)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (3, 3), padding="same")(con5_5)
        con5_5 = keras.layers.Add()([con5_1, con5_5])

        max3_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        max3_3 = BatchNormalization()(max3_3)
        max3_3 = PReLU()(max3_3)
        max3_3 = Convolution2D(f1 / 4, (1, 1), padding='same')(max3_3)

        output = concatenate([con1_1, con3_3, con5_5, max3_3])
        return output

    def CAB_module(x, f):
        scale = GlobalAveragePooling2D()(x)
        scale = Dense(f, activation="relu")(scale)
        scale = Dense(f, activation="sigmoid")(scale)
        output = multiply([x, scale])
        return output


    input = Input(size)

    ri1 = Convolution2D(64, (3, 3), padding="same")(input)
    ri1 = DRI_module(ri1, 64)
    ri1 = Activation("relu")(ri1)
    ri1 = Convolution2D(128, (3, 3), strides=2, padding="same")(ri1)
    ri1 = RES_module(ri1, 128, 3)
    ri1 = Activation("relu")(ri1)

    ri2 = Convolution2D(128, (3, 3), padding="same")(ri1)
    ri2 = DRI_module(ri2, 128)
    ri2 = Activation("relu")(ri2)
    ri2 = Convolution2D(256, (3, 3), strides=2, padding="same")(ri2)
    ri2 = RES_module(ri2, 256, 3)
    ri2 = Activation("relu")(ri2)

    ri3 = Convolution2D(256, (3, 3), padding="same")(ri2)
    ri3 = DRI_module(ri3, 256)
    ri3 = Activation("relu")(ri3)
    ri3 = Convolution2D(512, (3, 3), strides=2, padding="same")(ri3)
    ri3 = RES_module(ri3, 512, 3)
    ri3 = Activation("relu")(ri3)

    ri4 = Convolution2D(512, (3, 3), padding="same")(ri3)
    ri4 = DRI_module(ri4, 512)
    ri4 = Activation("relu")(ri4)
    ri4 = Convolution2D(1024, (3, 3), strides=2, padding="same")(ri4)
    ri4 = RES_module(ri4, 1024, 3)
    ri4 = Activation("relu")(ri4)

    feature = Convolution2D(1024, (3, 3), padding="same")(ri4)
    feature = DRI_module(feature, 1024)
    feature = Activation("relu")(feature)

    dc4 = concatenate([ri4, feature])
    dc4 = Conv2DTranspose(512, (3, 3), padding="same")(dc4)
    dc4 = Activation("relu")(dc4)
    dc4 = CAB_module(dc4, 512)
    dc4 = Convolution2D(512, (3, 3), padding="same")(dc4)

    dc3 = concatenate([ri3, dc4])
    dc3 = Conv2DTranspose(512, (3, 3), padding="same")(dc3)
    dc3 = Activation("relu")(dc3)
    dc3 = CAB_module(dc3, 512)
    dc3 = Convolution2D(512, (3, 3), padding="same")(dc3)

    dc2 = concatenate([ri2, dc3])
    dc2 = Conv2DTranspose(256, (3, 3), padding="same")(dc2)
    dc2 = Activation("relu")(dc2)
    dc2 = CAB_module(dc2, 256)
    dc2 = Convolution2D(256, (3, 3), padding="same")(dc2)

    dc1 = concatenate([ri1, dc2])
    dc1 = Conv2DTranspose(128, (3, 3), padding="same")(dc1)
    dc1 = Activation("relu")(dc1)
    dc1 = CAB_module(dc1, 128)
    dc1 = Convolution2D(128, (3, 3), padding="same")(dc1)

    dc1 = Convolution2D(1, (3, 3), padding="same")(dc1)
    dc1 = BatchNormalization()(dc1)
    dc1 = Activation("sigmoid")(dc1)

    model = Model(inputs=input, outputs=dc1)
    return model

"""-------------------Self Model---------------------------------------"""
# self definition
def residual_UNet(input_size=(256, 256, 5)):
    def dense_block(x, filters, kernel, activation='relu'):
        conv1 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(x)
        conv2 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = dense_block(conv2, 128, 3)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = dense_block(conv3, 256, 3)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = dense_block(conv4, 512, 3)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv4 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = dense_block(conv4, 1024, 3)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = dense_block(conv6, 512, 3)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = dense_block(conv7, 256, 3)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = dense_block(conv8, 128, 3)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = dense_block(conv9, 64, 3)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

# self definition
def proposal_UNet_layer1_layerBlock(block, input_size=(256, 256, 5), classes=1, n_layers_per_block = 4, block_layers = 5):
    print("block layer num = {}".format(block_layers))

    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    if block_layers >= 5:
        layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    if block_layers >= 4:
        layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    if block_layers >= 3:
        layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    if block_layers >= 2:
        layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    if block_layers >= 1:
        layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    if block_layers >= 2:
        layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    if block_layers >= 3:
        layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    if block_layers >= 4:
        layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    if block_layers >= 5:
        layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def UNet_DtoU5(block, name, input_size=(256, 256, 5), classes=1, n_layers_per_block = 8, block_num = 1):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-1")
    for i in range(block_num-1):
        layer1 = block(layer1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-" + str(i + 2))
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature2-1")
    for i in range(block_num-1):
        layer2 = block(layer2, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature2-" + str(i + 2))
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-1")
    for i in range(block_num-1):
        layer3 = block(layer3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-" + str(i + 2))
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature4-1")
    for i in range(block_num-1):
        layer4 = block(layer4, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature4-" + str(i + 2))
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-1")
    for i in range(block_num-1):
        layer5 = block(layer5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-" + str(i + 2))
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature6-1")
    for i in range(block_num-1):
        layer6 = block(layer6, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature6-" + str(i +2))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature7-1")
    for i in range(block_num-1):
        layer7 = block(layer7, 256, 3, n_layers_per_block=n_layers_per_block, name= name+"feature7-" + str(i +2))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature8-1")
    for i in range(block_num-1):
        layer8 = block(layer8, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature8-" + str(i +2))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature9-1")
    for i in range(block_num-1):
        layer9 = block(layer9, 64, 3, n_layers_per_block=n_layers_per_block, name= name+"feature9-" + str(i +2))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_layer1(block, input_size=(256, 256, 5), classes=1, n_layers_per_block = 4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_layer2(block, input_size=(256, 256, 5), classes=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)
    dense6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)
    dense7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)
    dense8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = block(conv6, 64, 2, n_layers_per_block=n_layers_per_block)
    dense9 = block(conv9, 64, 2, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_TDTU_layer1(block, input_size=(256, 256, 5), classes=1, TD=1, TU=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    if TD:
        pool1 = TransitionDown(conv1, 64)
    else:
        pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)


    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool2 = TransitionDown(dense2, 128)
    else:
        pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool3 = TransitionDown(dense3, 256)
    else:
        pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    if TD:
        pool4 = TransitionDown(drop4, 512)
    else:
        pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    if TU:
        merge6 = TransitionUp(drop4, drop5, 512)
    else:
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge7 = TransitionUp(dense3, dense6, 256)
    else:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense6))
        merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge8 = TransitionUp(dense2, dense7, 128)
    else:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense7))
        merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge9 = TransitionUp(conv1, dense8, 64)
    else:
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense8))
        merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_TDTU_layer2(block, input_size=(256, 256, 5), classes=1, TD=1, TU=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    if TD:
        pool1 = TransitionDown(conv1, 64)
    else:
        pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool2 = TransitionDown(dense2, 128)
    else:
        pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool3 = TransitionDown(dense3, 256)
    else:
        pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    if TD:
        pool4 = TransitionDown(drop4, 512)
    else:
        pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    if TU:
        merge6 = TransitionUp(drop4, drop5, 512)
    else:
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)
    dense6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge7 = TransitionUp(dense3, dense6, 256)
    else:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense6))
        merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)
    dense7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge8 = TransitionUp(dense2, dense7, 128)
    else:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense7))
        merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)
    dense8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge9 = TransitionUp(conv1, dense8, 64)
    else:
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense8))
        merge9 = concatenate([conv1, up9])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = block(conv6, 64, 2, n_layers_per_block=n_layers_per_block)
    dense9 = block(conv9, 64, 2, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

"""-------------------Block---------------------------------------"""

def residual_block(x, filters, kernel, n_layers_per_block = 0, activation='relu'):
    conv1 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(x)
    conv2 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)
    output = keras.layers.Add()([x, conv2])
    return output

def dense_block(x, filters, kernel, n_layers_per_block = 4, dropout_p = 0.2, growth_rate = 16):
    stack = x
    n_filters = filters

    for j in range(n_layers_per_block):
        l = BN_ReLU_Conv(stack, growth_rate, filter_size= kernel, dropout_p=dropout_p)
        stack = concatenate([stack, l])
        n_filters += growth_rate

    # skip_connection = stack
    # stack = TransitionDown(stack, filters, dropout_p)
    return stack


# https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras/blob/master/main.py
def RDBlocks(x, filters, kernel, name, n_layers_per_block=4, g=16):
    li = [x]
    pas = Convolution2D(filters=g, kernel_size=(kernel, kernel), strides=(1, 1), padding='same', activation='relu')(x)

    for i in range(n_layers_per_block - 1):
        li.append(pas)
        out = Concatenate()(li)  # conctenated out put
        pas = Convolution2D(filters=g, kernel_size=(kernel, kernel), strides=(1, 1), padding='same', activation='relu')(out)

    li.append(pas)
    out = Concatenate()(li)
    feat = Convolution2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(out)

    feat = Add(name= name)([feat, x])
    return feat

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l
