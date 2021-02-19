import numpy as np
import os
import cv2
import model
from keras.models import load_model

from keras.optimizers import *
# import model_deeplab3plus as DV3
# import model_FCDenseNet as FCDN
import classifiation
import postprocessing
import excel
import estimate
import data


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

if __name__ == "__main__":
    date = "20201128"
    training_num = 51984
    name_loss = "CE"
    name_model = ["UNet(2RDB8-DtoU-5)"]
                  # "UNet(ELu)",
                  # "SegNet",
                  # "DV3",
                  # "UNet"]

    name = [date + "_256_" + str(training_num) + "(08high)_" + name_model[0] + "_" + name_loss,
            date + "_256_" + str(training_num) + "(08low)_" + name_model[0] + "_" + name_loss]
            # date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[2] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[3] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[4] + "_" + name_loss]

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
    # (train_x, train_y) = (np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
    #                       np.load(".\\npy\\V1_start1_total51984_size256_y.npy"))
    # (test_x, test_y) = (np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
    #                       np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"))

    # (train_x, train_y) = (np.concatenate((np.load(".\\npy\\V2_start1_total57600_size256_x.npy"),
    #                                       np.load(".\\npy\\V2_start57601_total14400_size256_x.npy")), axis=0),
    #                       np.concatenate((np.load(".\\npy\\V2_start1_total57600_size256_y.npy"),
    #                                      np.load(".\\npy\\V2_start57601_total14400_size256_y.npy")), axis=0))
    # (train_x, train_y) = (np.load(".\\npy\\V2_start1_total57600_size256_x.npy"),
    #                     np.load(".\\npy\\V2_start1_total57600_size256_y.npy"))
    # (test_x, test_y) = (np.load(".\\npy\\V2_start57601_total14400_size256_x.npy"),
    #                       np.load(".\\npy\\V2_start57601_total14400_size256_y.npy"))
    # test_x = np.load(".\\npy\\start1_total72000_size256_test-x.npy")
    test_data_start = training_num + 1

    for i in range(len(name)):
        # print("Train data shape {}\n{}".format(train_x.shape, train_y.shape))
        print("Building model.")
        input_shape = (256, 256, 3)
        model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                        name="unet_2RD-5",
                                        input_size=input_shape,
                                        block_num=2)
        print("Loading data.")
        if i== 0:
            print("EX high data")
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CETrainData_iou.xlsx"
            total_num = 51984
            (train_x, train_y) = data.extract_high_result(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
                                                          np.load(".\\npy\\V1_start1_total51984_size256_y.npy"),
                                                          excel_file,
                                                          total_num,
                                                          threshold= 0.8)
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE_iou.xlsx"
            total_num = 12996
            (test_x, test_y) = data.extract_high_result(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
                                                        np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"),
                                                        excel_file,
                                                        total_num,
                                                        threshold=0.8)
        else:
            print("EX low data")
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CETrainData_iou.xlsx"
            total_num = 51984
            (train_x, train_y) = data.extract_low_result(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
                                                         np.load(".\\npy\\V1_start1_total51984_size256_y.npy"),
                                                         excel_file,
                                                         total_num)
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE_iou.xlsx"
            total_num = 12996
            (test_x, test_y) = data.extract_low_result(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
                                                       np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"),
                                                       excel_file,
                                                       total_num)

        epochs = 30
        batch = 3
        train_flag = 1
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        # model_select.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i]+ "_trainingNum" + str(len(train_x)) + "testingNum" + str(len(test_x)),
                                  size=input_shape)
        # model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            model_build.train(x_train=train_x, y_train=train_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            model_build.test(x_test= test_x, y_test=test_y, data_start=test_data_start, batch_size=batch)