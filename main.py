import numpy as np
import os
import cv2
import model
from keras.models import load_model

from keras.optimizers import *
# import model_deeplab3plus as DV3
import model_FCDenseNet as FCDN
import classifiation
import postprocessing
import excel
import estimate
import data
import hrnet

'''
    這個檔案可以根據我們要訓練預測的模型進行動作


    參數：
        date：存檔的檔案名稱中，日期
        training_num：訓練資料數量
        name_loss：使用的Loss名稱
        name_model：模型使用的名稱
        name：最終存的模型、預測影像、excel資料的名稱，這影響到這次程式運行會有多少個實驗要跑
        input_shape：輸入模型的資料大小
        batch：在訓練或測試的batch size
        train_flag：是否訓練，1 此階段要訓練 / 0 此階段不訓練
        test_flag：是否測試，1 此階段要測試 / 0 此階段不測試
        epochs：訓練epoch數量

    相關模型的空載入相關參數
    # model_select = model.UNet_DtoU5(block=model.RDBlocks,
        #                                 name="unet_2RD-5",
        #                                 input_size=input_shape,
        #                                 block_num=2)
        # model_select = FCDN.Tiramisu(input_shape= input_shape)
        #  model_select = hrnet.seg_hrnet(batch_size= 3, height= 256, width= 256, channel= 3, classes= 1)
    
'''


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
    date = "20201202"
    training_num = 51984
    name_loss = "CE"
    name_model = ["FCDN",
                  "hrnet"]
                  # "UNet(ELu)",
                  # "SegNet",
                  # "DV3",
                  # "UNet"]
    threshold = np.array([0.9])
    name = [date + "_256_" + str(training_num) + "_" + name_model[0] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss]

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
    (train_x, train_y) = (np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
                          np.load(".\\npy\\V1_start1_total51984_size256_y.npy"))
    (test_x, test_y) = (np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
                          np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"))

    test_data_start = training_num + 1

    for i in range(len(name)):
        # print("Train data shape {}\n{}".format(train_x.shape, train_y.shape))
        print("Building model.")
        input_shape = (256, 256, 3)

        model_select = load_model("") # 載入要使用的模型



        epochs = 30
        batch = 3
        train_flag = 1
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        # model_select.compile(optimizer=Adam(lr=1e-4), loss='                             ', metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)
        # model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            model_build.train(x_train=train_x, y_train=train_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            model_build.test(x_test= test_x, y_test=test_y, data_start=test_data_start, batch_size=batch)