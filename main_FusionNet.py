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
    activation = ["sigmoid",
                  "softmax"]
    name_model = ["FusionNet(sigmoid)",
                  "FusionNet(softmax)",
                  "FusionNet(softmax-sigmoid)"]
                  # "UNet(ELu)",
                  # "SegNet",
                  # "DV3",
                  # "UNet"]

    name = [date + "_256_" + str(training_num) + "_" + name_model[0] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss,
            date + "_256_" + str(training_num) + "_" + name_model[2] + "_" + name_loss]
            # date + "_256_" + str(training_num) + "_" + name_model[1] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[2] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[3] + "_" + name_loss,
            # date + "_256_" + str(training_num) + "_" + name_model[4] + "_" + name_loss]

    test_data_start = training_num + 1
    input_shape = (256, 256, 3)

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
    # (train_x, train_y) = (np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
    #                       np.load(".\\npy\\V1_start1_total51984_size256_y.npy"))
    # (test_x, test_y) = (np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
    #                       np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"))

    model_segmentation = model.UNet_DtoU5(block=model.RDBlocks,
                                          name="unet_2RD-5",
                                          input_size=input_shape,
                                          block_num=2)
    model_segmentation.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    # 主支線生成訓練測試x資料
    # model_segmentation.load_weights(".\\result\\model_record\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE.h5")
    # mainpipe_train_x = model_segmentation.predict(train_x, batch_size= 3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-mainpipe_train_x.npy", mainpipe_train_x)
    # mainpipe_test_x = model_segmentation.predict(test_x, batch_size=3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-mainpipe_test_x.npy", mainpipe_test_x)

    # 副支線生成訓練測試x資料
    model_judge = classifiation.Alexnet(input_shape=input_shape, output_class=2)
    model_judge.compile(optimizer= Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    name_subpipe = date + "_256_" + str(training_num) + "_SIMark_" + name_loss
    model_subpipe = model.SI_mark_processing([model_segmentation, model_judge], name=name_subpipe, input_shape= input_shape, pipelines = 2)
    subpipe_train_x = model_subpipe.predict(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"), np.load(".\\npy\\V1_start1_total51984_size256_y.npy"),
                                            judge_weights_path= ".\\result\\model_record\\20201128_256_51984(th08)_JudgeAlexNet_CE.h5",
                                            pipe_weights_path= [".\\result\\model_record\\20201202_256_51984(09high)_UNet(2RDB8-DtoU-5)_CE_trainingNum24020testingNum2662.h5",
                                                                ".\\result\\model_record\\20201202_256_51984(09low)_UNet(2RDB8-DtoU-5)_CE_trainingNum27964testingNum10334.h5"],
                                            data_start= test_data_start,
                                            save_path= model_subpipe.name + "trainData",
                                            judge_threshold = 0.9)
    np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_train_x.npy", subpipe_train_x)
    subpipe_test_x = model_subpipe.predict(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"), np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"),
                                           judge_weights_path= ".\\result\\model_record\\20201128_256_51984(th08)_JudgeAlexNet_CE.h5",
                                           pipe_weights_path= [".\\result\\model_record\\20201128_256_51984(08high)_UNet(2RDB8-DtoU-5)_CE_trainingNum36700testingNum5708.h5",
                                                               ".\\result\\model_record\\20201128_256_51984(08low)_UNet(2RDB8-DtoU-5)_CE_trainingNum36700testingNum5708.h5"],
                                           data_start= test_data_start,
                                           save_path=model_subpipe.name + "testData",
                                           judge_threshold = 0.9)
    np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-subpipe_test_x.npy", subpipe_test_x)
    for i in range(len(name)):
        print("Building model.")
        # model_select = model.UNet_DtoU5(block=model.RDBlocks,
        #                                 name="unet_2RD-5",
        #                                 input_size=input_shape,
        #                                 block_num=2)
        if i < 2:
            model_select = model.Fusion_net(activation= activation[i], size=(256, 256, 1))
        else:
            model_select = model.Fusion_net_twoActivation(size=(256, 256, 1))

        epochs = 30
        batch = 10
        train_flag = 1
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Loading data.")
            mainpipe_train_x = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-mainpipe_train_x.npy")
            subpipe_train_x = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_train_x.npy")
            print("End loading data.")
            print("Start training.")
            model_build.train(x_train=[mainpipe_train_x, subpipe_train_x], y_train=train_y, batch_size=batch, epochs= epochs)
            print("End training.")

        if test_flag:
            print("Loading data.")
            mainpipe_test_x = np.load(
                ".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-mainpipe_test_x.npy")
            subpipe_test_x = np.load(
                ".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-subpipe_test_x.npy")
            print("End loading data.")
            print("Start testing.")
            model_build.test(x_test= [mainpipe_test_x, subpipe_test_x], y_test=test_y, data_start=test_data_start, batch_size=batch)
            print("End testing.")