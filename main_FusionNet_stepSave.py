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

'''
    這個檔案訓練跟測試 Fused RDB U-Net

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

'''

if __name__ == "__main__":
    date = "20201128"
    training_num = 51984
    name_loss = "CE"
    activation = ["sigmoid",
                  "relu"]
    name_model = ["FusionNet(sigmoid)",
                  "FusionNet(relu)",
                  "FusionNet(relu-sigmoid)"]
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
    (train_x, train_y) = (np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
                          np.load(".\\npy\\V1_start1_total51984_size256_y.npy"))
    (test_x, test_y) = (np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
                          np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"))

    # model_segmentation = model.UNet_DtoU5(block=model.RDBlocks,
    #                                       name="unet_2RD-5",
    #                                       input_size=input_shape,
    #                                       block_num=2)
    # model_segmentation.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    # 主支線生成訓練測試x資料
    # model_segmentation.load_weights(".\\result\\model_record\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE.h5")
    # mainpipe_train_x = model_segmentation.predict(train_x, batch_size= 3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-mainpipe_train_x.npy", mainpipe_train_x)
    # mainpipe_test_x = model_segmentation.predict(test_x, batch_size=3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-mainpipe_test_x.npy", mainpipe_test_x)

    # 副支線生成訓練測試x資料
    # dist_data 儲存
    # print("Dist_data preocessing")
    # model_judge = classifiation.Alexnet(input_shape=input_shape, output_class=2)
    # model_judge.compile(optimizer= Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model_judge.load_weights(".\\result\\model_record\\20201216_256_51984(th09)_JudgeAlexNet_CE.h5")
    # dist_data = model_judge.predict(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"), batch_size=10)
    # dist_data = np.ones(dist_data[:, 0].shape) * (dist_data[:, 0] > 0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-Judge-distData.npy", dist_data)
    # dist_data = model_judge.predict(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"), batch_size=10)
    # dist_data = np.ones(dist_data[:, 0].shape) * (dist_data[:, 0] > 0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-Judge-distData.npy", dist_data)

    # subpipe 預測
    # print("Subpipe processing training")
    # dist_data = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-Judge-distData.npy")
    # # high_x = np.delete(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"), np.where(dist_data == 0), axis=0)
    # # high_y = np.delete(np.load(".\\npy\\V1_start1_total51984_size256_y.npy"), np.where(dist_data == 0), axis=0)
    # # model_segmentation.load_weights(".\\result\\model_record\\20201202_256_51984(09high)_UNet(2RDB8-DtoU-5)_CE_trainingNum24020testingNum2662.h5")
    # # predict_high_y = model_segmentation.predict(high_x, batch_size=3)
    # # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-predictHighY.npy", predict_high_y)
    # low_x = np.delete(np.load(".\\npy\\V1_start1_total51984_size256_x.npy"), np.where(dist_data == 1), axis=0)
    # low_y = np.delete(np.load(".\\npy\\V1_start1_total51984_size256_y.npy"), np.where(dist_data == 1), axis=0)
    # model_segmentation.load_weights(
    #     ".\\result\\model_record\\20201202_256_51984(09low)_UNet(2RDB8-DtoU-5)_CE_trainingNum27964testingNum10334.h5")
    # predict_low_y = model_segmentation.predict(low_x, batch_size=3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-predictLowY.npy", predict_low_y)
    #
    # predict_high_y = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-predictHighY.npy")
    # predict_low_y = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-predictLowY.npy")
    # dist_data = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-Judge-distData.npy")
    # high_order = dist_data == 1
    # index_high = 0
    # index_low = 0
    # predict_y = []
    # print("Concate predict data")
    # for index in range(51984):
    #     if high_order[index]:
    #         predict_y.append(predict_high_y[index_high])
    #         # predict_y[index] = predict_high_y[index_high]
    #         index_high += 1
    #         # np.delete(high_y_predict, 0, axis=0)
    #     else:
    #         predict_y.append(predict_low_y[index_low])
    #         # predict_y[index] = predict_low_y[index_low]
    #         index_low += 1
    #         # np.delete(low_y_predict, 0, axis=0)
    # predict_y = np.array(predict_y)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_predictY.npy", predict_y)
    # ooutput_y = postprocessing.check_threshold(predict_y,
    #                                            size=(256, 256, 1),
    #                                            threshold=0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_outputY.npy", ooutput_y)

    # print("Subpipe processing test data")
    # dist_data = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-Judge-distData.npy")
    # high_x = np.delete(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"), np.where(dist_data == 0), axis=0)
    # high_y = np.delete(np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"), np.where(dist_data == 0), axis=0)
    # model_segmentation.load_weights(
    #     ".\\result\\model_record\\20201202_256_51984(09high)_UNet(2RDB8-DtoU-5)_CE_trainingNum24020testingNum2662.h5")
    # predict_high_y = model_segmentation.predict(high_x, batch_size=3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-predictHighY.npy", predict_high_y)
    # low_x = np.delete(np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"), np.where(dist_data == 1), axis=0)
    # low_y = np.delete(np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"), np.where(dist_data == 1), axis=0)
    # model_segmentation.load_weights(
    #     ".\\result\\model_record\\20201202_256_51984(09low)_UNet(2RDB8-DtoU-5)_CE_trainingNum27964testingNum10334.h5")
    # predict_low_y = model_segmentation.predict(low_x, batch_size=3)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-predictLowY.npy", predict_low_y)
    #
    # high_order = dist_data == 1
    # index_high = 0
    # index_low = 0
    # predict_y = []
    # # predict_high_y = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-predictHighY.npy")
    # # predict_low_y = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-predictLowY.npy")
    # print("Concate predict data")
    # print(predict_high_y.shape)
    # print(predict_low_y.shape)
    # for index in range(12996):
    #     if high_order[index]:
    #         predict_y.append(predict_high_y[index_high])
    #         # predict_y[index] = predict_high_y[index_high]
    #         index_high += 1
    #         # np.delete(high_y_predict, 0, axis=0)
    #     else:
    #         predict_y.append(predict_low_y[index_low])
    #         # predict_y[index] = predict_low_y[index_low]
    #         index_low += 1
    #         # np.delete(low_y_predict, 0, axis=0)
    # predict_y = np.array(predict_y)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-subpipe_predictY.npy", predict_y)
    # ooutput_y = postprocessing.check_threshold(predict_y,
    #                                            size=(256, 256, 1),
    #                                            threshold=0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-subpipe_outputY.npy", ooutput_y)

    for i in range(len(name)):
        print("Building model.")

        if i ==1:
            model_select = model.Fusion_net(activation= activation[i], size=(256, 256, 1))
        else:
            continue
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
            subpipe_train_x = np.load(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_predictY.npy")
            print("End loading data.")
            print("Start training.")
            model_build.train(x_train=[mainpipe_train_x, subpipe_train_x], y_train=train_y, batch_size=batch, epochs= epochs)
            print("End training.")

        if test_flag:
            print("Loading data.")
            mainpipe_test_x = np.load(
                ".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-mainpipe_test_x.npy")
            subpipe_test_x = np.load(
                ".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTestData(V1)-subpipe_predictY.npy")
            print("End loading data.")
            print("Start testing.")
            model_build.test(x_test= [mainpipe_test_x, subpipe_test_x], y_test=test_y, data_start=test_data_start, batch_size=batch)
            print("End testing.")