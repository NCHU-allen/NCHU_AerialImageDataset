import cv2
import numpy as np
import os
import excel

# file.rstrip(".tif") """remove .tif"""

def dataset_generator_V1(angle, size = 256):
    def rotate(image, angle, center=None, scale=1.0):

        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated
    count_file = 0
    count_file_read = 0

    for filename in os.listdir(".\\Original\\train\\images"):
        count_file_read +=1
        image = cv2.imread(".\\Original\\train\\images\\" + filename)
        print("{}. Image name：{}".format(count_file_read, filename))
        label = cv2.imread(".\\Original\\train\\gt\\" + filename)
        print("{}. Label name：{}".format(count_file_read, filename))
        (x, y, channels) = image.shape
        print("image shape")
        print(image.shape)
        print("label shape")
        print(label.shape)

        # for dx in range(x // size):
        #     for dy in range(y // size):
        for dy in range(y // size):
            for dx in range(x // size):

                print("{}. image save：".format(count_file_read))
                print(count_file + 1)

                x_low = size * dx
                x_heigh = size * (dx + 1)
                y_low = size * dy
                y_heigh = size * (dy + 1)

                if x_heigh > x or y_heigh > y:
                    break

                save_tif = rotate(image[x_low:x_heigh, y_low:y_heigh], angle=angle)
                save_label = rotate(label[x_low:x_heigh, y_low:y_heigh], angle=angle)

                cv2.imwrite(".\\self_dataset\\Strategy2\\x\\" + str(count_file + 1) + ".tif", save_tif)
                cv2.imwrite(".\\self_dataset\\Strategy2\\y\\" + str(count_file + 1) + ".tif", save_label)

                count_file +=1

def dataset_generator(size = 256):
    count_file = 0
    count_file_read = 0

    for filename in os.listdir(".\\Original\\train\\images"):
        count_file_read +=1
        image = cv2.imread(".\\Original\\train\\images\\" + filename)
        print("{}. Image name：{}".format(count_file_read, filename))
        label = cv2.imread(".\\Original\\train\\gt\\" + filename)
        print("{}. Label name：{}".format(count_file_read, filename))
        (x, y, channels) = image.shape
        print("image shape")
        print(image.shape)
        print("label shape")
        print(label.shape)

        for dx in range((x // size) + 1):
            for dy in range((y // size) + 1):
                x_low = size * dx
                x_heigh = size * (dx + 1)
                y_low = size * dy
                y_heigh = size * (dy + 1)

                if x_heigh > x or y_heigh > y:
                    save_tif = np.zeros([size,size, channels], dtype= np.uint8)
                    save_label = np.zeros([size,size,channels], dtype= np.uint8)

                    if x_heigh > x and y_heigh > y:
                        save_tif[: (x-x_low), :(y-y_low)] = image[x_low:x, y_low:y]
                        save_label[: (x-x_low), :(y-y_low)] = label[x_low:x, y_low:y]
                    elif x_heigh > x:
                        save_tif[: (x-x_low), :] = image[x_low:x, y_low:y_heigh]
                        save_label[: (x-x_low), :] = label[x_low:x, y_low:y_heigh]
                    elif y_heigh > y:
                        save_tif[:, :(y-y_low)] = image[x_low:x_heigh, y_low:y]
                        save_label[:, :(y-y_low)] = label[x_low:x_heigh, y_low:y]
                    # break

                else:
                    save_tif = image[x_low:x_heigh, y_low:y_heigh]
                    save_label = label[x_low:x_heigh, y_low:y_heigh]
                print("{}. image save：".format(count_file_read))
                print(count_file + 1)
                cv2.imwrite(".\\test_data\\" + str(count_file + 1) + ".tif", save_tif)
                cv2.imwrite(".\\self_dataset\\y\\" + str(count_file + 1) + ".tif", save_label)

                count_file +=1

def dataset_read(total = 35922, size= 256):
    x = np.zeros([total, size, size, 3], dtype=np.float32)
    y = np.zeros([total, size, size, 1], dtype=np.uint8)
    count_file_read = 0

    for filename in os.listdir(".\\self_dataset\\Strategy2\\x"):
        if count_file_read % 100 == 0:
            print("{}. Read file：{}".format(count_file_read + 1, filename))

        image = cv2.imread(".\\self_dataset\\Strategy2\\x\\" + filename)
        label = np.expand_dims(cv2.imread(".\\self_dataset\\Strategy2\\y\\" + filename, 0),
                               axis= -1)

        x[count_file_read] = image.astype("float32") / 255
        y[count_file_read] = label.astype("float32") / 255

        count_file_read += 1
    # return x
    return (x, y)

def extract_high_result(x, y, excel_file, total_num, extract_index= "iou", threshold= 0.5):
    file = excel.Excel(file_path= excel_file)
    print("Excel file {} is opened.".format(excel_file))

    if extract_index == "iou" or extract_index == "IoU":
        index = file.read_excel(start= "c3:c" + str(2+total_num))
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

    index = np.array(index)

    print("Index is {}.".format(extract_index))
    low_result = np.array(np.where(index <= threshold))
    re_x = np.delete(x, low_result, axis=0)
    re_y = np.delete(y, low_result, axis=0)

    return (re_x, re_y)

def extract_low_result(x, y, excel_file, total_num, extract_index= "iou", threshold= 0.5):
    file = excel.Excel(file_path= excel_file)
    print("Excel file {} is opened.".format(excel_file))

    if extract_index == "iou" or extract_index == "IoU":
        index = file.read_excel(start= "c3:c" + str(2+total_num))
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

    index = np.array(index)

    print("Index is {}.".format(extract_index))
    low_result = np.array(np.where(index > threshold))
    re_x = np.delete(x, low_result, axis=0)
    re_y = np.delete(y, low_result, axis=0)
    return (re_x, re_y)

# def image_concate(output_size = 5000, input_size = 256):
#     count = 0
#
#     for filename in os.listdir(".\\Original\\test\\images"):
#         count += 1
#         output_img = np.zeros([output_size, output_size], dtype= np.uint8)
#
#         for index in range(400):
#
#
#
#         cv2.imwrite(".\\test_data\\test_result\\" + filename + ".tif")




# def file_clip(input_file, size= 256):
#     ori_input = cv2.imread(input_file)
#     (x, y, ch) = ori_input.shape
#
#     for plux_x in range(x // size):


if __name__ == "__main__":
    (train_x, train_y) = (np.load(".\\npy\\V1_start1_total51984_size256_x.npy"),
                          np.load(".\\npy\\V1_start1_total51984_size256_y.npy"))
    (test_x, test_y) = (np.load(".\\npy\\V1_start51985_total12996_size256_x.npy"),
                        np.load(".\\npy\\V1_start51985_total12996_size256_y.npy"))
    # dataset_generator(size= 256)
    (x, y) = dataset_read(total= 64980, size= 256)
    total = 51984
    np.save('.\\npy\\strategy2_start1_total' + str(total) + "_size256_x.npy", x[:total])
    np.save('.\\npy\\strategy2_start1_total' + str(total) + "_size256_y.npy", y[:total])
    np.save('.\\npy\\strategy2_start' + str(total + 1) + '_total' + str(12996) + "_size256_x.npy", x[total:])
    np.save('.\\npy\\strategy2_start' + str(total + 1) + '_total' + str(12996) + "_size256_y.npy", y[total:])

    # total = 14400
    # print("first save : {}".format(x[57600:].shape))
    # np.save('.\\npy\\start57601_total' + str(total) + "_size256_x.npy", x[57600:])
    # np.save('.\\npy\\start57601_total' + str(total) + "_size256_y.npy", y[57600:])



# (x, y, channels) = label.shape
# mask = np.zeros([x, y, 7], dtype=np.float32)
#
# for dx in range(x):
#     for dy in range(y):
#         flag = 0
#         for i in range(7):
#             if (label[dx, dy] == label_class[i]).all():
#                 mask[dx, dy, i] = 1
# cv2.imshow("original", label)
# for i in range(7):
#     cv2.imshow(label_name[i], (mask[:,:,i]*255).astype("uint8"))
