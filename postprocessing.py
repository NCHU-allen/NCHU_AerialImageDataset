import cv2
import estimate
import numpy as np
import excel
import os
import rasterio

def check_threshold(y_pred, size= (256, 256, 1), threshold= 0.5):
    y_out = np.zeros(((len(y_pred), size[0], size[1], size[2])), dtype= np.uint8)
    print("Check the threshold.")
    for index in range(len(y_pred)):
        for x in range(size[0]):
            for y in range(size[1]):
                if y_pred[index, x, y] > threshold:
                    y_out[index, x, y] = 1
                else:
                    y_out[index, x, y] = 0
        print("threshold image:{}".format(index+1))
    return y_out

def recover_area(img_predict, start, number, file_path = ".\\dataset_V3.2-edge\\edge\\", size = (256, 256, 1), threshold= 0.5):
    result = np.zeros([number, size[0], size[1], size[2]], np.uint8)
    kernel = np.ones((2, 2), np.uint8)

    for index in range(number):
        data_raster = rasterio.open(file_path + str(index + start) + ".tif")
        data_raster_edge = data_raster.read(1)
        _, binary_edge = cv2.threshold(data_raster_edge, 0, 255, cv2.THRESH_BINARY_INV)
        binary_edge = cv2.erode(binary_edge.astype("uint8"), kernel, iterations=1)

        _, contours, hierarchy = cv2.findContours(binary_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in range(len(contours)):
            land_area = np.zeros([size[0], size[1], size[2]], np.uint8)
            cv2.drawContours(land_area, contours, contour, 1, -1)
            union_area = np.sum(np.bitwise_and(land_area, img_predict[index]))

            if union_area / np.sum(land_area) <= threshold:
                print("recover low : {}".format(union_area / np.sum(land_area)))
                continue

            cv2.drawContours(result[index], contours, contour, 1, -1)

        result[index] = np.expand_dims(cv2.dilate(result[index], kernel, iterations=1), axis=-1)
    return result

if __name__ == "__main__":
    file_path = ".\\dataset_V3.2-edge\\edge\\"
    size = (256, 256, 1)
    threshold = 0.5
    result = np.zeros([size[0], size[1], size[2]], np.uint8)
    kernel = np.ones((2, 2), np.uint8)

    data_raster = rasterio.open(file_path + str(23) + ".tif")
    data_raster_edge = data_raster.read(1)
    _, binary_edge = cv2.threshold(data_raster_edge, 0, 1, cv2.THRESH_BINARY_INV)
    binary_edge = cv2.erode(binary_edge.astype("uint8"), kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(binary_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in range(len(contours)):
        land_area = np.zeros([size[0], size[1], size[2]], np.uint8)
        cv2.drawContours(land_area, contours, contour, 255, -1)
        cv2.imshow(str(contour), land_area)
        cv2.imwrite("D:\\allen\project\git\map\\result\\23_" + str(contour + 1) + ".png", land_area)
        # union_area = np.sum(np.bitwise_and(land_area, 255 * img_predict[index]))

        # if union_area / np.sum(land_area) <= threshold:
        #     continue
        # cv2.drawContours(result, contours, contour, 255, -1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()