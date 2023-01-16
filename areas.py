#!/usr/bin/env python3.11
import cv2
import numpy as np
from pathlib import Path
from pandas import DataFrame

SQUARE_LENGTH = 10


def circle_black_contours(img):
    black_thresh = cv2.inRange(img, (0, 0, 0), (80, 40, 70))
    contours, _ = cv2.findContours(
        black_thresh,
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return img, "Can't find black areas"

    result = img.copy()
    total_area = 0

    for c in contours:
        total_area += cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 1)
    return result, total_area


def crop_yellow_frame(img):
    # change colorspace
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define yellow color boundaries
    hsv_yellow_color_lower = np.array([0, 0, 225])
    hsv_yellow_color_upper = np.array([50, 255, 255])
    thresh_yellow = cv2.inRange(img_hsv, hsv_yellow_color_lower, hsv_yellow_color_upper)

    # finding the biggest contour
    contours, _ = cv2.findContours(thresh_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    margin = 50
    return img[y+margin:y+h-margin, x+margin: x+w-margin]

if __name__ == '__main__':
    images = Path().glob("*.jpg")
    image_strings = [str(p) for p in images]
    areas = []

    # finding black areas from every image in folder 
    for i in image_strings:
        stream = open(i, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        yellow_frame = crop_yellow_frame(img)

        # cv2.imshow('th', yellow_frame)
        result_image, area = circle_black_contours(yellow_frame)

        # cv2.imshow('result', result_image)
        one_centimeter = result_image.shape[1] / SQUARE_LENGTH
        area_in_centimetres = round(area / one_centimeter / one_centimeter, 1)

        areas.append(area_in_centimetres)
        
    # write results to excel file
    df = DataFrame({'File name': image_strings, 'Area in centimeters square': areas})
    df.to_excel(f"Areas.xlsx", 'Лист 1', index=False)
