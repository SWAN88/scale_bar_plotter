import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import streamlit as st
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def click_scalebar(event, x, y, flags, params):
    raw_img = params["img"]
    wname = params["wname"]
    point_list = params["point_list"]
    point_num = params["point_num"]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(point_list) < point_num:
            point_list.append([x, y])
    
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(point_list) > 0:
            point_list.pop(-1)
        
    img = raw_img.copy()
    h, w = img.shape[0], img.shape[1]
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)

    for i in range(len(point_list)):
        cv2.drawMarker(img, (point_list[i][0], point_list[i][1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1)
        if 0 < i:
            cv2.line(img, (point_list[i][0], point_list[i][1]),
                     (point_list[i-1][0], point_list[i-1][1]), (0, 255, 0), 2)
        if i == point_num-1:
            cv2.line(img, (point_list[i][0], point_list[i][1]),
                     (point_list[0][0], point_list[0][1]), (0, 255, 0), 2)
            pixel_num = point_list[1][0] - point_list[0][0]
            cv2.putText(img, f"# of pixels in scale bar is {pixel_num}. Press any button to close", (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if 0 < len(point_list) < point_num:
        cv2.line(img, (x, y), (point_list[len(point_list)-1][0], point_list[len(point_list)-1][1]), (0, 255, 0), 2)

    cv2.putText(img, f"Cursor coordinate: ({x}, {y})", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Left Click: put marker, Right Click: remove marker", (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(wname, img)
    

def click_and_crop(event, x, y, flags, params):
    raw_img = params["img"]
    wname = params["wname"]
    refPt = params['refPt']
    
    img = raw_img.copy()
    h, w = img.shape[0], img.shape[1]
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))

    if len(refPt) == 2:
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.putText(img, "Crop image saved", (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(img, f"Cursor coordinate: ({x}, {y})", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(wname, img)

st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")




# img = cv2.imread('sample2.tif')
# img = image_resize(img, height=1000)
# plt.imshow(img)