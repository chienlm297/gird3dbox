import glob
import json
import random as rd

import cv2
import numpy as np

COLOR = [(0, 255, 0), (0, 255, 0), (0, 255, 0)]


def random_color():
    return (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))


def fill_box(img, pt1, pt2, pt3, pt4, color):
    img = cv2.drawContours(img, [np.array([pt2, pt1, pt3, pt4])], -2, color, -3)
    return img


def color_each_floor(index):
    pass


def draw_cube(img, imgpts, color, draw: bool):
    num_box_fill = 0
    if draw:
        num_box_fill = 1
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]], -1, random_color(), -3)

    for i, j in zip(range(4), range(4, 8)):
        pt1 = imgpts[i]
        pt2 = imgpts[j]
        pt3 = imgpts[i + 1] if i < 3 else imgpts[0]
        pt4 = imgpts[j + 1] if j < 7 else imgpts[4]

        if draw:
            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 0, 255), -3)
            img = fill_box(img, pt1, pt2, pt3, pt4, (0, 0, 255))
            img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 1)

    # # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, color, 1)
    img = cv2.drawContours(img, [imgpts[:4]], -1, color, 1)

    return img, num_box_fill


def make_aixsPoints(x: int, y: int, z: int):
    return np.float32(
        [
            [x + 0, y + 0, -z],
            [x + 0, y + 1, -z],
            [x + 1, y + 1, -z],
            [x + 1, y + 0, -z],
            [x + 0, y + 0, -z - 1],
            [x + 0, y + 1, -z - 1],
            [x + 1, y + 1, -z - 1],
            [x + 1, y + 0, -z - 1],
        ]
    )


# Load the camera calibration data

with np.load("Images/Chessboard_9x18/calib.npz") as calibData:
    mtx, dist, rvecs, tvecs = [calibData[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

# Define the chess board rows and columns
rows = 8
cols = 17
heights = 2

# set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# prepare the object points
# they are the sme for all images
objectPoints = np.zeros((rows * cols, 1, 3), np.float32)
objectPoints[:, :, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 1, 2)
print(objectPoints.shape)

box_point = (
    np.mgrid[0:rows, 0:cols, 0:heights]
    .T.reshape(-1, rows * heights, 3)
    .astype("float32")
)
# label_file = open("label.csv", mode="w")
# label_writer = csv.writer(
#     label_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
# )
jsonFile = open("dataview/Labels.json", "w")
# loop over the image files

image_idx = 0
for path in glob.glob("fix2view/*.jpg"):
    index = 0
    # load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    # print(corners)
    if ret:
        print(path)
        # refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # find the rotation and translation vectors
        val, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objectPoints, corners, mtx, dist
        )
        white_img = np.zeros(img.shape, dtype=np.uint8)
        white_img.fill(255)
        x_coords = []
        y_coords = []
        num_box_fill = 0
        for x_ in range(0, rows):
            x_coords.append(x_)
            y_coords = []
            for y_ in range(0, cols):
                y_coords.append(y_)
                for z in range(heights):
                    color = COLOR[z % heights - 1]
                    for x in range(0, rows, 1):
                        for y in range(0, cols, 1):
                            axisPoints = make_aixsPoints(x, y, z)

                            # Project the 3D axis points to the image plane
                            axisImgPoints, jac = cv2.projectPoints(
                                axisPoints, rvecs, tvecs, mtx, dist
                            )

                            # Draw the axis lines
                            draw = (x in x_coords and y in y_coords) if z == 0 else 0

                            corners = corners.astype("int")
                            axisImgPoints = axisImgPoints.astype("int")
                            # img = draw_cube(img, axisImgPoints, color, draw)
                            white_img, _num = draw_cube(
                                white_img, axisImgPoints, color, draw
                            )
                            # if _num == 1 :
                            #     print(_num)
                            num_box_fill += _num
                if image_idx == 0:
                    one_hot_label = [0] * (rows * cols)
                    one_hot_label[: index + 1] = [1] * (index + 1)
                    cv2.imwrite("dataview/view1/{}.png".format(index), white_img)
                    dataDict = {
                        "image_name": "{}.png".format(index),
                        # "x_range": x_coords,
                        # "y_range": y_coords,
                        "label": one_hot_label
                    }
                    dataDict = json.dumps(dataDict)
                    jsonFile.write(dataDict + "\n")
                    index += 1
                else:
                    cv2.imwrite("dataview/view2/{}.png".format(index), white_img)
                    index += 1
    image_idx += 1
