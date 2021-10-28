import cv2
import numpy as np
import glob
import random as rd


COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def random_color():
    return (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))


def color_each_floor(index):
    pass


def draw_cube(img, imgpts, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]], -1, random_color(), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 1)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, color, 1)

    return img


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

with np.load("../data/calib.npz") as calibData:
    mtx, dist, rvecs, tvecs = [calibData[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

# define the chess board rows and columns

rows = 6
cols = 9
heights = 3

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
print(box_point.shape)
print(box_point[1])

# loop over the image files
index = 0
# for path in glob.glob("../data/left[0-1][0-9].jpg"):
for path in glob.glob("images/*.jpg"):
    index += 1
    # load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    if ret:
        # refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # find the rotation and translation vectors
        val, rvecs, tvecs, inliers = cv2.solvePnPRansac(
            objectPoints, corners, mtx, dist
        )
        for z in range(heights):
            color = COLOR[z % heights - 1]
            for x in range(rows):
                for y in range(cols):
                    axisPoints = make_aixsPoints(x, y, z)

                    # Project the 3D axis points to the image plane
                    axisImgPoints, jac = cv2.projectPoints(
                        axisPoints, rvecs, tvecs, mtx, dist
                    )

                    # draw the axis lines
                    # Draw the axis lines
                    corners = corners.astype("int")
                    axisImgPoints = axisImgPoints.astype("int")
                    img = draw_cube(img, axisImgPoints, color)

        cv2.imwrite("save_image/{}.png".format(index), img)

    # Display the image
    cv2.imshow("chess board", img)
    cv2.waitKey(700)

cv2.destroyAllWindows()
