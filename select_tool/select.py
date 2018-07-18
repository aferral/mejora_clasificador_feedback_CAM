

import cv2
import numpy as np
import os
import pickle

class img_selector:
    def __init__(self,img):

        self.drawing = False  # true if mouse is pressed
        ix, iy = -1, -1
        self.imgLeft = None
        self.imgRight = None
        self.mascaraTemp = None

        self.imgRight=img
        self.size=1
        factor = 0.5

        # mouse callback function
        def addSquare(square):
            square = list(map(lambda x: int(x / factor), list(square)))

            # Crezco la mascara
            self.mascaraTemp[int(square[2]):int(square[3]),
            int(square[0]):int(square[1])] = True

            imageActiva = img
            imgOr = cv2.imread(os.path.join(imagesFolder, imageActiva))

            # pintar azul donde se debe
            imgLeft = imgOr.copy()

            if self.mascaraTemp.any():
                imgLeft[self.mascaraTemp] = (255, 0, 0)
            # Laa imagen de la derecha es el trozo
            derp = np.zeros((imgOr.shape))

            derp[:, :, 0] = self.mascaraTemp.astype(np.int8) * imgOr[:, :, 0]
            derp[:, :, 1] = self.mascaraTemp.astype(np.int8) * imgOr[:, :, 1]
            derp[:, :, 2] = self.mascaraTemp.astype(np.int8) * imgOr[:, :, 2]
            derp = derp.astype(np.uint8)
            self.imgLeft = cv2.resize(imgLeft, (0, 0), fx=factor, fy=factor)
            self.imgRight = cv2.resize(derp, (0, 0), fx=factor, fy=factor)

        def paintPixels(event, x, y, flags, param):

            size=self.size
            square = (
            x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing == True:
                    addSquare(square)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                addSquare(square)


        def update_img():
            imageActiva = img
            original_img = cv2.imread(os.path.join(imagesFolder, imageActiva))
            selected_pixels = np.zeros(original_img.shape)
            self.mascaraTemp = np.zeros((original_img.shape[0],original_img.shape[1])).astype(np.bool)


            self.imgLeft = cv2.resize(original_img, (0, 0), fx=factor, fy=factor)
            self.imgRight = cv2.resize(selected_pixels, (0, 0), fx=factor,fy=factor)

        def nothing(delta):
            self.size = delta

        # IZQ original image
        cv2.namedWindow('original_image')
        cv2.moveWindow('original_image', 100, 100)

        cv2.createTrackbar('Brush size', 'original_image', 0, 10, nothing)
        cv2.setMouseCallback('original_image', paintPixels)

        update_img()

        # DER selected pixels
        cv2.namedWindow('selected_area')
        cv2.moveWindow('selected_area', 600, 100)

        while (1):
            cv2.imshow('original_image', self.imgLeft)
            cv2.imshow('selected_area', self.imgRight)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

        pass

if __name__ == '__main__':
    # ----------------------START PARAMS

    imagesFolder = "/home/aferral/Descargas"
    dataOutFolder = "/home/aferral/Descargas"
    imageTypes = ['jpg']


    # Read images in path
    if not os.path.exists(dataOutFolder):
        os.makedirs(dataOutFolder)
    imageList = []
    for impath in os.listdir(imagesFolder):
        if impath.split('.')[-1] in imageTypes:
            print(impath)
            imageList.append(impath)

    imageList.sort()

    img_selector(imageList[0])


