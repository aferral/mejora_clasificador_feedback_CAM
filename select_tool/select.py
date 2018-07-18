

import cv2
import numpy as np
import os
import pickle






class img_selector:
    def __init__(self,controller,current_model):

        self.drawing = False  # true if mouse is pressed
        ix, iy = -1, -1

        self.img_org = img
        self.img_cam = img_cam
        self.img_select = img
        self.mascaraTemp = None

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
            self.img_cam = cv2.resize(imgLeft, (0, 0), fx=factor, fy=factor)
            self.img_select = cv2.resize(derp, (0, 0), fx=factor, fy=factor)

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

            self.img_org = cv2.resize(original_img, (0, 0), fx=factor, fy=factor)
            self.img_cam = cv2.resize(original_img, (0, 0), fx=factor, fy=factor)
            self.img_select = cv2.resize(selected_pixels, (0, 0), fx=factor,fy=factor)

        def nothing(delta):
            self.size = delta

        # Original image
        cv2.namedWindow('original_image')
        cv2.moveWindow('original_image', 100, 100)

        # CAM image
        cv2.namedWindow('CAM_image')
        cv2.moveWindow('CAM_image', 400, 100)
        cv2.createTrackbar('Brush size', 'CAM_image', 0, 10, nothing)
        cv2.setMouseCallback('CAM_image', paintPixels)

        update_img()

        # Selected pixels
        cv2.namedWindow('selected_area')
        cv2.moveWindow('selected_area', 700, 100)

        while (1):
            cv2.imshow('original_image', self.img_org)
            cv2.imshow('CAM_image', self.img_cam)
            cv2.imshow('selected_area', self.img_select)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    def update_model(self,current_model):
        self.model = current_model
        self.current_index = self.model.get_current_index()
        img = self.model.get_img_index(self.current_index)
        img_cam = self.model.get_img_cam(self.current_index)
        mask = self.model.get_mask(self.current_index)

    def commit_mask_changes(self):

        #tomar mascara nueva
        updated_mask = None

        # editar archivo de mascaras
        self.controller.event_draw_mask(updated_mask)

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

    img_selector(imageList[0],imageList[0])


