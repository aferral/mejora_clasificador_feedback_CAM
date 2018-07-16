

import cv2
import numpy as np
import os
import pickle

#----------------------START PARAMS

imagesFolder = "/home/aferral/Descargas"
dataOutFolder = "/home/aferral/Descargas"
imageTypes = ['jpg']
imageDim = 96
startAt = 0

#----------------------END PARAMS
factor=0.5


# Read images in path
if not os.path.exists(dataOutFolder):
    os.makedirs(dataOutFolder)
imageList = []
for impath in os.listdir(imagesFolder):
    if impath.split('.')[-1] in imageTypes:
        print(impath)
        imageList.append(impath)

imageList.sort()


drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
imgLeft = None
imgRight = None
current = startAt
mascaraTemp = None

# mouse callback function
def addSquare(square):
    global imgRight,imgLeft,mascaraTemp, current


    square = list(map(lambda x: int(x / factor), list(square)))
    print('using square ',square)


    #Crezco la mascara
    mascaraTemp[int(square[2]):int(square[3]), int(square[0]):int(square[1])] = True

    imageActiva = imageList[current]
    imgOr = cv2.imread(os.path.join(imagesFolder, imageActiva))


    #pintar azul donde se debe
    imgLeft = imgOr.copy()

    if mascaraTemp.any():
        imgLeft[mascaraTemp] = (255,0,0)
    #Laa imagen de la derecha es el trozo
    derp = np.zeros((imgOr.shape))

    derp[:,:,0] = mascaraTemp.astype(np.int8) * imgOr[:,:,0]
    derp[:, :, 1] = mascaraTemp.astype(np.int8) * imgOr[:,:,1]
    derp[:, :, 2] = mascaraTemp.astype(np.int8) * imgOr[:,:,2]
    # imgRight[mascaraTemp,0] = imgOr[mascaraTemp,0]
    # imgRight[mascaraTemp, 1] = imgOr[mascaraTemp, 1]
    # imgRight[mascaraTemp, 2] = imgOr[mascaraTemp, 2]
    # imgRight[mascaraTemp, :] = imgOr[mascaraTemp, :]
    derp = derp.astype(np.uint8)
    imgRight = derp
    imgLeft = cv2.resize(imgLeft, (0, 0), fx=factor, fy=factor)
    imgRight = cv2.resize(derp, (0, 0), fx=factor, fy=factor)

def paintPixels(event, x, y, flags, param):
    global ix,iy,drawing,mode,imgLeft, mascaraTemp

    size = cv2.getTrackbarPos("Brush size", 'original_image')
    square = (x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_RBUTTONDOWN:
        update_img()
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                addSquare(square)
            else:
                addSquare(square)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            addSquare(square)
        else:
            addSquare(square)
def update_img():
    global current, imgLeft, mascaraTemp, imgRight

    imageActiva = imageList[current]

    #*********ACA VEO SI DEBO GUARDAR RESETEAR O NO GUARDAR LA IMAGEN ACTUAL Y SUS CAMBIOS
    imageName = "".join(imageActiva.split('.')[:-1])
    savedFile = os.path.join(dataOutFolder, imageName+'.pkl')

    switch = cv2.getTrackbarPos('DC/R/S', 'original_image')
    dontsave = (switch == 0)
    reset = (switch == 1)
    save = (switch == 2)


    if reset:
        os.remove(savedFile)
    elif save:
        with open(savedFile, 'wb+') as f:
            pickle.dump([mascaraTemp], f)
    elif dontsave:
        pass



    #*************Ahora paso a la siguiente imagen***********************
    delta = cv2.getTrackbarPos('Prev/Next', 'original_image')
    current = current + 1 if (delta == 1) else current - 1

    imageActiva = imageList[current]
    original_img = cv2.imread(os.path.join(imagesFolder, imageActiva))


    print("current ", current, " ", imageActiva)
    print(imageList[current:current+2])
    print(imageList[0])
    print(imageList[-1])

    imageName = "".join(imageActiva.split('.')[:-1])
    savedFile = os.path.join(dataOutFolder, imageName+'.pkl')

    if not os.path.exists(savedFile):
        print("No existia archivo-imagen lo creo")
        with open(savedFile, 'wb+') as f:
            pickle.dump([np.zeros((original_img.shape[0],original_img.shape[1]))], f)

    with open(savedFile,'rb') as f:
        datos = pickle.load(f)
        mascaraTemp = datos[0].astype(np.bool)
        #agarrar esos pixeles y colocar en imagen 2
        selected_pixels = np.zeros(original_img.shape)
        selected_pixels[mascaraTemp] = original_img[mascaraTemp]

        #En imagen 1 pintar imagen azul en esas cordenadas
        if mascaraTemp.any():
            original_img[mascaraTemp] = (255,0,0)

        # Expandir imagenes y colocar en gobal correpondiente

        imgLeft = cv2.resize(original_img, (0, 0), fx=factor, fy=factor)
        imgRight = cv2.resize(selected_pixels,(0,0),fx=factor,fy=factor)

def nothing(delta):
    pass


#IZQ original image
cv2.namedWindow('original_image')
cv2.moveWindow('original_image',100,100)

cv2.createTrackbar('Prev/Next','original_image',0,1,nothing)
cv2.createTrackbar('Brush size','original_image',0,10,nothing)
cv2.createTrackbar('DC/R/S','original_image',0,2,nothing)

cv2.setMouseCallback('original_image', paintPixels)

update_img()

# DER selected pixels
cv2.namedWindow('selected_area')
cv2.moveWindow('selected_area',600,100)

while(1):

    cv2.imshow('original_image', imgLeft)
    cv2.imshow('selected_area', imgRight)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()