import tkinter

import PIL
from PIL import ImageTk
from tkinter import Frame
import numpy as np

# todo:
# mejorar remplazo en canvas
# revisar update
# incorporar version dummy

class img_selector(Frame):
    def __init__(self,controller,current_model,master,*args,**kwargs):

        self.controller = controller

        self.drawing = False  # true if mouse is pressed
        ix, iy = -1, -1
        # self.img_org = img
        # self.img_cam = img_cam
        # self.img_select = img
        self.mascaraTemp = None

        self.size=1
        self.factor = 1

        # Draw
        super().__init__(master,*args,**kwargs)

        self.update_model(current_model)



    def update_model(self,current_model):
        self.model = current_model
        # self.current_index = self.model.get_current_index()
        # img = self.model.get_img_index(self.current_index)
        # img_cam = self.model.get_img_cam(self.current_index)
        # mask = self.model.get_mask(self.current_index)

        self.pil_or = PIL.Image.open("/home/aferral/Descargas/23815fe65adf63f76e9179fcf7389926.png")
        self.img_or = PIL.ImageTk.PhotoImage(image=self.pil_or)
        self.img_cam = PIL.ImageTk.PhotoImage(image=self.pil_or) # todo

        self.mascaraTemp = np.zeros((self.img_cam.height(), self.img_cam.width(),3))

        pil_img = PIL.Image.fromarray(self.mascaraTemp.astype('uint8'), 'RGB')
        self.img_select = PIL.ImageTk.PhotoImage(image=pil_img)



        # Label for original image
        w0 = tkinter.Toplevel(self)
        w0.title('Img_org')
        tkinter.Label(w0, text="Img_org").pack()
        tkinter.Label(w0, image=self.img_or).pack()

        # Canvas for CAM image
        w1 = tkinter.Toplevel(self)
        w1.title('Img_cam')
        tkinter.Label(w1, text="Img_cam").pack()
        self.canvas_cam = tkinter.Canvas(w1, width=self.img_cam.width(), height=self.img_cam.height())
        tkinter.Button(w1, text="Reset mask").pack()
        tkinter.Button(w1, text="Save mask").pack()

        # Configure canvas window
        def set_size(x):
            self.size = int(x)

        self.sl = tkinter.Scale(w1, from_=0, to=40, orient=tkinter.HORIZONTAL, command=set_size).pack()
        self.canvas_cam.pack()



        # mouse callback function
        def addSquare(square):
            square = list(map(lambda x: int(x / self.factor), list(square)))

            # Crezco la mascara
            self.mascaraTemp[square[2]:square[3], square[0]:square[1]] = True

            # Dibujar en azul areas en imagen
            python_green = "#476042"
            self.canvas_cam.create_rectangle(square[0], square[2], square[1], square[3],fill=python_green)

            # Dibujar areas cortadas en select
            np_org = np.array(self.pil_or)[:,:,0:3]
            np_org[np.bitwise_not(self.mascaraTemp.astype(np.bool))] = 0
            pil_img = PIL.Image.fromarray(np_org.astype('uint8'), 'RGB')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2

        def paintPixels(event):
            x, y = event.x, event.y

            size = self.size
            square = (x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)
            addSquare(square)


        self.canvas_cam.bind("<B1-Motion>", paintPixels)
        self.canvas_cam.create_image((0, 0), anchor=tkinter.NW, image=self.img_cam)



        # Label for selection image
        w2 = tkinter.Toplevel(self)
        w2.title('Img_org')
        tkinter.Label(w2, text="Selection").pack()
        self.selection = tkinter.Label(w2, image=self.img_select)
        self.selection.pack()




    def commit_mask_changes(self):

        #tomar mascara nueva
        updated_mask = None

        # editar archivo de mascaras
        self.controller.event_draw_mask(updated_mask)


if __name__ == '__main__':

    tk = tkinter.Tk()
    tkinter.Label(tk,text='test').pack()
    selector = img_selector(None,None,tk)
    selector.pack()
    tk.mainloop()