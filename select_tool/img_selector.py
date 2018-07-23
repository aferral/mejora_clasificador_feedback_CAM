import tkinter

import PIL
from PIL import ImageTk
from tkinter import Frame
import numpy as np


class img_selector(Frame):
    def __init__(self,controller,current_model,master,*args,**kwargs):

        self.controller = controller

        self.current_mask = None
        self.size=1
        self.factor = 1

        self.img_or = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))
        self.img_cam = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))
        self.img_select = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))

        # Draw
        super().__init__(master,*args,**kwargs)

        # Label for original image
        w0 = tkinter.Toplevel(self)
        w0.title('Img_org')
        tkinter.Label(w0, text="Img_org").pack()
        self.label_img_or = tkinter.Label(w0, image=self.img_or)
        self.label_img_or.pack()

        # Canvas for CAM image
        w1 = tkinter.Toplevel(self)
        w1.title('Img_cam')
        tkinter.Label(w1, text="Img_cam").pack()
        self.canvas_cam = tkinter.Canvas(w1, width=self.img_cam.width(), height=self.img_cam.height())
        tkinter.Button(w1, text="Reset mask (Doesnt commit yet)",command=self.reset_mask).pack()
        tkinter.Button(w1, text="Commit changes",command=self.commit_mask_changes).pack()

        # Label for selection image
        w2 = tkinter.Toplevel(self)
        w2.title('Img_org')
        tkinter.Label(w2, text="Selection").pack()
        self.selection = tkinter.Label(w2, image=self.img_select)
        self.selection.pack()



        # Configure canvas window
        def set_size(x):
            self.size = int(x)

        self.sl = tkinter.Scale(w1, from_=0, to=40, orient=tkinter.HORIZONTAL, command=set_size).pack()
        self.canvas_cam.pack()

        self.canvas_img = self.canvas_cam.create_image((0, 0), anchor=tkinter.NW, image=self.img_cam)


        # mouse callback function
        def addSquare(square):
            square = list(map(lambda x: int(x / self.factor), list(square)))

            # Crezco la mascara
            self.current_mask[square[2]:square[3], square[0]:square[1]] = True

            # Dibujar en blanco areas en imagen
            if self.rgb:
                np_org = np.array(self.cam_or)[:,:,0:3]
                np_org[self.current_mask] = [0, 0, 255]
                pil_img = PIL.Image.fromarray(np_org.astype('uint8'), 'RGB')
            else:
                np_org = np.array(self.cam_or)[:,:]
                np_org[self.current_mask] = 255
                pil_img = PIL.Image.fromarray(np_org.astype('uint8'), 'L')

            self.img_cam=ImageTk.PhotoImage(image=pil_img)
            self.canvas_cam.itemconfig(self.canvas_img,image=self.img_cam)


            # Dibujar areas cortadas en select
            if self.rgb:
                np_org = np.array(self.cam_or)[:,:,0:3]
                np_org[np.bitwise_not(self.current_mask)] = 0
                pil_img = PIL.Image.fromarray(np_org.astype('uint8'), 'RGB')
            else:
                np_org = np.array(self.cam_or)[:,:]
                np_org[np.bitwise_not(self.current_mask)] = 0
                pil_img = PIL.Image.fromarray(np_org.astype('uint8'), 'L')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2

        def paintPixels(event):
            x, y = event.x, event.y
            size = self.size
            square = (x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)
            addSquare(square)

        self.canvas_cam.bind("<B1-Motion>", paintPixels)

        self.update_model(current_model)



    def update_model(self,current_model):
        self.model = current_model

        self.current_index = self.model.get_current_index()
        img,img_cam = self.model.get_img_cam_index(self.current_index)
        mask = self.model.get_mask(self.current_index)

        self.current_mask = mask

        if (len(img.shape) == 3 and img.shape[2] == 3):
            self.rgb = True
            self.img_pil = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
            self.cam_or = PIL.Image.fromarray(img_cam.astype('uint8'), 'RGB')
            np_org = np.array(self.cam_or)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(np_org.astype('uint8'), 'RGB')
        else: #GRAYSCALE
            self.rgb = False
            self.img_pil = PIL.Image.fromarray(img.astype('uint8'), 'L')
            self.cam_or = PIL.Image.fromarray(img_cam.astype('uint8'), 'L')
            np_org = np.array(self.cam_or)[:, :]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(np_org.astype('uint8'), 'L')



        self.img_or = PIL.ImageTk.PhotoImage(image=self.img_pil)
        self.img_cam = PIL.ImageTk.PhotoImage(image=self.cam_or)
        self.img_select = ImageTk.PhotoImage(image=select_pil)

        # configurar elementos en GUI
        self.label_img_or.configure(image=self.img_or)

        self.canvas_cam.config(width=self.img_cam.width(),height = self.img_cam.height())
        self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

        self.selection.configure(image=self.img_select)

    def reset_mask(self):
        self.update_model( self.model)


    def commit_mask_changes(self):
        print('Commiting mask changes for index: {0}'.format(self.current_index))
        updated_mask = self.current_mask

        self.controller.event_draw_mask(updated_mask)


if __name__ == '__main__':

    tk = tkinter.Tk()
    tkinter.Label(tk,text='test').pack()
    selector = img_selector(None,None,tk)
    selector.pack()
    tk.mainloop()