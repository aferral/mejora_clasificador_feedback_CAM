import tkinter

import PIL
from PIL import ImageTk
from tkinter import Frame
import numpy as np
import cv2


def resize(img,factor):
    img=img.copy().astype('uint8')
    if factor != 1:
        return cv2.resize(img, (0, 0), fx=factor, fy=factor).astype('uint8')
    return img


class img_selector(Frame):


    def draw_square(self,square):
        square = list(map(lambda x: int(x / self.factor), list(square)))

        # Crezco la mascara
        self.current_mask[square[2]:square[3], square[0]:square[1]] = True

        # Dibujar en blanco areas en imagen
        if self.rgb:
            np_org = np.array(self.unresized_cam_pil)[:, :, 0:3]
            np_org[self.current_mask] = [0, 0, 255]
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')

            self.img_cam = ImageTk.PhotoImage(image=pil_img)
            self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

            np_org = np.array(self.unresized_cam_pil)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2


        else:
            np_org = np.array(self.unresized_cam_pil)[:, :]
            np_org[self.current_mask] = 255
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'L')

            self.img_cam = ImageTk.PhotoImage(image=pil_img)
            self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

            np_org = np.array(self.unresized_cam_pil)[:, :]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'L')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2

    def __init__(self,controller,current_model,master,*args,**kwargs):

        self.controller = controller

        self.current_mask = None

        self.size=1
        self.factor = 3
        self.n_clases = 2

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
        self.cam_selected =  tkinter.StringVar(w1)
        self.list_cam = tkinter.OptionMenu(w1,self.cam_selected, *['class {0} score: {1}' for i in range(self.n_clases)])
        self.list_cam.pack()
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





        def paintPixels(event):
            x, y = event.x, event.y
            size = self.size
            square = (x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)
            self.draw_square(square)

        self.canvas_cam.bind("<B1-Motion>", paintPixels)

        self.update_model(current_model)



    def update_model(self,current_model):
        self.model = current_model
        self.n_clases = self.model.get_n_classes()

        self.current_index = self.model.get_current_index()
        img,all_cams,scores = self.model.get_img_cam_index(self.current_index)
        mask = self.model.get_mask(self.current_index)

        # configure options select
        self.list_cam['menu'].delete(0, 'end')


        labels = ["class: {0} score: {1}".format(i,scores[i]) for i in range(self.n_clases)]
        self.cam_selected.set(labels[0])
        for i in range(self.n_clases):
            self.list_cam['menu'].add_command(label=labels[i], command=tkinter._setit(self.cam_selected, labels[i]))


        img_cam = all_cams[0]
        self.current_mask = mask

        if (len(img.shape) == 3 and img.shape[2] == 3):
            self.rgb = True
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'RGB')
            self.unresized_cam_pil = PIL.Image.fromarray(resize(img_cam, 1), 'RGB')  # dont scale this

            # configure cam image
            np_org = np.array(self.unresized_cam_pil)[:, :, 0:3]
            np_org[self.current_mask] = [0, 0, 255]
            self.cam_pil = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')

            # configure select image
            np_org = np.array(self.unresized_cam_pil)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(resize(np_org,self.factor), 'RGB')



        else: #GRAYSCALE
            self.rgb = False
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'L')
            self.unresized_cam_pil = PIL.Image.fromarray(resize(img_cam, 1), 'L') # dont scale this

            # configure cam image
            np_org = np.array(self.unresized_cam_pil)[:, :]
            np_org[self.current_mask] = 255
            self.cam_pil = PIL.Image.fromarray(resize(np_org, self.factor), 'L')

            # configure select image
            np_org = np.array(self.unresized_cam_pil)[:, :]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(resize(np_org, self.factor), 'L')



        self.img_or = PIL.ImageTk.PhotoImage(image=self.img_pil)
        self.img_cam = PIL.ImageTk.PhotoImage(image=self.cam_pil)
        self.img_select = ImageTk.PhotoImage(image=select_pil)

        # configurar elementos en GUI
        self.label_img_or.configure(image=self.img_or)

        self.canvas_cam.config(width=self.img_cam.width(),height = self.img_cam.height())
        self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

        self.selection.configure(image=self.img_select)

    def reset_mask(self):
        img_cam = np.array(self.unresized_cam_pil)
        self.current_mask = np.zeros((img_cam.shape[0],img_cam.shape[1])).astype(np.bool)
        self.draw_square((0,0,0,0))


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