import tkinter
import pickle
import PIL
from PIL import ImageTk
from tkinter import Frame
import numpy as np
import cv2

selection_rgb_color = [0,255,0]

def resize(img,factor):
    img=img.copy().astype('uint8')
    if factor != 1:
        return cv2.resize(img, (0, 0), fx=factor, fy=factor).astype('uint8')
    return img

def call_one_use_select(cam_or,img_or=None):

    tk = tkinter.Tk()
    t=one_use_img_selector(cam_or, tk, img_to_use=img_or)
    t.pack()
    return_value = [0]
    def on_closing():
        return_value[0]=t.current_mask
        tk.destroy()

    tk.protocol("WM_DELETE_WINDOW", on_closing)
    tk.mainloop()
    return return_value[0]



class one_use_img_selector(Frame):

    def draw_square(self,square):
        square = list(map(lambda x: int(x / self.factor), list(square)))

        # Add square to mask
        self.current_mask[square[2]:square[3], square[0]:square[1]] = True

        # Draw selected area for visualization image
        np_org = np.array(self.unresized_vis_for_select)[:, :, 0:3]
        np_org[self.current_mask] = selection_rgb_color
        pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')
        self.img_cam = ImageTk.PhotoImage(image=pil_img)
        self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

        if self.rgb:
            # Draw selected pixels in selection canvas
            np_org = np.array(self.unresized_image_for_select)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2

        else:
            # Draw selected pixels in selection canvas
            np_org = np.array(self.unresized_image_for_select)[:, :]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'L')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2


    def __init__(self,img_cam : np.array,master,*args,img_to_use=None,**kwargs):

        self.current_mask = None

        self.size=1
        self.factor = 5
        self.img_or = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3), 'RGB'))
        self.img_cam = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3), 'RGB'))
        self.img_select = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))

        # Draw
        super().__init__(master,*args,**kwargs)

        def change_img_size(x):
            val=int(x)
            self.factor=val
            self.update_images(self.current_img, self.img_cam_ref)

        def read_imgs(self):
            def random_fun():
                with open('in_buffer_sel.pkl','rb') as f:
                    data_in=pickle.load(f)
                self.update_model(data_in['img'], data_in['cam'])
            return random_fun

        def commit_mask_changes(self):
            def random_bunf():
                print('EXPORTING mask changes to {0}'.format('out_mask.pkl'))
                with open('out_mask.pkl', 'wb') as f:
                    pickle.dump(self.current_mask, f)
            return random_bunf

        def set_size(x):
            self.size = int(x)

        w = tkinter.Scale(master,from_=1, to=5, orient=tkinter.HORIZONTAL,command=change_img_size)
        w.pack()
        tkinter.Button(master, text="Read imgs", command=read_imgs(self)).pack()
        tkinter.Button(master, text="Export mask", command=commit_mask_changes(self)).pack()
        self.sl = tkinter.Scale(master, from_=0, to=40, orient=tkinter.HORIZONTAL, command=set_size).pack()

        # Label for original image
        # w0 = tkinter.Toplevel(self)
        temporal_frame = tkinter.Frame()
        temporal_frame.pack(expand=1)

        w0=temporal_frame
        self.label_img_or = tkinter.Label(w0, image=self.img_or)
        self.label_img_or.grid(row=0,column=0)


        # Canvas for CAM image
        # w1 = tkinter.Toplevel(self)
        w1 = temporal_frame
        self.canvas_cam = tkinter.Canvas(w1, width=self.img_cam.width(), height=self.img_cam.height())
        self.canvas_cam.grid(row=0,column=1)


        # Label for selection image
        # w2 = tkinter.Toplevel(self)
        w2=temporal_frame
        self.selection = tkinter.Label(w2, image=self.img_select)
        self.selection.grid(row=0,column=2)

        # Configure canvas window
        self.canvas_img = self.canvas_cam.create_image((0, 0), anchor=tkinter.NW, image=self.img_cam)


        def paintPixels(event):
            x, y = event.x, event.y
            size = self.size
            square = (x - size * 0.5, x + size * 0.5, y - size * 0.5, y + size * 0.5)
            self.draw_square(square)

        self.canvas_cam.bind("<B1-Motion>", paintPixels)


        self.update_model(img_to_use, img_cam)


    def update_images(self,img,img_cam):

        # configure visualization image
        self.unresized_vis_for_select = PIL.Image.fromarray(resize(img_cam, 1), 'RGB')  # dont scale this
        # configure cam image
        np_org = np.array(self.unresized_vis_for_select)[:, :, 0:3]
        np_org[self.current_mask] = selection_rgb_color
        self.cam_pil = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')

        if (len(img.shape) == 3 and img.shape[2] == 3):
            self.rgb = True
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'RGB')
            self.unresized_image_for_select = PIL.Image.fromarray(resize(img, 1), 'RGB')  # dont scale this

            # configure select image
            np_org = np.array(self.unresized_image_for_select)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(resize(np_org,self.factor), 'RGB')


        else: #GRAYSCALE
            self.rgb = False
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'L')
            self.unresized_image_for_select = PIL.Image.fromarray(resize(img, 1), 'L') # dont scale this

            # configure select image
            np_org = np.array(self.unresized_image_for_select)[:, :]
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

    def update_model(self,img_sel,img_canvas):

        if img_selector is None:
            img_sel = img_canvas

        # Save reference to all visualizations
        self.current_img = img_sel
        # aplica colormaps a cams
        self.img_cam_ref = cv2.cvtColor(cv2.applyColorMap(img_canvas, cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)

        # Get mask and configure images
        self.current_mask = np.zeros_like(img_canvas).astype(np.bool)
        self.update_images(self.current_img, self.img_cam_ref)





class img_selector(Frame):

    def draw_square(self,square):
        square = list(map(lambda x: int(x / self.factor), list(square)))

        # Add square to mask
        self.current_mask[square[2]:square[3], square[0]:square[1]] = True

        # Draw selected area for visualization image
        np_org = np.array(self.unresized_vis_for_select)[:, :, 0:3]
        np_org[self.current_mask] = selection_rgb_color
        pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')
        self.img_cam = ImageTk.PhotoImage(image=pil_img)
        self.canvas_cam.itemconfig(self.canvas_img, image=self.img_cam)

        if self.rgb:
            # Draw selected pixels in selection canvas
            np_org = np.array(self.unresized_image_for_select)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2


        else:
            # Draw selected pixels in selection canvas
            np_org = np.array(self.unresized_image_for_select)[:, :]
            np_org[np.bitwise_not(self.current_mask)] = 0
            pil_img = PIL.Image.fromarray(resize(np_org, self.factor), 'L')
            img2 = ImageTk.PhotoImage(image=pil_img)
            self.selection.configure(image=img2)
            self.selection.image = img2


    def __init__(self,controller,current_model,master,*args,**kwargs):

        self.controller = controller

        self.current_mask = None

        self.size=1
        self.factor = 2
        self.n_clases = 2 # dummy initial value. Updated when a model is loaded

        self.img_or = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))
        self.img_cam = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))
        self.img_select = ImageTk.PhotoImage(image=PIL.Image.fromarray(np.random.rand(100,100,3).astype('uint8'), 'RGB'))

        # Draw
        super().__init__(master,*args,**kwargs)

        # Label for original image
        w0 = tkinter.Toplevel(self)
        w0.title('Img_org')
        self.title_img_or = tkinter.StringVar(w0)
        tkinter.Label(w0, textvariable=self.title_img_or).pack()
        self.label_img_or = tkinter.Label(w0, image=self.img_or)
        self.label_img_or.pack()


        labels_cam = ['class {0} score: {1}' for i in range(self.n_clases)]


        # Canvas for CAM image
        w1 = tkinter.Toplevel(self)
        w1.title('Img_cam')
        tkinter.Label(w1, text="Img_cam").pack()
        self.cam_selected =  tkinter.StringVar(w1)

        self.list_cam = tkinter.OptionMenu(w1,self.cam_selected, *labels_cam)
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

    def change_selected_list(self,var, value, index):
        def temp(*args):
            var.set(value)
            print("CHANGING CAM TO {0} -- {1}".format(value, index))
            self.update_images(self.current_img,self.visualizations[index])

        return temp

    def update_images(self,img,img_cam):

        # configure visualization image
        self.unresized_vis_for_select = PIL.Image.fromarray(resize(img_cam, 1), 'RGB')  # dont scale this
        # configure cam image
        np_org = np.array(self.unresized_vis_for_select)[:, :, 0:3]
        np_org[self.current_mask] = selection_rgb_color
        self.cam_pil = PIL.Image.fromarray(resize(np_org, self.factor), 'RGB')

        if (len(img.shape) == 3 and img.shape[2] == 3):
            self.rgb = True
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'RGB')
            self.unresized_image_for_select = PIL.Image.fromarray(resize(img, 1), 'RGB')  # dont scale this

            # configure select image
            np_org = np.array(self.unresized_image_for_select)[:, :, 0:3]
            np_org[np.bitwise_not(self.current_mask)] = 0
            select_pil = PIL.Image.fromarray(resize(np_org,self.factor), 'RGB')


        else: #GRAYSCALE
            self.rgb = False
            self.img_pil = PIL.Image.fromarray(resize(img,self.factor), 'L')
            self.unresized_image_for_select = PIL.Image.fromarray(resize(img, 1), 'L') # dont scale this

            # configure select image
            np_org = np.array(self.unresized_image_for_select)[:, :]
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

    def update_model(self,current_model):
        self.model = current_model
        self.n_clases = self.model.get_n_classes()

        self.current_index = self.model.get_current_index()
        img,all_cams,scores,r_label = self.model.get_img_cam_index(self.current_index)

        # add real label
        self.title_img_or.set("Img org. Real label {0}".format(r_label))

        # Save reference to all visualizations
        self.current_img = img
        # aplica colormaps a cams
        self.visualizations = {i : cv2.cvtColor(cv2.applyColorMap(all_cams[i], cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB) for i in range(len(all_cams))}



        # configure options select
        self.list_cam['menu'].delete(0, 'end')
        labels = ["class: {0} score: {1}".format(i,scores[i]) for i in range(self.n_clases)]
        self.cam_selected.set('')
        self.list_cam['menu'].delete(0, 'end')
        for i in range(self.n_clases):
            self.list_cam['menu'].add_command(label=labels[i], command=self.change_selected_list(self.cam_selected,labels[i],i))

        # Get mask and configure images
        current_index = 0
        self.current_mask = self.model.get_mask(self.current_index)
        self.cam_selected.set(labels[current_index])
        self.update_images(self.current_img, self.visualizations[current_index])


    def reset_mask(self):
        img_cam = np.array(self.unresized_image_for_select)
        self.current_mask = np.zeros((img_cam.shape[0],img_cam.shape[1])).astype(np.bool)
        self.draw_square((0,0,0,0))


    def commit_mask_changes(self):
        print('Commiting mask changes for index: {0}'.format(self.current_index))
        updated_mask = self.current_mask

        self.controller.event_draw_mask(updated_mask)


if __name__ == '__main__':
    img_or = np.random.rand(200, 200, 3)
    cam_or = np.random.rand(200, 200).astype(np.uint8)
    call_one_use_select(cam_or,img_or=img_or)
