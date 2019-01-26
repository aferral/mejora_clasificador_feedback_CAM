"""
Tener herramienta interactiva de imagenes, indexar imagenes que evalue cambio de probs y cam.
Esta herramienta selecciona trozos en imagenes

# Cargar modelo - dataset.

# Generar lista dado indexacion o iteracion batches

# Por cada elemento mostrar 3 ventanas
# ventana imagen, ventana cam + imagen , ventana de seleccion

# botones SIG, ANTERIOR,

MODULOS

Modulo central requiere config-file con path a (dataset, model)
-- Al iniciar carga algunas imagenes, modelo0
-- Se preocupa de cargar mascara binario por defecto o a armar una
-- Carga lista de indices y manda a modulo
-- Carga lista de mascaras y revisa que indices estan asociados


1. Modulo carga de modelo(carpeta_models -- config_file?? ajustar tupla dataset - model)
    -- SOlo hace de interfaz visual respecto a carga de modulo central
    -- Lista de tuplas dataset,modelo

2. Modulo muestra imagen - cam - select (img,img_cam,mascar_si_existe)
    -- Si detecta cambio en mascara hace callback de update y eso lo ve central

3. Ajuste de parametros


4. Lista de indices o batch. Muestra si esta asignada mascara
    -- Solamente muestra lista y actua de callback a central


5. Seleccion de donde guardar o abrir archivo de mascaras
    -- Muestra lista de archivos de mascara
    -- Actua de callback al seleccionar uno


1. Tener claro que va pasar con los indices y datasets
Creo que todos los dataset deberia tenre
    -- List indexs
    -- Get index especifico




"""
# todo cambiar path a relativos en select export
# todo agregar informacion extra en interfaz indexs
# todo rename de mask files
# todo anotar descripcion en mask file??
# todo revisar tipos de objetos
# todo requiero limpiar un modelo no en uso ????
from tkinter import Widget, Frame, Toplevel, Label, Entry, Button, RIGHT
from tkinter import Tk, Listbox, END, mainloop
import numpy as np
from tkinter import LEFT,StringVar
import os
from classification_models.classification_model import Abstract_model
from datasets.dataset import Dataset
from select_tool.img_selector import img_selector
from select_tool.m_file_parser import model_manager_obj, get_config_file_list, config_folder
from utils import now_string
import json

class mask_select(Frame):
    def __init__(self, controller, current_model : model_manager_obj, master, *args, **kwargs):
        self.controller = controller


        super().__init__(master, *args, **kwargs)
        self.listbox = Listbox(self, exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>",self.selection)
        Label(self,text="New mask file: ").pack()
        self.entry_text = StringVar()
        Entry(self,textvariable=self.entry_text).pack()
        Button(self,text='Create',command=self.add_new).pack()

        self.update_model(current_model)


    def update_model(self, current_model):
        self.model = current_model
        self.mask_file_list = self.model.get_mask_file_list()
        self.current_mask_file=self.model.get_current_mask_file()

        self.listbox.delete(0, END)
        for item in self.mask_file_list:
            self.listbox.insert(END, item)
        self.listbox.select_set(0)
        self.listbox.event_generate("<<ListboxSelect>>")

    def selection(self,event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        selected = value
        self.controller.event_change_mask_file(selected)
        pass

    def add_new(self):
        new = self.entry_text.get()
        self.controller.event_add_mask_file(new)
        pass

class index_select(Frame):
    def __init__(self,controller,current_model,master,*args,**kwargs):
        self.controller = controller
        self.current_model = current_model

        from tkinter import EXTENDED,Scrollbar,Y

        #dibujar widget
        super().__init__(master, *args, **kwargs)

        f_st=Frame(self)
        Label(f_st,text="Current_index: ").pack()
        Label(f_st, text="Current_filter: ").pack()
        f_st.pack()

        frame_index_listbox = Frame(self)
        self.listbox = Listbox(frame_index_listbox, exportselection=False,selectmode=EXTENDED)
        self.listbox.pack(side=LEFT)

        scrollbar = Scrollbar(frame_index_listbox)
        scrollbar.pack(side=LEFT, fill=Y)
        frame_index_listbox.pack()


        # attach listbox to scrollbar
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        f=Frame(self)
        Label(f,text="Filtro: ").pack(side=LEFT)
        Entry(f).pack(side=LEFT)
        f.pack()

        f2=Frame(self)
        Button(f2, text='Filter').pack(side=LEFT)
        Button(f2,text='Clean Filter').pack(side=LEFT)
        Button(f2, text='Export sel for gen',command=self.export_selection).pack(side=LEFT)
        f2.pack()

        f3=Frame(self)
        Button(self, text='<<',command=self.next_prev(-1)).pack(side=LEFT)
        Button(self, text='>>',command=self.next_prev(1)).pack(side=LEFT)
        f3.pack()

        self.update_model(current_model)
        self.listbox.select_set(0)
        self.listbox.event_generate("<<ListboxSelect>>")
        self.listbox.bind('<<ListboxSelect>>', self.selection)

    def export_selection(self):
        sel_files_folder = os.path.join('config_files','select_files')
        os.makedirs(sel_files_folder,exist_ok=True)

        sel_list = self.listbox.curselection()
        index_list = [self.listbox.get(ind) for ind in sel_list]
        print("Exporting list for image generator. List: {0}".format(index_list))
        now_s = now_string()
        out_selection_file = {'index_list' : index_list,
                              'train_result_path': self.current_model.current_config_file,
                              'details' : '',
                              'mask_file' : self.current_model.current_mask_file}
        sel_file_name = "{0}_{1}_{2}_selection.json".format(self.current_model.classifier_key,self.current_model.dataset_key,now_s)
        sel_path = os.path.join(sel_files_folder,sel_file_name)
        with open(sel_path,'w') as f:
            json.dump(out_selection_file,f)
        print("Select in {0}".format(sel_path))

    def update_model(self, current_model):
        self.model = current_model
        self.index_list = self.model.get_index_list()
        self.current_index = self.model.get_current_index()
        self.mask_list = self.model.get_current_mask_index_list()


        self.listbox.delete(0, END)
        for item in sorted(self.index_list):
            self.listbox.insert(END, item)

    def next_prev(self,x):
        def selection():
            ind_l = self.index_list.index(self.current_index)
            n = len(self.index_list)
            n_ind_l = (ind_l+x) % n
            next = self.index_list[n_ind_l]
            self.current_index = next
            self.listbox.selection_clear(0, END)
            self.listbox.select_set(n_ind_l)
            self.controller.event_change_index(next)

        return selection

    def selection(self,event):
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print("v: {0}".format(value))
        selected_index = value
        self.current_index = selected_index
        self.controller.event_change_index(selected_index)
        pass


class params_select:
    def __init__(self,controller,current_model):
        self.controller = controller
        self.current_model = current_model



class model_select(Frame):
    def __init__(self,controller,model_list,current_model,master,*args,**kwargs):
        self.controller = controller
        self.model_list = model_list
        self.current_model = current_model

        # Dibujar widget
        super().__init__(master, *args, **kwargs)
        self.listbox = Listbox(self, exportselection=False)
        self.listbox.bind('<<ListboxSelect>>', self.selection)
        self.listbox.pack()

        for item in self.model_list:
            self.listbox.insert(END, item)

        self.listbox.select_set(0)
        self.listbox.event_generate("<<ListboxSelect>>")

    def selection(self,x):
        w = x.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print(value)

        # Send message to interface to load another model
        f_path = os.path.join(config_folder, value)
        self.current_model.change_model(f_path)
        self.controller.event_change_model(self.current_model)
        pass




class controller:
    def __init__(self, default_model_obj,model_file_list):

        all_models=model_file_list


        assert(len(all_models) > 0)


        # variables of the controller
        self.current_model = default_model_obj
        self.model_list = all_models


        # modules

        self._params_selector = params_select(self, self.current_model)



        root =Tk()

        w1 = root
        w1.title("Model_selector")
        self._model_chooser = model_select(self, all_models, self.current_model,w1)
        self._model_chooser.pack(side="top", fill="both", expand=True)

        w2 = Toplevel(root)
        w2.title("Img_selector")
        self._img_selector = img_selector(self, self.current_model, w2)
        self._img_selector.pack(side="top", fill="both", expand=True)

        w3 = Toplevel(root)
        w3.title("Index_selector")
        self._index_selector = index_select(self, self.current_model,w3)
        self._index_selector.pack(side="top", fill="both", expand=True)

        w4 = Toplevel(root)
        w4.title("Mask_selector")
        self._mask_f_select = mask_select(self, self.current_model, w4)
        self._mask_f_select.pack(side="top", fill="both", expand=True)

        root.mainloop()



    def event_change_model(self,selected_model):
        print("Changing current {0} for {1}".format(self.current_model,selected_model))
        self.current_model = selected_model

        # Reload
        self._img_selector.update_model(self.current_model)
        self._index_selector.update_model(self.current_model)
        self._mask_f_select.update_model(self.current_model)

    def event_draw_mask(self, mask):
        index = self.current_model.get_current_index()
        self.current_model.set_mask(index,mask)
        self._index_selector.update_model(self.current_model)


    def event_change_index(self, selected_index):
        self.current_model.set_current_index(selected_index)
        # update img-selector
        self._img_selector.update_model(self.current_model)

    def event_change_mask_file(self, selected_mask_file):
        self.current_model.set_mask_file(selected_mask_file)

        #update img,index selector
        self._img_selector.update_model(self.current_model)
        self._index_selector.update_model(self.current_model)

    def event_add_mask_file(self, new_mask_file):
        self.current_model.add_mask_file(new_mask_file)
        self._mask_f_select.update_model(self.current_model)


    def event_params_change(self, selected_model):
        pass

# Load tuples (dataset, model) in config file

config_list = get_config_file_list()
print(config_list)
f_path = os.path.join(config_folder,config_list[9])
print("Using: {0}".format(f_path))

with model_manager_obj(f_path) as model_manager:
    central = controller(model_manager,config_list)



