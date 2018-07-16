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


TODO:
- MANEJAR FACTOR de resize de imagen en interfaz
- Manejar movimiento entre imagen con teclas??




"""


# todo revisar tipos de objetos
# todo requiero limpiar un modelo no en uso ????
from tkinter import Widget, Frame, Toplevel, Label, Entry, Button
from tkinter import Tk, Listbox, END, mainloop

class selected_model_obj:
    def __init__(self):
        all_indexs = ['index0', 'index1', 'index2']
        all_mask_files = ['mask_f0', 'mask_f1',
                          'mask_f2']  # TODO ASOCIAR A MODELOS
        assert (len(all_indexs) > 0)
        assert (len(all_mask_files) > 0)

        self.dataset_model_tuple = None
        self.dataset_obj = None

        self.current_index = all_indexs[0]
        self.index_list = None
        self.current_mask_file = None
        self.current_mask_list = []
        self.mask_file_list = None

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, index):
        # check if index existe in index_list
        assert(index in self.index_list)
        self.current_index = index

    def get_index_list(self):
        return self.index_list

    def get_mask_list(self):
        return self.current_mask_list

    def get_current_mask_file(self):
        return self.current_mask_file

    def set_mask_file(self, mask_file):
        # check if mask_file in list
        assert(mask_file in self.mask_file_list)
        self.current_mask_file = mask_file

    def get_mask_file_list(self):
        return self.mask_file_list

    def add_mask_file(self, name):

        self.mask_file_list.append(name)
        # todo crear achivo a disco


    def get_img_index(self, index):
        # check if index existe in index_list
        assert(index in self.index_list)
        return self.dataset_obj.get_index(index)

    def get_img_cam(self, index):
        # todo
        return None

    def get_mask(self, index):
        # todo asegurate de crearla en caso de que no exista.
        return None

    def set_mask(self, index, mask):
        # todo asegurate de crearla en caso de que no exista.
        return None




class mask_select(Frame):
    def __init__(self,controller,current_model : selected_model_obj,master,*args,**kwargs):
        self.controller = controller
        self.update_model(current_model)

        super().__init__(master, *args, **kwargs)
        listbox = Listbox(self)
        listbox.pack()

        listbox.insert(END, "LISTA_MASK_SELECT")

        for item in ["one", "two", "thr"
                                   "ee", "four"]:
            listbox.insert(END, item)

    def update_model(self, current_model):
        self.model = current_model
        self.mask_file_list = self.model.get_mask_file_list()
        self.current_mask_file=self.model.get_current_mask_file()


    def selection(self):
        selected = None
        self.controller.event_change_mask_file(selected)
        pass

    def add_new(self):
        new = None
        self.controller.event_add_mask_file(new)
        pass

class index_select(Frame):
    def __init__(self,controller,current_model,master,*args,**kwargs):
        self.controller = controller

        self.update_model(current_model)

        #dibujar widget
        super().__init__(master, *args, **kwargs)
        listbox = Listbox(self)
        listbox.pack()

        listbox.insert(END, "LISTA_INDEX_SELECT")

        for item in ["one", "two", "thr"
                                   "ee", "four"]:
            listbox.insert(END, item)

    def update_model(self, current_model):
        self.model = current_model
        self.index_list = self.model.get_index_list()
        self.current_index = self.model.get_current_index()
        self.mask_list = self.model.get_mask_list()



    def selection(self):
        selected_index = None
        self.controller.event_change_index(selected_index)
        pass


class params_select:
    def __init__(self,controller,current_model):
        self.controller = controller
        self.current_model = current_model



class img_select(Frame):
    def __init__(self,controller,current_model,master,*args,**kwargs):
        self.controller = controller
        self.update_model(current_model)

        # Dibujar widget
        super().__init__(master, *args, **kwargs)

        Label(self, text="Img").grid(row=0, column=0)
        Label(self, text="CAM").grid(row=0, column=1)
        Label(self, text="Mask").grid(row=0, column=2)

        Entry(self).grid(row=1, column=0)
        Entry(self).grid(row=1, column=1)
        Entry(self).grid(row=1, column=2)

        Button(self, text="Delete").grid(row=2, column=0)
        Button(self, text="Save").grid(row=2, column=1)



    def update_model(self,current_model):
        self.model = current_model
        self.current_index = self.model.get_current_index()
        img=self.model.get_img_index(self.current_index)
        img_cam = self.model.get_img_cam(self.current_index)
        mask = self.model.get_mask(self.current_index)




    def commit_mask_changes(self):

        #tomar mascara nueva
        updated_mask = None

        # editar archivo de mascaras
        self.controller.event_draw_mask(updated_mask)


class model_select(Frame):
    def __init__(self,controller,model_list,current_model,master,*args,**kwargs):
        self.controller = controller
        self.model_list = model_list
        self.current_model = current_model


        # Dibujar widget
        super().__init__(master, *args, **kwargs)
        listbox = Listbox(self)
        listbox.pack()

        listbox.insert(END, "a list entry")

        for item in ["one", "two", "thr"
                                   "ee", "four"]:
            listbox.insert(END, item)

    def selection(self):
        selected = None

        # Send message to interface to load another model
        self.controller.event_change_model(selected)
        pass



# Estado del controlador
# lista_modelos, lista indices**, lista_mascaras, modelo_activo, indice_activo,mascara_file_selected

class controller:
    def __init__(self):

        # todo logica de objeto o usar strings dict o que cosa_????
        all_models=['cifar10-vgg','imagenet-vgg']



        assert(len(all_models) > 0)


        # variables of the controller
        self.current_model = selected_model_obj() #todo temp
        self.model_list = all_models


        # modules

        self._params_selector = params_select(self, self.current_model)



        root =Tk()

        w1 = root
        self._model_chooser = model_select(self, all_models, self.current_model,w1)
        self._model_chooser.pack(side="top", fill="both", expand=True)

        w2 = Toplevel(root)
        self._img_selector = img_select(self, self.current_model, w2)
        self._img_selector.pack(side="top", fill="both", expand=True)

        w3 = Toplevel(root)
        self._index_selector = index_select(self, self.current_model,w3)
        self._index_selector.pack(side="top", fill="both", expand=True)

        w4 = Toplevel(root)
        self._mask_dataset = mask_select(self, self.current_model,w4)
        self._mask_dataset.pack(side="top", fill="both", expand=True)

        root.mainloop()



    def event_change_model(self,selected_model):
        print("Changing current {0} for {1}".format(self.current_model,selected_model))
        self.current_model = selected_model

        # Reload
        self._img_selector.update_model(self.current_model)
        self._index_selector.update_model(self.current_model)
        self._mask_dataset.update_model(self.current_model)

    def event_draw_mask(self, mask):
        index = self.current_model.get_current_index()
        self.current_model.set_mask(index,mask)


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


    def event_params_change(self, selected_model):
        pass

# Load tuples (dataset, model) in config file

# Obtener indices desde
central = controller()



