import os
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import random
from skimage.filters import threshold_otsu, rank

import cv2
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from utils import parse_config_recur
from select_tool.config_data import model_obj_dict, dataset_obj_dict
import tensorflow as tf


def get_model_data_from_config(config_file,use_this_model=None,use_this_m_class=None,dataset='train'):
    
    # read config_file
    data_config = parse_config_recur(config_file)
    t_params = data_config['train_params']
    m_k = data_config['model_key']
    m_p = data_config['model_params']
    d_k = data_config['dataset_key']
    d_p = data_config['dataset_params']
    
    batch_size = t_params['b_size'] if 'b_size' in t_params else 20
    epochs = t_params['epochs'] if 'epochs' in t_params else 1

    # Choose model to load
    if data_config['model_load_path_at_train_file'] is not None:
        model_load_path = data_config['model_load_path_at_train_file']
        print("Using model load path from TRAIN_FILE {0}".format(model_load_path))
    else:
        model_load_path = data_config['model_load_path_train_result']
        print("Using model load path from SELECT_FILE {0}".format(model_load_path))
    if use_this_model:
        model_load_path = use_this_model
        
        
    # open dataset and model class
    if use_this_m_class:
        model_class = use_this_m_class
    else:
        model_class = model_obj_dict[m_k]
         
    dataset_class = dataset_obj_dict[d_k]

    
    with tf.Graph().as_default():
        base_dataset = dataset_class(epochs, batch_size, **d_p)  # type: Dataset

        with model_class(base_dataset,**m_p) as model:
            model.load(model_load_path)

            if dataset == 'train':
                model.dataset.initialize_iterator_train(model.sess)
            elif dataset == 'val':
                model.dataset.initialize_iterator_val(model.sess)
            elif dataset == 'test':
                model.dataset.initialize_iterator_test(model.sess)
            else:
                raise Exception('Not recognized dataset {0}'.format(dataset))

            counter=0
            a_acts=[]
            a_vectors=[]
            a_targets=[]
            a_inds=[]

            while True:
                try:
                    fd = model.prepare_feed(is_train=False, debug=False)

                    t_s = [model.softmax_weights, model.last_conv,model.indexs,model.pred,model.targets]
                    # extract activations in batch, CAM
                    last_w,act_layer,index_batch,soft_max_out,targets = model.sess.run(t_s,feed_dict=fd)

                    gap_v = act_layer.mean(axis=(1,2))


                    a_acts.append(act_layer)
                    a_vectors.append(gap_v)
                    a_targets.append(targets)
                    a_inds.append(index_batch)


                    if counter % 50 == 0:
                        print(counter)
                    counter += 1

                except tf.errors.OutOfRangeError:
                    log = 'break at {0}'.format(counter)
                    break

            a_acts = np.concatenate(a_acts,axis=0)
            a_vectors = np.concatenate(a_vectors,axis=0)
            a_targets = np.concatenate(a_targets,axis=0)
            a_inds = [x.decode('utf8') for x in np.array(a_inds).flatten()]

        return a_acts,a_vectors,a_targets,a_inds,base_dataset,last_w


def calc_interval(ba, max_v,min_v,brackets):
    delta= (max_v-min_v)*1.0/brackets
    st=min_v+ba*delta
    en=min_v+(ba+1)*delta
    return st,en

def calc_pos(val,max_v,min_v,brackets):
    delta= (max_v-min_v)*1.0/brackets
    val_abs = val-min_v
    pos = val_abs // delta
    res = val_abs % delta
    pos = pos if res != 0 else pos-1 
    
    return brackets-1 - pos



font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.6
fontColor              = (255,255,255)
lineType               = 2


def val_analysis(model,val_data,val_t,print_r=True):
    preds = model.predict(val_data)
    
    if print_r:
        print(classification_report(val_t,preds))
    return accuracy_score(val_t,preds)

class plot_how:
    def __init__(self,max_v,img_per_b):
        fig=plt.figure(figsize=(20,20))
        self.ax = fig.add_subplot(111)
        self.ax.set_ylim(0,max_v*1.2)
        self.ax.set_xlim(-1,1)
        
        self.img_shape = (0.2, max_v*0.8*(1.0/img_per_b))
        
        pass
    
    def plot(self,img,val,act):
            x,y = random.random()-0.5,val
            
            w,h = self.img_shape
            
            bb = Bbox.from_bounds(x,y, w,h )  
            bb2 = TransformedBbox(bb,self.ax.transData)
            bbox_image = BboxImage(bb2,
                                norm = None,
                                origin=None,
                                clip_on=True)
            
            
            bbox_image.set_data(img)
            self.ax.add_artist(bbox_image)
    
    def end(self):
        pass

    
class plot_grid:
    def __init__(self,max_v,min_v,img_per_b,brackets,cols=1):
        
        self.nb = np.ceil(np.sqrt(img_per_b))**2
        self.sqb = sqb = np.sqrt(self.nb)
        
        self.n = self.nb * brackets
        
        self.cols = cols
        self.rows = brackets // self.cols
        
        self.wb=wb=200
        self.hb=hb=200
        
        self.wib = int(wb/sqb)
        self.hib = int(hb/sqb)
        
        self.added={}

        self.index_fun = lambda v : calc_pos(v,max_v,min_v,brackets)
        
        
        self.current=0
        
        wt = wb*self.rows
        ht = hb*self.cols
        self.canvas = np.zeros((wt,ht,3),dtype=np.uint8)
        
    def to_uint(img):
        if len(img.shape) > 2 and img.shape[-1] == 3 :
            delta = img.max(axis=(0,1)) - img.min(axis=(0,1))
            m = img.min(axis=(0,1))
        else:
            delta = img.max() - img.min()
            m = img.min()
            
        norm_img = (img.astype(np.float32)-m)*1.0 / delta
        out = (norm_img * 255).astype(np.uint8) 
        return out
        
    def get_img(self,img,act):
        return cv2.resize(img,(self.wib,self.hib))
    
    def plot(self,img,val,act):
        assert(self.current < self.n)
        
        img = img if img.dtype == np.uint8 else to_uint(img)
        
        # calc pos in grid
        ind=self.index_fun(val)
        xbase= ind // self.cols
        ybase = ind % self.cols
        
        local_i= self.current % self.nb
        
        # add what value was used in what block
        self.added.setdefault(int(ind),[]).append(val)
        
        # resize
        out_img = self.get_img(img,act)
        

        # draw
        xr= local_i // self.sqb
        yr= local_i % self.sqb
        
        
        deltaw = int(self.wb / self.sqb)
        deltah = int(self.hb / self.sqb)
        
        
        x0,xf = int(xbase * self.wb + xr * deltaw), int(xbase * self.wb + (xr+1) * deltaw) 
        y0,yf = int(ybase * self.hb + yr * deltah), int(ybase * self.hb + (yr+1) * deltah )
        
        self.canvas[x0:xf, y0:yf] = out_img

        self.current += 1

    def end(self,name):

        
        
        # draw grid lines
        line_width = 2
        line_color = (0,255,0)
        out_color = (255,0,0)
                
        w,h = self.canvas.shape[0:2]
        delta_w = int(w//self.cols)
        delta_h = int(h//self.rows)
        for i in range(1,self.cols):
            st=(delta_w*i)
            self.canvas[st:st+line_width,:] = line_color
        for i in range(1,self.rows):
            st=(delta_h*i)
            self.canvas[:,st:st+line_width] = line_color
        # outer lines
        self.canvas[0:line_width,:] = out_color
        self.canvas[:,0:line_width] = out_color
        self.canvas[-line_width:,:] = out_color
        self.canvas[:,-line_width:] = out_color
        
        # draw each bracket title
        for i in range(self.rows):
            for j in range(self.cols):
                ind = i*self.cols+j
                v_l = self.added[ind] if ind in self.added else None
                mean=sum(v_l)*1.0/len(v_l) if ind in self.added else None
                mean_str = "{0:.2f}".format(mean) if v_l is not None else 'NO VALUES' 
                
                xbase= ind // self.cols
                ybase = ind % self.cols
                deltaw = int(self.wb / self.sqb)
                deltah = int(self.hb / self.sqb)
                x0,y0 = int((xbase * self.wb)+deltaw),int(ybase*self.hb)
                
                cv2.putText(self.canvas,'VAL {0} - {1} '.format(mean_str,ind), 
                    (y0,x0), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
            


        return self.canvas
    
    

    
# plot only slices
class plot_grid_slice(plot_grid):
    def get_img(self,img,act):
        downsample=False
        
        if downsample:
            threshold_global_otsu = threshold_otsu(act)
            global_otsu = act >= threshold_global_otsu
            img_r = cv2.resize(img,(act.shape))
            img_r[~global_otsu] = 0  
            return cv2.resize(img_r,(self.wib,self.hib))
        else:
            threshold_global_otsu = threshold_otsu(act)
            global_otsu = act >= threshold_global_otsu
            img_r = cv2.resize(global_otsu.astype(np.uint8),(img.shape[:-1]))
            img[~(img_r.astype(np.bool))] = 0  
            return cv2.resize(img,(self.wib,self.hib))


def get_filter_vis(a_inds,a_vectors,a_acts,base_dataset,i,img_per_b=8,brackets=9):
    just_d = a_vectors[:,i]
    act_d = a_acts[:,:,:,i]

    # rank activations
    rank = just_d.argsort()
    max_v,min_v = just_d[rank[-1]]*1.02,just_d[rank[0]]

    #plt_c = plot_how(max_v,img_per_b)

    #plt_c = plot_grid(max_v,min_v,img_per_b,brackets)
    plt_c = plot_grid_slice(max_v,min_v,img_per_b,brackets,cols=3)


    for ba in reversed(range(brackets)):
        r0,r1 = calc_interval(ba, max_v,min_v,brackets)
        sel=np.where(np.bitwise_and((just_d <= r1), (just_d > r0) ))     
        n = sel[0].shape[0]
        r_sel=np.random.permutation(n)
        #print('f: {0} , n: {1}, min {2}, max {3} , r0: {4} , r1: {5}'.format(i,n,min_v,max_v,r0,r1))
        for it,x in enumerate(r_sel):
            if it > img_per_b:
                break
            sel_i = x

            ind_sel = a_inds[sel[0][sel_i]]
            val = just_d[sel[0][sel_i]]
            act = act_d[sel[0][sel_i]]
            # get image b
            img,label =base_dataset.get_train_image_at(ind_sel)
            img=img[0]

            plt_c.plot(img,val,act)

    return plt_c.end('F_{0}'.format(i))

# 1D analysis REQUIERE 
# option to retunr value
def draw_filters(a_vectors,a_acts,a_inds,base_dataset,img_per_b = 8,brackets = 9,split=True):
    seed=1
    np.random.seed(seed)
    r,c = 8,8
    whole_block = [[0 for j in range(c)]  for i in range(r)]
    n_dims = a_vectors.shape[1]

    for i in range(n_dims) :

        whole_block[i//r][i%c] = get_filter_vis(a_inds,a_vectors,a_acts,base_dataset,i,img_per_b=img_per_b,brackets=brackets)


    out_f='filtros_out'
    os.makedirs(out_f,exist_ok=True)
    if split:
        for i in range(r):
            for j in range(c):
                img=whole_block[i][j]
                ind=i*r+j
                out_path = os.path.join(out_f,'F_{0}.png'.format(ind))
                cv2.imwrite(out_path,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        out_path = os.path.join(out_f,'Filters.png')
        block_img = np.concatenate( [ np.concatenate(whole_block[i],axis=1) for i in range(len(whole_block)) ],axis=0)
        cv2.imwrite(out_path,cv2.cvtColor(block_img, cv2.COLOR_RGB2BGR))

def get_hist_vis(ax,a_vectors,a_targets,ind_feature,n_c,cl_names):

    for c in range(n_c):
        sel = (a_targets.argmax(axis=1) == c)
        ax.hist(a_vectors[sel,ind_feature],histtype='step',label=cl_names[c])
        ax.legend()
    return ax


def draw_hist(a_vectors,a_targets,cl_names):
    n_d = a_vectors.shape[1]
    fig, ax=plt.subplots(8,8, clear=True,figsize=(40,40))
    int_targets = a_targets.argmax(axis=1)
    n_c = len(set(a_targets.argmax(axis=1).tolist()))

    for i in range(n_d):
        ax[i//8][i%8] = get_hist_vis(ax[i//8][i%8],a_vectors,a_targets,i,n_c,cl_names)

    out_path = os.path.join('filtros_out','Histograms.png')
    plt.savefig(out_path, dpi=70, bbox_inches='tight')