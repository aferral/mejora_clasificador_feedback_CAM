import os 
from collections import namedtuple
import pytesseract
import re
import cv2
from shutil import copyfile
import hashlib
import functools


folder_img_resources = 'img2'
os.makedirs(folder_img_resources,exist_ok=True)

vars_names = 'acc_val_start cnf_val_start acc_val_end cnf_val_end acc_test_start cnf_test_start acc_test_end cnf_test_end '

Data_val_test=namedtuple('Data_val_test', vars_names)


class Quote_this:
	def __init__(self,data):
		self.data = data
	def format(self):
		d = self.data if self.data != '' else '-'
		return r"\verb "+"{0} \n".format(d)

class Clasf_list:
	def __init__(self,data,bold_index=0):
		self.bold_index = bold_index
		self.data = data
	def format(self):
		temp_list = [r'\textbf{{{0}}}'.format(v) if self.bold_index  == i else str(v) for i,v in enumerate(self.data)]
		return '|'.join(temp_list)


class Cnf_matrix:
	def __init__(self,data):
		str_mat  = filter(lambda x : len(x) > 0 ,data.replace('[','').replace(']','').split('\n'))
		self.data = [list(map(int,r.strip().split())) for r in str_mat]
		self.r = len(data)
		self.c = len(data[0])

	def format(self):
		out=r"""\[ \begin{{vmatrix}} {0}  \end{{vmatrix}} \]"""
		f_data = r'\\'.join([' & '.join(map(str,d_row)) for d_row in self.data])
		return out.format(f_data)

class img_resource:
	def __init__(self,img_path,size='1',add_height=False):
		# copy img to folder
		self.size = size
		self.add_height = add_height
		img_unique_name = hashlib.md5(img_path.encode('utf-8')).hexdigest()+'.png'
		dest = os.path.join(folder_img_resources,img_unique_name)
		copyfile(img_path, dest)
		self.img_path = dest
		

	def format(self):
		wop = 'width={0}in'.format(self.size)
		hop = 'height=0.5in' if self.add_height else ''
		options = ','.join([wop,hop])
		out=r"\includegraphics[{1}]{{{0}}}"
		return out.format(self.img_path,options)


formateer = [Quote_this, Cnf_matrix,img_resource, Clasf_list]


def format_to_latex(data):
	rows,cols = len(data),len(data[0])

	t_descr = '|'+('l|'*cols)

	base = r"\begin{{table}}[] \centering \fontsize{{8}}{{12}} \selectfont {0} \end{{table}}"
	table_header = r'\hskip-1.1in\begin{{tabular}}{{ {t_descr} }} {t_body}  \end{{tabular}} \caption{{Table caption}}'
	body = r'\hline {0}'
	row = r"{0} \\ \hline"

	def data_handler(data):
		if type(data) == int:
			return str(data)
		elif type(data) == float:
			return str(data)
		elif type(data) == str:
			return data
		elif type(data) in formateer:
			return data.format()
		else:
			raise Exception('No handler for data {0} {1}'.format(type(data),data))

	all_rows = [row.format(' & '.join(map(data_handler,data_row))) for data_row in data]
	all_rows_f = '\n'.join(all_rows)

	body_f = body.format(all_rows_f)
	table_header_f = table_header.format(t_descr = t_descr, t_body=body_f)
	base_f = base.format(table_header_f)

	return base_f


def process_conf_txt(path):

	with open(path,'r') as f:
		text = f.read()
		lines = text.split('\n')
		acc = lines[0].split('Accuracy')[1]
		cnf_matrix = '\n'.join(lines[1:])
	return acc,cnf_matrix

def process_eval_file(path):


	with open(path,'r') as f:
		t=f.read()
	x=t.strip().split('\n\n \n \n')
	val_lines,test_lines = x[0::2],x[1::2]

	def process_st(st):
		lines = st.split('\n')
		ind_acc = list(filter(lambda t : 'set accuracy is' in t[1], enumerate(lines)))
		ind,line = ind_acc[0]
		acc=line.split('is')[1].strip()
		conf_matrix = '\n'.join(lines[ind+1:])
		return acc,conf_matrix


	acc_val_start, cnf_val_start = process_st(val_lines[0])
	acc_val_end, cnf_val_end = process_st(val_lines[-1])

	acc_test_start, cnf_test_start = process_st(test_lines[0])
	acc_test_end, cnf_test_end = process_st(test_lines[-1])

	out=Data_val_test(
		acc_val_start=acc_val_start,cnf_val_start=cnf_val_start,
		acc_val_end=acc_val_end,cnf_val_end=cnf_val_end,
		acc_test_start=acc_test_start,cnf_test_start=cnf_test_start,
		acc_test_end=acc_test_end,cnf_test_end=cnf_test_end
		)


	return out


def get_latex_from_config_file(conf_file_path, method_name, add_initial, sel_indexs):


	# carga confusion matrixs train, test, val
	path_conf_m_v_t = os.path.join(conf_file_path,'eval.txt')
	path_conf_t_s =  os.path.join(conf_file_path,'start','conf_matrix.txt')
	path_conf_t_e = os.path.join(conf_file_path,'end','conf_matrix.txt')

	assert(all([os.path.exists(p) for p in [path_conf_m_v_t,path_conf_t_s,path_conf_t_e] ]))



	# carga accuracy, confusion matrixs 
	d = process_eval_file(path_conf_m_v_t)

	acc_st_train, cnf_matrix_st_train = process_conf_txt(path_conf_t_s)
	acc_end_train, cnf_matrix_end_train = process_conf_txt(path_conf_t_e)

	acc_st_val, cnf_matrix_st_val = d.acc_val_start, d.cnf_val_start
	acc_end_val, cnf_matrix_end_val = d.acc_val_end, d.cnf_val_end

	acc_st_test, cnf_matrix_st_test = d.acc_test_start, d.cnf_test_start
	acc_end_test, cnf_matrix_end_test = d.acc_test_end, d.cnf_test_end



	data_for_table = []
	if add_initial:
		data_for_table.append(['Metodo','Train','Validation','Test',])
		data_for_table.append(['Base',acc_st_train,acc_st_val,acc_st_test])
	data_for_table.append([method_name,  acc_end_train, acc_end_val, acc_end_test])

	r,c = len(data_for_table), len(data_for_table[0])

	data_accuracy = data_for_table




	data_for_table = []
	if add_initial:
		header_row = 0
		data_for_table.append(['Metodo','Train','Validation','Test'])
		data_for_table.append(['Base',cnf_matrix_st_train,cnf_matrix_st_val,cnf_matrix_st_test])
	else:
		header_row = -1


	data_for_table.append([method_name,  cnf_matrix_end_train, cnf_matrix_end_val, cnf_matrix_end_test ])
	format_row = lambda row : [row[0],Cnf_matrix(row[1]),Cnf_matrix(row[2]),Cnf_matrix(row[3]) ]
	data_for_table = list(map(lambda x : x[1] if x[0] == header_row else format_row(x[1]) , enumerate(data_for_table) ))


	r,c = len(data_for_table), len(data_for_table[0])

	data_cnf= data_for_table




	# lee indices
	pngs = list(filter(lambda x : '.png' in x , os.listdir(conf_file_path)))
	indexs = set(list(map(lambda x : x.split('__')[0].replace('real_','') ,pngs)))

	# abre imagenes en original
	files_or = os.listdir(os.path.join(conf_file_path,'originals'))
	masks_or = list(filter(lambda x : '_mask.png' in x , files_or))
	imgs_or = list(filter(lambda x : not(x in masks_or) , files_or))

	all_imgs_data = {}
	for i_n, ind in enumerate(sorted(indexs)):

		def extract_class_from_img(img_path):
			temp_img=cv2.imread(img_path)
			t2=cv2.inRange(temp_img, (255, 255, 255), (255, 255,255))
			text = pytesseract.image_to_string(t2).replace('B','8')

			a=re.compile(r'\d.\d+'); 
			cl_text = a.findall(text)
			return cl_text


		# anota cam inicial / final
		lt=list(sorted(filter(lambda x : ind in x , pngs),key=lambda x : int(x.replace('.png','').split('it_')[1])))
		im0 = lt[0]
		imf = lt[-1]

		# anota probabilidades
		cam_ini_path = os.path.join(conf_file_path,im0)
		cam_fin_path = os.path.join(conf_file_path,imf)
		cl_text0 = extract_class_from_img(os.path.join(conf_file_path,im0))
		cl_textf = extract_class_from_img(os.path.join(conf_file_path,imf))

		index_img_or = list(filter(lambda x : ind in x, imgs_or))[0]
		index_mask_or = list(filter(lambda x : ind in x, masks_or))[0]

		data_d = {}

		data_d['Imagenes'] = img_resource(os.path.join(conf_file_path,'originals',index_img_or))
		data_d['Nombre'] = Quote_this(ind)
		data_d['Mascaras'] = img_resource(os.path.join(conf_file_path,'originals',index_mask_or))
		data_d['CAM inicial'] = img_resource(cam_ini_path,size=1.2,add_height=True)
		data_d['Clasf inicial'] = Clasf_list(cl_text0,bold_index=sel_indexs[i_n])
		data_d['Visualizaciones finales'] = '-'
		data_d[method_name] = img_resource(cam_fin_path,size=1.2,add_height=True)
		data_d['Clasf final'] =  Clasf_list(cl_textf,bold_index=sel_indexs[i_n])


		
		all_imgs_data[ind]=data_d

	if add_initial:
		keys= ['Imagenes','Nombre','Mascaras','CAM inicial','Clasf inicial','Visualizaciones finales',method_name,'Clasf final']
	else:
		keys= [method_name,'Clasf final']


	data_for_table = []

	indexs_order = sorted(indexs)
	for k in keys:
		row = [k]

		row_data = [all_imgs_data[ind][k] for ind in indexs_order]

		data_for_table.append(row+row_data)

	r,c = len(data_for_table),len(data_for_table[0])


	data_vis = data_for_table

	return data_accuracy, data_cnf , data_vis





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--confs', nargs='+',help='List of configs files to use')
parser.add_argument('--names', nargs='+',help='List of names (same order as confs)')
parser.add_argument('--indexs_labels', nargs='+',type=int,help='List of names (same order as confs)')
args= parser.parse_args()


confs = args.confs
nombres = args.names

indices_sel = args.indexs_labels


 

# consigue los datos en [(data_acc0, data_cnf0, data_vis0) ,(data_acc1, data_cnf1, data_vis1) ... ]
all_data = [get_latex_from_config_file(cnf_file, name, ind == 0 , indices_sel) for ind, (name, cnf_file)  in enumerate(zip(nombres,confs))]


for lista_lista_datos in zip(*all_data): # EL zip pasa a [(tupla con todas las data_acc), (tupla con todas las data_cnf), (tupla con todos los vis) ]
	# Ahora es necesario concadenar las listas 
	# en python [1,2] + [3,4] = [1,2,3,4]
	datos = functools.reduce(lambda x,y : x+y, lista_lista_datos)

	print(format_to_latex(datos))

	print(' ')
	print(' ')
	print(' ')
