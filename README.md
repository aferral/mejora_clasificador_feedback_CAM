# Trabajo

Este trabajo propone un método para mejorar clasificadores ya entrenados mediante la utilización de feedback de un experto. Para lograr tal objetivo se realiza un proceso de 3 etapas visualización, selección y reajuste. 

**Visualización:** Corresponde a observar lo que el modelo considera importante, de forma de buscar errores. La visualización implementada corresponde al algoritmo CAM (Class Activation Mapping).

**Selección:**  Si el experto al observar las visualizaciones encuentra que el modelo se esta centrando en áreas irrelevantes puede seleccionarlas para su posterior reajuste.

**Reajuste:** Corresponde a modificar los parámetros de la red con el objetivo de eliminar activaciones en áreas consideradas como irrelevantes. Se espera que este cambio mejore la generalización de modelo, ya que las nuevas características del modelo cumplen tanto con la clasificación como con el filtro visual del experto.

El proceso de reajuste se puede realizar de dos formas:

1. **Reemplazos en áreas irrelevantes:** Todas las áreas irrelevantes se modifican en la imagen original. Luego este nuevo conjunto de imágenes se utiliza en un nuevo proceso de backpropagation por 100 iteraciones. Dentro de estos métodos se analiza utilizar modelos generativos, recortes aleatorios y dropout selectivo.


2. **Reajuste según función de perdida:** También se plantea modificar la función de perdida para que la optimización elimine automáticamente las activaciones irrelevantes. Para esto se plantea una función llamada PASA (Perdida por Activación Selectiva Ajustada) en esta se penaliza mediante una mascara binaria toda activación en el área irrelevante, pero se pondera este termino por la probabilidad de la clase objetivo. Se razona que se debe penalizar el área irrelevante, pero a medida que aumenta la certeza de la predicción debe bajar la influencia.

# Dependencias
Este código necesita python 3.6 y las dependencias especificadas en el archivo requirements.txt. Para instalar todas las dependencias se recomienda utilizar un nuevo ambiente de desarrollo, lo cual puede realizarse mediante:
```
virtualenv -p python3.6 venv # crear un nuevo ambiente de desarrollo para python3.6 en la carpeta venv
source venv/bin/activate # activar el ambiente de desarrollo
pip install -r requirements.txt   # Instala todas las depedencias especificadas
(Ejecutar experimentos)
deactivate # Desabilita el ambiente de desarrollo una vez se deja de usar
```

# Nota sobre ejecución

Todos los scripts de este proyecto asumen que la estructura base parte en la raíz del directorio, por lo tanto la forma de llamar un script es mediante módulos es decir:
```
python -m  datasets.imagenet_data # ESTO SI
python datasets/imagenet_data.py # ESTO NO
```

# Como conseguir los datos

__Forma fácil: Es posible descargar todos los datos contenidos en este zip (). Descomprimir los contenidos en una carpeta llamada temp. Los contenidos descomprimidos utilizan 37 G de espacio.__

**IMAGENET subset dataset:** Ejecutar el script en tf_records_parser lo cual descargara el subconjunto de imágenes y los procesara en tf_records que leen los clasificadores. Con esto el dataset ya esta habilitado.

**Sketchs dataset:** Es necesario seguir el proceso detallado en estos archivo (https://github.com/aferral/Recuperacion_sketchs/blob/master/preproceso.py) con esto se genera los tf_records. Luego colocar ./temp/tf_records_quickdraw . 

**Simbols dataset:** Se requiere seguir el proceso detallado en ./datasets/simple_figures.py

**Otros datasets:** Se encuentran datasets extras, los cuales no se usaron en todos los experimentos. Mayores detalles en cifar10_data.py , cwr_dataset.py.


# Estructura del código

## Classification_models
Agrupa las diversas arquitecturas a utilizar. Es necesario múltiples modelos debido a que cambian las dimensiones por cada dataset y además cuando es necesario agregar capas. Los clasificadores más importantes son imagenet_subset_conv_loss.py que define la función PASA y classification_model.py que define la base para todos los clasificadores.

## Tf_record parser
Para acelerar el proceso de los datos todos los dataset se convierten a un tipo de archivo llamado tf_records el cual esta optimizado para ejecutarse en tensorflow (librería de proceso de datos). En esta carpeta se encuentran los scripts que transforman los diversos datasets a tal formato. Todos están listos para ejecutarse mientas se suministren las carpetas con los datos (revisar primeras lineas de cada archivo para la ubicacion de las carpetas).


## Datasets
Se creo una clase especial para agrupar todas las funcionalidades necesarias para los datasets (./datasets/dataset.py). Los datasets específicos extienden esta clase definiendo la ubicación de los tf_records y otras funciones de ser necesarias.

## ImageNet_Utils
Repositorio extra necesario para descargar los datos de imagenet de forma automática.

## Select_tool
Agrupa todos los objetos necesarios para la interfaz de visualización y selección. Para desplegar la interfaz utilizar:
```
python -m select_tool.main
```

## Vis_exp
Genera diversas visualizaciones dado los datos procesados. Puede generar una visualización interactiva de una reducción de dimensionalidad de las imágenes, puede transformar todas las visualizaciones CAM en imágenes, puede producir visualización de cada filtro en especifico, etc. En esta carpeta es de gran importancia "visualization_exp.py" que contiene las visualizaciones generadas en el capitulo discusiones (visualización de activaciones dado todo el dataset).

## Config_files
En esta carpeta se guardan los json que representan los entrenamientos de los clasificadores. Se utilizan para agrupar parámetros en configuraciones. A medida que se entrenen modelos los resultados aparecerán en esta carpeta.

## Plot_utils
Toma los resultados de diversos experimentos y los agrega en tablas. Fue utilizado para generar las tablas en latex de forma automática.

## Image_generator
Funciones utilizadas para el método de reajuste por reemplazos. Para descargar el modelo generativo es necesario ejecutar deploy_generative_inpaiting.sh, el cual descarga el modelo generativo entregado por (https://github.com/JiahuiYu/generative_inpainting).
```
sh deploy_generative_model.sh
```

# Proceso para reproducir resultados

1. **Descarga de datos:** Revisar sección anterior "como conseguir los datos"
2. **Entrenar modelos iniciales:** Para entrenar el clasificador dado un dataset se debe escribir un archivo de configuración, ejemplos de estos se encuentran en "./config_files/train_files". Una vez seleccionado el archivo ejecutar:
```
python -m exp_scripts.do_interactive_exp <path_a_config_file>
```

3. **Localizar configuración de resultado:** El entrenamiento genera un modelo, el cual se guarda a disco con un archivo de configuración que lista todos los parámetros utilizados. Estos archivos de resultados se encuentran en "./config_files/train_results" 

4. **Seleccionar áreas irrelevantes:** Se selecciona las áreas irrelevantes mediante la interfaz. Para abrir el modelo entrenado utilizar el siguiente comando.
```
python -m select_tool.main <path_a_result_config>
```

5. **Exportar selección:** Para exportar las selecciones de la interfaz es necesario presionar "export sel for gen". Esto generara dos archivo uno con todos los parametros de la selección en "./config_files/select_files" y su mascara en "./config_files/mask_files". 

6. **Configurar reajuste:** Con estos parámetros solo queda configurar la forma del reajuste. Esto se realiza en "./exp_scripts/exp_backprops.py" donde es necesario editar las últimas lineas con todos los parámetros detallados anteriormente. En estas mismas lineas se elige el tipo de ajuste a realizar. Una vez configurado realizar:
```
python -m exp_scripts.exp_backprops
```

7. **Visualizar resultados:** Los resultados del experimento del paso anterior se encuentran en "./out_backprops/<nombre_exp>". Dentro de esta carpeta se encuentran visualizaciones de cada CAM por iteración, la accuracy en cada iteración y las matrices de confusión. Además se grafican las imágenes con errores antes y después del método.
# Trabajo

Este trabajo propone un método para mejorar clasificadores ya entrenados mediante la utilización de feedback de un experto. Para lograr tal objetivo se realiza un proceso de 3 etapas visualización, selección y reajuste. 

**Visualización:** Corresponde a observar lo que el modelo considera importante, de forma de buscar errores. La visualización implementada corresponde al algoritmo CAM (Class Activation Mapping).

**Selección:**  Si el experto al observar las visualizaciones encuentra que el modelo se esta centrando en áreas irrelevantes puede seleccionarlas para su posterior reajuste.

**Reajuste:** Corresponde a modificar los parámetros de la red con el objetivo de eliminar activaciones en áreas consideradas como irrelevantes. Se espera que este cambio mejore la generalización de modelo, ya que las nuevas características del modelo cumplen tanto con la clasificación como con el filtro visual del experto.

El proceso de reajuste se puede realizar de dos formas:

1. **Reemplazos en áreas irrelevantes:** Todas las áreas irrelevantes se modifican en la imagen original. Luego este nuevo conjunto de imágenes se utiliza en un nuevo proceso de backpropagation por 100 iteraciones. Dentro de estos métodos se analiza utilizar modelos generativos, recortes aleatorios y dropout selectivo.


2. **Reajuste según función de perdida:** También se plantea modificar la función de perdida para que la optimización elimine automáticamente las activaciones irrelevantes. Para esto se plantea una función llamada PASA (Perdida por Activación Selectiva Ajustada) en esta se penaliza mediante una mascara binaria toda activación en el área irrelevante, pero se pondera este termino por la probabilidad de la clase objetivo. Se razona que se debe penalizar el área irrelevante, pero a medida que aumenta la certeza de la predicción debe bajar la influencia.

# Dependencias
Este código necesita python 3.6 y las dependencias especificadas en el archivo requirements.txt. Para instalar todas las dependencias se recomienda utilizar un nuevo ambiente de desarrollo, lo cual puede realizarse mediante:
```
virtualenv -p python3.6 venv # crear un nuevo ambiente de desarrollo para python3.6 en la carpeta venv
source venv/bin/activate # activar el ambiente de desarrollo
pip install -r requirements.txt   # Instala todas las depedencias especificadas
(Ejecutar experimentos)
deactivate # Desabilita el ambiente de desarrollo una vez se deja de usar
```

# Nota sobre ejecución

Todos los scripts de este proyecto asumen que la estructura base parte en la raíz del directorio, por lo tanto la forma de llamar un script es mediante módulos es decir:
```
python -m  datasets.imagenet_data # ESTO SI
python datasets/imagenet_data.py # ESTO NO
```

# Como conseguir los datos

__Forma fácil: Es posible descargar todos los datos contenidos en este zip (). Descomprimir los contenidos en una carpeta llamada temp. Los contenidos descomprimidos utilizan 37 G de espacio.__

**IMAGENET subset dataset:** Ejecutar el script en tf_records_parser lo cual descargara el subconjunto de imágenes y los procesara en tf_records que leen los clasificadores. Con esto el dataset ya esta habilitado.

**Sketchs dataset:** Es necesario seguir el proceso detallado en estos archivo (https://github.com/aferral/Recuperacion_sketchs/blob/master/preproceso.py) con esto se genera los tf_records. Luego colocar ./temp/tf_records_quickdraw . 

**Simbols dataset:** Se requiere seguir el proceso detallado en ./datasets/simple_figures.py

**Otros datasets:** Se encuentran datasets extras, los cuales no se usaron en todos los experimentos. Mayores detalles en cifar10_data.py , cwr_dataset.py.


# Estructura del código

## Classification_models
Agrupa las diversas arquitecturas a utilizar. Es necesario múltiples modelos debido a que cambian las dimensiones por cada dataset y además cuando es necesario agregar capas. Los clasificadores más importantes son imagenet_subset_conv_loss.py que define la función PASA y classification_model.py que define la base para todos los clasificadores.

## Tf_record parser
Para acelerar el proceso de los datos todos los dataset se convierten a un tipo de archivo llamado tf_records el cual esta optimizado para ejecutarse en tensorflow (librería de proceso de datos). En esta carpeta se encuentran los scripts que transforman los diversos datasets a tal formato. Todos están listos para ejecutarse mientas se suministren las carpetas con los datos (revisar primeras lineas de cada archivo para la ubicacion de las carpetas).


## Datasets
Se creo una clase especial para agrupar todas las funcionalidades necesarias para los datasets (./datasets/dataset.py). Los datasets específicos extienden esta clase definiendo la ubicación de los tf_records y otras funciones de ser necesarias.

## ImageNet_Utils
Repositorio extra necesario para descargar los datos de imagenet de forma automática.

## Select_tool
Agrupa todos los objetos necesarios para la interfaz de visualización y selección. Para desplegar la interfaz utilizar:
```
python -m select_tool.main
```

## Vis_exp
Genera diversas visualizaciones dado los datos procesados. Puede generar una visualización interactiva de una reducción de dimensionalidad de las imágenes, puede transformar todas las visualizaciones CAM en imágenes, puede producir visualización de cada filtro en especifico, etc. En esta carpeta es de gran importancia "visualization_exp.py" que contiene las visualizaciones generadas en el capitulo discusiones (visualización de activaciones dado todo el dataset).

## Config_files
En esta carpeta se guardan los json que representan los entrenamientos de los clasificadores. Se utilizan para agrupar parámetros en configuraciones. A medida que se entrenen modelos los resultados aparecerán en esta carpeta.

## Plot_utils
Toma los resultados de diversos experimentos y los agrega en tablas. Fue utilizado para generar las tablas en latex de forma automática.

## Image_generator
Funciones utilizadas para el método de reajuste por reemplazos. Para descargar el modelo generativo es necesario ejecutar deploy_generative_inpaiting.sh, el cual descarga el modelo generativo entregado por (https://github.com/JiahuiYu/generative_inpainting).
```
sh deploy_generative_model.sh
```

# Proceso para reproducir resultados

1. **Descarga de datos:** Revisar sección anterior "como conseguir los datos"
2. **Entrenar modelos iniciales:** Para entrenar el clasificador dado un dataset se debe escribir un archivo de configuración, ejemplos de estos se encuentran en "./config_files/train_files". Una vez seleccionado el archivo ejecutar:
```
python -m exp_scripts.do_interactive_exp <path_a_config_file>
```

3. **Localizar configuración de resultado:** El entrenamiento genera un modelo, el cual se guarda a disco con un archivo de configuración que lista todos los parámetros utilizados. Estos archivos de resultados se encuentran en "./config_files/train_results" 

4. **Seleccionar áreas irrelevantes:** Se selecciona las áreas irrelevantes mediante la interfaz. Para abrir el modelo entrenado utilizar el siguiente comando.
```
python -m select_tool.main <path_a_result_config>
```

5. **Exportar selección:** Para exportar las selecciones de la interfaz es necesario presionar "export sel for gen". Esto generara dos archivo uno con todos los parametros de la selección en "./config_files/select_files" y su mascara en "./config_files/mask_files". 

6. **Configurar reajuste:** Con estos parámetros solo queda configurar la forma del reajuste. Esto se realiza en "./exp_scripts/exp_backprops.py" donde es necesario editar las últimas lineas con todos los parámetros detallados anteriormente. En estas mismas lineas se elige el tipo de ajuste a realizar. Una vez configurado realizar:
```
python -m exp_scripts.exp_backprops
```

7. **Visualizar resultados:** Los resultados del experimento del paso anterior se encuentran en "./out_backprops/<nombre_exp>". Dentro de esta carpeta se encuentran visualizaciones de cada CAM por iteración, la accuracy en cada iteración y las matrices de confusión. Además se grafican las imágenes con errores antes y después del método.

