# -*- coding: utf-8 -*-

"""
Este código esta basado en el curso de DeepLearning del profesor Andrew Ng
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import pickle
from dnn_app_utils_v4_full import *
from dnn_app_utils_v4_full import load_dataset

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


#train_x_orig, train_y, test_x_orig, test_y = load_dataset_mnist() 
#train_x_orig, train_y, test_x_orig, test_y = load_dataset_mnist_2classes()     # Cargar la base de datos de MNIST
train_x_orig, train_y, test_x_orig, test_y, classes = load_data_aplication()  # Cargar la base de datos de la aplicación

# Visualizar uno de los ejemplos de la base de datos
index = 11                              # ejemplo 11
plt.figure(0)
#plt.imshow(train_x_orig[index])         # visualiza ejemplo

## Obtener número de ejemplos de entrenamiento y test
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
#
#
# imprimir características de la base de datos
print ("Número de ejemplos de entrenamiento: " + str(m_train))
print ("Número de ejemplos de testing: " + str(m_test))
print ("Tamaño de cada imagen: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
#
#
#
# Vectorizar los ejemplos de train y test 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
##
## Normalizar los valores de las características entre 0-1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
##
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

### Definiendo la Arquitectura de la Red ####

n_x = train_x.shape[0]     # num_px * num_px * 3
n_y = train_y.shape[0]

layers_dims = (n_x, 10, 10, n_y) # modelo de 4 capas

   

def L_layer_model(X, YS, layers_dims, learning_rate = 0.075, num_iterations = 1000, print_cost=False):
    """
    Implementa una red neuronal de L-capas: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Argumentos:
    X -- Datos de Entrada de tamaño (n_x, número de ejemplos)
    YS -- Vector de etiquetas deseadas (ejemplo: conteniendo 0 y 1), tamaño (1, número de ejemplos)
    layers_dims -- dimensiones de las capas (n_x, n_h, n_y)
    num_iterations -- Número de iteraciones del ciclo de optimizacióm
    learning_rate -- factor de aprendizaje de la regla del descenso del gradiente
    print_cost -- Si es True, se imprimira el costo cada 100 iteraciones
    
    Returna:
    parameters -- Diccionario de parámetros aprendidos por el modelo. Estos son usados en la predicción    """    
    
    print(X.shape)
    np.random.seed(1)
    costs = []                         # Almacenar el costo para cada iteración
    
    # inicialización de Parámetros. (≈ 1 line of code)
       
    parameters = inicializar_parametros_profundos(layers_dims)
      
    ### Fin ###
    
    # Ciclo del descenso del gradiente
    
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        #### Haga su código acá ### (≈ 1 line of code)

        AL, caches = L_model_forward(X,parameters)
               
        ### Fin ###
        
        # Calcular la función de Costo
        
        ### Haga su código acá ### (≈ 1 line of code)
        
        cost = compute_cost(AL,YS,"LG")
        
        ### Fin ###
    
        # backward propagation:
    
        #### Haga su código acá ### (≈ 1 line of code)
    
        grads = L_model_backward(AL,YS,caches,"LG")
        
        ### Fin ###
 
        # Actualizar Parámetros.
        
        ### Haga su código acá ###  (≈ 1 line of code)

        parameters = update_parameters(parameters,grads,learning_rate)
        
        ### Fin ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.figure(1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    
    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1000, print_cost = True)


###### Predicción Gestos ################

#### Haga su código acá ### 

my_image = "A (2).jpg" # Cambiar imagen 
#my_image = "D (17).jpg" # Cambiar imagen
my_label_y = [0] # Clase verdadera

img = cv2.imread(my_image)
img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
imagen_flatten = img.reshape(train_x_orig.shape[1]*train_x_orig.shape[2]*train_x_orig.shape[3], -1)

## Fin ##

####################################

###### Predicción MNIST ################

#### Haga su código acá ### 


## Fin ##

#################################


img_x = imagen_flatten/255.

my_predicted_image = predict(img_x, my_label_y, parameters)

plt.figure(2)
plt.imshow(img)
print ("y = " + str(np.squeeze(my_label_y)) + ", your L-layer model predicts a \"" + str(int(np.squeeze(my_predicted_image))) +  "\" picture.")

#pickle.dump( parameters, open( "saveParameters_4capas_10neuronas_0_075lr.p", "wb" ) )
#pG= pickle.load( open( "saveParameters_4capas_10neuronas_0_075lr.p", "rb" ) )

