# -*- coding: utf-8 -*-

"""
Este código esta basado en el curso de DeepLearning del profesor Andrew Ng
"""

import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v4_empty import *
from dnn_app_utils_v4_empty import load_dataset



train_x, train_y, test_x, test_y = load_dataset()

## Obtener número de ejemplos de entrenamiento y test
m_train = train_x.shape[0]
num_px = train_x.shape[1]
m_test = test_x.shape[0]
#
#
# imprimir características de la base de datos
print ("Número de ejemplos de entrenamiento: " + str(m_train))
print ("Número de ejemplos de testing: " + str(m_test))
print ("Tamaño de cada imagen: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))


### Definiendo la Arquitectura de la Red ####
n_x = train_x.shape[0]
n_y = train_y.shape[0]

layers_dims = (n_x, 20, 20, 20, n_y) # modelo de 4 capas

   

def L_layer_model(X, YS, layers_dims, learning_rate = 0.0075, num_iterations = 1000, print_cost=False):
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

        AL, caches = L_model_forward(X, parameters)
               
        ### Fin ###
        
        # Calcular la función de Costo
        
        ### Haga su código acá ### (≈ 1 line of code)
        
        cost = compute_cost(AL, YS, 'MSE')
        
        ### Fin ###
    
        # backward propagation: 
    
        #### Haga su código acá ### (≈ 1 line of code)
    
        grads = L_model_backward(AL, YS, caches, 'MSE')
        
        ### Fin ###
 
        # Actualizar Parámetros.
        
        ### Haga su código acá ###  (≈ 1 line of code)

        parameters = update_parameters(parameters, grads, learning_rate)
        
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

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)

#
#####################################
#
####### Predicción Regresión ################
my_predicted_target = predict(train_x, train_y, test_x, test_y, parameters)
