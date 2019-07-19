# -*- coding: utf-8 -*-

"""
Este código esta basado en el curso de DeepLearning del profesor Andrew Ng
"""

import sklearn
import sklearn.datasets
import numpy as np
import pandas as pd
from random import shuffle
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
    

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b2", ..., "WL-1", "bL":
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    b(l+1) -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### ( 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l-1],layers_dims[l]) * 0.01
        parameters['b' + str(l+1)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l-1],layers_dims[l]))
        assert(parameters['b' + str(l+1)].shape == (layers_dims[l], 1))

        
    return parameters
    
    
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)# integer representing the number of layers
     
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l-1],layers_dims[l]) * (2/layers_dims[l])**(0.5)
        parameters['b' + str(l+1)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters

    

def predict(train_X, train_Y, test_X, test_Y, parameters):
    """
    model evaluation for training set and testing set
    """
    
    # model evaluation for training set
    y_train_predict, caches =  L_model_forward(train_X, parameters)
    rmse = (np.sqrt(mean_squared_error(train_Y, y_train_predict)))
    r2 = r2_score(train_Y, y_train_predict)
    
    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 is {}'.format(r2))
    print("\n")
    
    # model evaluation for testing set
    y_train_predict, caches =  L_model_forward(test_X, parameters)
    rmse = (np.sqrt(mean_squared_error(test_Y, y_train_predict)))
    r2 = r2_score(test_Y, y_train_predict)
    
    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 is {}'.format(r2))
    
    return y_train_predict

    
def load_dataset():
    boston_dataset = sklearn.datasets.load_boston()    
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)    
    boston['MEDV'] = boston_dataset.target
    
    #Normalizando caracteristicas y salidas
    for k in boston.keys():
        maxim = max(boston[k])
        boston[k] = boston[k]/maxim 
        
    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
    Y = boston['MEDV']
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state=5)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y).reshape(404,1)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y).reshape(102,1)
    return train_X.T, train_Y.T, test_X.T, test_Y.T


def inicializar_Parametros(n_x, n_h, n_y):
    """
    Esta función da valores aleatorios a una red neuronal perceptron multicapa con una arquitectura de
    3 capas (1 capa de entrada con n_x neuronas, 1 capa oculta con n_h neuronas, 1 capa de salida con n_y neuronas)
    
    Argumentos:
    n_x -- Número de neuronas de la capa de entrada
    n_h -- Número de neuronas de la capa de oculta
    n_y -- Número de neuronas de la capa de salida
    
    Returna:
        
    parametros --  Diccionario de Python que contiene a:
                    W1 -- Matriz de pesos de tamaño (n_x, n_h)
                    b2 -- Vector bias de tamaño (n_h, 1)
                    W2 -- Matriz de pesos de tamaño(n_h, n_y)
                    b3 -- Vector bias de tamaño  (n_y, 1)
    """
       
    np.random.seed(1)
    
    W1 = np.random.randn(n_x, n_h)*0.01
    b2 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_h, n_y)*0.01
    b3 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_x, n_h))
    assert(b2.shape == (n_h, 1))
    assert(W2.shape == (n_h, n_y))
    assert(b3.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b2": b2,
                  "W2": W2,
                  "b3": b3}
    
    return parameters     

#parameters = inicializar_Parametros(3,2,1)
#print("W1 = " + str(parameters["W1"]))
#print("b2 = " + str(parameters["b2"]))
#print("W2 = " + str(parameters["W2"]))
#print("b3 = " + str(parameters["b3"]))


# =================================== Resultado de inicializar_Parametros =========
# 
# W1 = [[ 0.01624345 -0.00611756]
#  [-0.00528172 -0.01072969]
#  [ 0.00865408 -0.02301539]]
# b2 = [[ 0.]
#  [ 0.]]
# W2 = [[ 0.01744812]
#  [-0.00761207]]
# b3 = [[ 0.]]
# =============================================================================

#
def inicializar_parametros_profundos(layer_dims):
    """
    Argumenots:
    layer_dims -- Array de Python que contiene las dimensiones de cada capa de la red (arquitectura de la red)
    
    Returna:
    parameters -- Diccionario de Python que contiene a: "W1", "b2", ..., "WL-1", "bL":
                  Wl -- Matriz de pesos de tamaño (layer_dims[l-1], layer_dims[l])
                  b(l+1) -- Vector bias de tamaño (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        ### Haga su código acá ###  (≈ 2 lines of code)
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l-1],layer_dims[l]) * 0.01
        parameters['b' + str(l+1)] = np.zeros((layer_dims[l], 1))

        ### Fin ###
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l-1],layer_dims[l]))
        assert(parameters['b' + str(l+1)].shape == (layer_dims[l], 1))

        
    return parameters


#parameters = inicializar_parametros_profundos([5,4,3])
#print("W1 = " + str(parameters["W1"]))
#print("b2 = " + str(parameters["b2"]))
#print("W2 = " + str(parameters["W2"]))
#print("b3 = " + str(parameters["b3"]))


# ================================= Resultados de inicializar_parametros_profundos=======

# W1 = [[ 0.01788628  0.0043651   0.00096497 -0.01863493]
#  [-0.00277388 -0.00354759 -0.00082741 -0.00627001]
#  [-0.00043818 -0.00477218 -0.01313865  0.00884622]
#  [ 0.00881318  0.01709573  0.00050034 -0.00404677]
#  [-0.0054536  -0.01546477  0.00982367 -0.01101068]]
# b2 = [[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]]
# W2 = [[-0.01185047 -0.0020565   0.01486148]
#  [ 0.00236716 -0.01023785 -0.00712993]
#  [ 0.00625245 -0.00160513 -0.00768836]
#  [-0.00230031  0.00745056  0.01976111]]
# b3 = [[ 0.]
#  [ 0.]
#  [ 0.]]
# =============================================================================


def linear_forward(A, W, b):
    """
    Implementa la parte lineal de la propagación hacia adelante

    Arguments:
    A -- activationes de la capa anterior (o de los datos de entrada): (tamaño de la capa anterior, número de ejemplos)
    W -- Matriz de pesos: Matriz de tamaño (Tamaño de la capa anterior , Tamaño de la capa actual)
    b -- Vector bias, Vector de tamaño (Tamaño de la capa actual, 1)

    Returns:
    Z -- el nivel de activación Z de la capa actual.
    cache -- Diccionario de Python que contiene a: "A", "W" and "b" ; son almacenados para hallar el backward de cada capa
    """
    
    ### Haga su código acá ###(≈ 1 line of code) 
    
    Z = np.dot(W.T,A)+b    # usar np.dot()
    
    ### EFIN ###
    
    assert(Z.shape == (W.T.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

#A, W, b = linear_forward_test_case()
#Z, linear_cache = linear_forward(A, W, b)
#print("Z = " + str(Z))

# ============================Results of linear_forward========================
# Z = [[ 3.26295337 -1.23429987]]
# =============================================================================


#
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implementa la propagación hacia adelante para la capa LINEAR->ACTIVATION (activaciòn de la capa)

    Arguments:
    A_prev -- activaciones de la capa anterior (o de los datos de entrada): (tamaño de la capa anterior, número de ejemplos)
    W -- Matriz de pesos: Matriz de tamaño (Tamaño de la capa anterior , Tamaño de la capa actual)
    b -- Vector bias, Vector de tamaño (Tamaño de la capa actual, 1)
    activation -- La función de activaciòn que será usada en esta capa, string: "sigmoid" o "relu"

    Returns:
    A -- La salida de la función de activación
    cache --Diccionario de Python que contiene a: "linear_cache" y "activation_cache";
            son almacenados para hallar el backward de cada capa
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        ### Haga su código acá ### (≈ 2 lines of code)
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
        ### FIN ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
         ### Haga su código acá ### (≈ 2 lines of code)
         
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
        ### FIN ###
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Z, Z
        
    assert (A.shape == (W.shape[1], A.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#A_prev, W, b = linear_activation_forward_test_case()
#A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
#print("With sigmoid: A = " + str(A))
#
#A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
#print("With ReLU: A = " + str(A))
#

# ====================Results of linear_activation_forward ====================
# With sigmoid: A = [[ 0.96890023  0.11013289]]
# With ReLU: A = [[ 3.43896131  0.        ]]
# =============================================================================


def L_model_forward(X, parameters):
    """
    Implementa toda la propagación hacia adelante de la red para una arquitectura
    [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    
    Arguments:
    X -- datos de entrada, Array de tamaño (tamaño de entrada, número de ejemplos)
    parameters -- Salida de la función  of inicializar_parametros_profundos()
    
    Returns:
    AL -- Valores de la función de activación de las neuronas de la capa de salida
    caches -- Lista de caches:
              cada cache de linear_activation_forward() (hay L-1 caches, indexados de 0 a L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2 #W+b   # number of layers - 1 en la red
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        
        ### Haga su código acá ### (≈ 2 lines of code)
        
        A, cache = A, cache =  linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l+1)], "relu")
        caches.append(cache)

        ### FIN ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
   
    ### Haga su código acá ### (≈ 2 lines of code)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L+1)], None)
    caches.append(cache)
    
    ## FIN ###

    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

#X, parameters = L_model_forward_test_case_2hidden()
#AL, caches = L_model_forward(X, parameters)
#print("AL = " + str(AL))
#print("Length of caches list = " + str(len(caches)))

# ===============Results of  L_model_forward =========================
# AL = [[ 0.03921668  0.70498921  0.19734387  0.04728177]]
# Length of caches list = 3
# =============================================================================

def compute_cost(AL, YS, costFuntion):
    """
    Implementa la función de costo: cross-entropy cost(logistic Regression) o Median Squared Error

    Arguments:
    AL -- vector de probabilidades que corresponde a la predicción, tamaño (1, número de ejemplos)
    YS -- Vector de etiquetas deseadas (ejemplo: conteniendo 0 y 1), tamaño (1, número de ejemplos)

    Returna:
    cost -- cross-entropy cost o Median Squared Error
    """
    
    m = YS.shape[1]

    # Halla la perdida desde AL and YS.
    
    ### Haga su código acá ### (≈ 4 lines of code) usar np.multiply, np.sum, np.log
    
    if costFuntion == "LG":
    
        cost = -np.sum(np.multiply(np.log(AL),YS) + np.multiply(np.log(1-AL),(1-YS)))/m
        
    elif costFuntion == "MSE":
        
        cost = (np.sum((AL-YS)*(AL-YS))/2)/m
        
    
    ### FIN ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

#YS, AL = compute_cost_test_case()
#
#print("cost LG= " + str(compute_cost(AL, Y, "LG")))
#print("cost MSE= " + str(compute_cost(AL, Y, "MSE")))


# ================Results of  compute_cost==================================
# cost LG= 0.414931599615
# cost MSE= 0.0683333333333
# =============================================================================

# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implementa la parte de propagación hacia atrás para una sola capa l

    Argumentos:
    dZ -- Gradiente del costo con respecto a la salida lineal de la capa actual l
    cache -- tupla: (A_prev, W, b) que llega desde la propagación hacia adelante en la capa actual

    Returna:
    dA_prev -- Gradiente del costo con respecto a la activación (de la capa anterior l-1), el tamaño es el mismo de A_prev
    dW -- Gradiente del costo con respecto a  W (capa actual l), el tamaño es el mismo de W
    db -- Gradiente del costo con respecto a b (capa actual l),  el tamaño es el mismo de b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### Haga su código acá ### (≈ 3 lines of code)
    
    dW = np.dot(A_prev,dZ.T)/m   # usar np.dot
    db = np.sum(dZ,axis=1,keepdims=True)/m   
    dA_prev = np.dot(W,dZ)
    
    ### FIN ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

#dZ, linear_cache = linear_backward_test_case()
#dA_prev, dW, db = linear_backward(dZ, linear_cache)
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db))

# ==================Results of  linear_backward ===============================
# dA_prev = [[ 0.51822968 -0.19517421]
#  [-0.40506361  0.15255393]
#  [ 2.37496825 -0.89445391]]
# dW = [[-0.10076895]
#  [ 1.40685096]
#  [ 1.64992505]]
# db = [[ 0.50629448]]
# =============================================================================
#
def linear_activation_backward(dA, cache, activation):
    """
    Implementa la propagación hacia atrás backward-propagation para la capa LINEAR->ACTIVATION.
    
    Arguments:
    dA -- gradient de la activación de la capa l 
    cache -- Tupla:  (linear_cache, activation_cache)
    activation -- El tipo de función de activación a usar en esta capa, string: "sigmoid" or "relu"
    
    Returna:
    dA_prev -- Gradiente del costo con respecto a la activación (de la capa anterior l-1), el tamaño es el mismo de A_prev
    dW -- Gradiente del costo con respecto a  W (capa actual l), el tamaño es el mismo de W
    db -- Gradiente del costo con respecto a b (capa actual l),  el tamaño es el mismo de b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        ### Haga su código acá ###  (≈ 2 lines of code)
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
        ### FIN ###
        
    elif activation == "sigmoid":
        
         ### Haga su código acá ###  (≈ 2 lines of code)
        
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    else:
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### FIN ###
    return dA_prev, dW, db

#dAL, linear_activation_cache = linear_activation_backward_test_case()
#
#dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
#print ("sigmoid:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db) + "\n")
#
#dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
#print ("relu:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db))


# ==============Results of  linear_backward====================================
# sigmoid:
# dA_prev = [[ 0.11017994  0.01105339]
#  [ 0.09466817  0.00949723]
#  [-0.05743092 -0.00576154]]
# dW = [[ 0.10266786]
#  [ 0.09778551]
#  [-0.01968084]]
# db = [[-0.05729622]]
# 
# relu:
# dA_prev = [[ 0.44090989 -0.        ]
#  [ 0.37883606 -0.        ]
#  [-0.2298228   0.        ]]
# dW = [[ 0.44513824]
#  [ 0.37371418]
#  [-0.10478989]]
# db = [[-0.20837892]]
# =============================================================================
#
#
def L_model_backward(AL, YS, caches, costFuntion):
    """
   Implementa la propagación hacia atrás backward-propagation para toda la red [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID
    
    Arguments:
    AL -- vector de probabilidades que corresponde a la predicción, tamaño (1, número de ejemplos)
    YS -- Vector de etiquetas deseadas (ejemplo: conteniendo 0 y 1), tamaño (1, número de ejemplos)
    
    caches -- lista of caches que contiene a:
              cada cache de linear_activation_forward() con "relu" (caches[l], for l in range(L-1) i.e l = 0...L-2)
              la cache de linear_activation_forward() con "sigmoid" (caches[L-1])
    
    Returns:
    grads -- Un diccionario con los gradientes
             grads["dA" + str(l+1)] = ... 
             grads["dW" + str(l+1)] = ...
             grads["db" + str(l+2)] = ... 
    """
    grads = {}
    L = len(caches)+1   # número de capas
    m = AL.shape[1]
    YS = YS.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    
    ### Haga su código acá ### (4 line of code)
    
    if costFuntion == "LG":
    
        dAL = - (np.divide(YS, AL) - np.divide(1 - YS, 1 - AL)) # derivative of cost with respect to AL
        
    elif costFuntion == "MSE":
        
        dAL = - (YS-AL)
        
    ### FIN ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL-1"], grads["dbL"]
    
    ### Haga su código acá ### (approx. 2 lines)
    
    current_cache = caches[L-2]
    grads["dA" + str(L-1)], grads["dW" + str(L-1)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = None)
    
    ### FIN ###
    
    # Loop from l=L-3 to l=0
    
    for l in reversed(range(L-2)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], current_cache". Outputs: "grads["dA" + str(l+1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 2)] 
        
        ### Haga su código acá ### (approx. 5 lines)
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp =  linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 2)] = db_temp 
        
        ### FIN ###

    return grads

#AL, Y_assess, caches = L_model_backward_test_case()
#grads = L_model_backward(AL, Y_assess, caches,"LG")
#print("grads LG")
#print ("dW1 = "+ str(grads["dW1"]))
#print ("db2 = "+ str(grads["db2"]))
#print ("dA2 = "+ str(grads["dA2"])) 
#
#grads = L_model_backward(AL, Y_assess, caches,"MSE")
#print("grads MSE")
#print ("dW1 = "+ str(grads["dW1"]))
#print ("db2 = "+ str(grads["db2"]))
#print ("dA2 = "+ str(grads["dA2"])) 


# ============Results of  L_model_backward======================================
# grads LG
# dW1 = [[ 0.41010002  0.          0.05283652]
#  [ 0.07807203  0.          0.01005865]
#  [ 0.13798444  0.          0.01777766]
#  [ 0.10502167  0.          0.0135308 ]]
# db2 = [[-0.22007063]
#  [ 0.        ]
#  [-0.02835349]]
# dA2 = [[ 0.12913162 -0.44014127]
#  [-0.14175655  0.48317296]
#  [ 0.01663708 -0.05670698]]
# grads MSE
# dW1 = [[ 0.10087189  0.          0.01299615]
#  [ 0.0192033   0.          0.00247412]
#  [ 0.03393989  0.          0.00437275]
#  [ 0.02583208  0.          0.00332816]]
# db2 = [[-0.05413055]
#  [ 0.        ]
#  [-0.00697408]]
# dA2 = [[-0.18214833 -0.10826111]
#  [ 0.19995659  0.11884557]
#  [-0.02346765 -0.01394816]]
# =============================================================================

#
def update_parameters(parameters, grads, learning_rate):
    """
    Actualizar los parametros usando la regla del descenso del gradiente
    
    Argumentos:
    parameters -- Diccionario en Python conteniendo los parameters 
    grads -- Diccionario en Python conteniendo los gradientes, salida de L_model_backward
    
    Returns:
    parameters -- Diccionario en Python conteniendo los parameters actualizados 
                  parameters["W" + str(l+1)] = ... 
                  parameters["b" + str(l+2)] = ...
    """
    
    L = len(parameters) // 2 + 1 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    
    ### Haga su código acá ###  (≈ 3 lines of code)
    
    for l in range(L-1):
        parameters["W" + str(l+1)] -= learning_rate*grads["dW" + str(l + 1)]
        parameters["b" + str(l+2)] -= learning_rate*grads["db" + str(l + 2)]
        
    ### FIN ###
    
    return parameters

#parameters, grads = update_parameters_test_case()
#parameters = update_parameters(parameters, grads, 0.1)
#
#print ("W1 = "+ str(parameters["W1"]))
#print ("b2 = "+ str(parameters["b2"]))
#print ("W2 = "+ str(parameters["W2"]))
#print ("b3 = "+ str(parameters["b3"]))

# =================Results of  update_parameters================================
# W1 = [[-0.59562069 -1.76569676 -1.0535704 ]
#  [-0.09991781 -0.80627147 -0.86128581]
#  [-2.14584584  0.51115557  0.68284052]
#  [ 1.82662008 -1.18258802  2.20374577]]
# b2 = [[-0.04659241]
#  [-1.28888275]
#  [ 0.53405496]]
# W2 = [[-0.55569196]
#  [ 0.0354055 ]
#  [ 1.32964895]]
# b3 = [[-0.84610769]]
# =============================================================================



  