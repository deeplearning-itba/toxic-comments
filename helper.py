def get_memory_size(x):
    print("Size of the array: ",
        x.size)
    
    print("Memory size of one array element in bytes: ",
        x.itemsize)
    
    # memory size of numpy array in bytes
    print("Memory size of numpy array in GB:",
        x.size * x.itemsize / (1024**3))
    
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras import regularizers
from keras.layers import Activation
from keras import optimizers


def get_model(input_features, output_size, hidden_units=200, lr=0.001, lambd=0):
    default_initializer = initializers.glorot_uniform()
    model_sig_nn = Sequential()
    model_sig_nn.add(Dense(hidden_units,
                        input_dim=input_features, 
                        kernel_regularizer=regularizers.l2(lambd), 
                        kernel_initializer=default_initializer,
                        name="Capa_Oculta_1"))
    model_sig_nn.add(Activation('sigmoid'))
    model_sig_nn.add(Dense(hidden_units,
                        input_dim=input_features, 
                        kernel_regularizer=regularizers.l2(lambd), 
                        kernel_initializer=default_initializer,
                        name="Capa_Oculta_2"))
    model_sig_nn.add(Activation('sigmoid'))
    model_sig_nn.add(Dense(output_size,
                        kernel_regularizer=regularizers.l2(lambd), 
                        kernel_initializer=default_initializer,
                        name="Capa_Salida"))
    model_sig_nn.add(Activation('sigmoid', name="output")) 

    #selectedOptimizer = optimizers.SGD(lr=lr)
    selectedOptimizer = optimizers.Adam(learning_rate=lr, decay=0.001)

    model_sig_nn.compile(loss = 'binary_crossentropy', optimizer=selectedOptimizer, 
                        metrics=['accuracy']) #auc

    return model_sig_nn