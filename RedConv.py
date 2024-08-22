import tensorflow as tf
import numpy
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#NORMALIZANDO
x = x / 255.0
y = numpy.array(y)

neuronas = [64]
densas = [2]
convpoo = [1]
drop = [0]


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape = x.shape[1:], activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
#model.add(tf.keras.layers.Dropout(0.2)) # SI TENEMOS OVERFITING

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation = "relu"))

model.add(tf.keras.layers.Dense(1,activation = "sigmoid"))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"]
              )

model.fit(x,y,batch_size = 32, epochs = 5, validation_split = 0.3)

for neurona in neuronas:
    for conv in convpoo:
        for densa in densas:
            for d in drop:
                NAME = "RedConv_n{}_cl{}_d{}_dropout{}".format(neurona,conv,densa,d)
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'logs/{}'.format(NAME))

                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Conv2D(neurona,(3,3), input_shape = x.shape[1:], activation = "relu"))
                model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

                if d == 1:
                    model.add(tf.keras.layers.Dropout(0.2))
                
                for i in range(conv):
                    model.add(tf.keras.layers.Conv2D(neurona, (3,3)))
                    model.add(tf.keras.layers.Activation("relu"))
                    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

                model.add(tf.keras.layers.Flatten())

                for i in range(densa):
                    model.add(tf.keras.layers.Dense(neurona))
                    model.add(tf.keras.layers.Activation("relu"))

                model.add(tf.keras.layers.Dense(1))
                model.add(tf.keras.layers.Activation('sigmoid'))

                model.compile(loss = "binary_crossentropy",
                              optimizer = "adam",
                              metrics = ["accuracy"])
                
                model.fit(x,y,batch_size = 32, epochs = 10, validation_split = 0.3, callbacks = [tensorboard])
                model.save("src/models/model.h5")

