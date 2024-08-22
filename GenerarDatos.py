import os 
import cv2
import random as rn
from tqdm import tqdm
import numpy as np
import pickle


def Generar_Datos():

    data = []
    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR,categoria)
        valor = CATEGORIAS.index(categoria)
        listdir = os.listdir(path)

        for i in tqdm(range(len(listdir)),desc = categoria):
            imagen_nombre = listdir[i]
            try:
                imagen_ruta = os.path.join(path,imagen_nombre)
                imagen = cv2.imread(imagen_ruta,cv2.IMREAD_GRAYSCALE)# SOLO ESCALA DE GRISES
                imagen = cv2.resize(imagen,(IMAGE_SIZE,IMAGE_SIZE))
                # plt.imshow(imagen, cmap="gray") # PARA MOSTRAR IMAGENESS
                #plt.show()
                data.append([imagen,valor])
            except Exception as e:
                pass
    rn.shuffle(data) #BARAJANDO LOS DATOS

    x = []
    y = []

    for i in tqdm(range(len(data)), desc= "Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    x = np.array(x).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)

    pickle_out = open("x.pickle","wb")
    pickle.dump(x,pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()


CATEGORIAS = ['cats','dogs']
IMAGE_SIZE = 150

if __name__ == "__main__":
    DATADIR = 'src/data/train'
    Generar_Datos()