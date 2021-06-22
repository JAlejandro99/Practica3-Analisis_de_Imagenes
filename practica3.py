import cv2
import copy
import numpy as np

def sep_canales(img_rgb):
    #Imagen en canal rojo
    r=copy.copy(img_rgb)
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    cv2.imshow('R-RGB', r)
    
    #Imagen en canal verde
    g=copy.copy(img_rgb)
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    cv2.imshow('G-RGB', g)
    
    #Imagen en canal azul
    b=copy.copy(img_rgb)
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    cv2.imshow('B-RGB', b)
    
    return r,g,b

def tarea_dos():
    img=cv2.imread('Imagen-letreros.jpg')
    img_gris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Imagen en escala de grises en BGR
    img_rgb= cv2.cvtColor(img_gris, cv2.COLOR_GRAY2RGB)#Convierte imagen en grises en RGB
    
    #Imagen en canal rojo
    r=copy.copy(img_rgb)
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    cv2.imshow('R-RGB', r)
    
    #Imagen en canal verde
    g=copy.copy(img_rgb)
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    cv2.imshow('G-RGB', g)
    
    #Imagen en canal azul
    b=copy.copy(img_rgb)
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    cv2.imshow('B-RGB', b)
    
    cv2.waitKey(0)
    
    #Binariza la imagen del canal más conveniente con el método de Otsu
    img_bin=otsu(g)
    
    #Transformada hit or miss
    kernel = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]), dtype="int")
    
    trans_img=cv2.morphologyEx(img_bin, cv2.MORPH_HITMISS, kernel)
    cv2.imshow('Transformada',trans_img)
    cv2.waitKey(0)

#Umbralización adaptativa
def adaptativa(img):
    #ADAPTIVE_THRESH_MEAN_C : el valor umbral es equivalente al valor del área vecina
    #ADAPTIVE_THRESH_GAUSSIAN_C : en este caso el valor umbral es la suma de los pesos
    #de los valores vecinos donde dichos valores correspondían a pesos de una ventana
    #gaussiana
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Imagen en escala de grises
    th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('Adaptativa MEAN',th2)
    cv2.imshow('Adaptativa GAUSSIAN',th3)
    cv2.waitKey(0)

    return th2,th3
    
#Método Otsu
def otsu(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Imagen en escala de grises
    ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2
    #cv2.imshow('Metodo Otsu',th2)
    #cv2.waitKey(0)

#Funcion que realiza la multiumbralizacion de una imagen
#Recibe como parametros la imagen y una lista con los umbrales que el usuario introduce
def multiumbralizacion(im,umbrales):
    
    #Verifica si la imagen tiene 3 canales RGB
    if(len(im.shape)==3):
        #Si los tiene la convierte a escala de grises con un solo canal
        im2=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    #Inicializa limite izquierdo en cero
    lim_izq = 0
    #Crea una lista vacía donde se guardan las imagenes
    objetos = [None]*(len(umbrales)+1)
    
    #Umbralizacion por cada objeto
    for i in range (0,len(umbrales)+1):
        if(i == len(umbrales)):
            #Establece limite derecho
            lim_der = 255
        else:
            #Establece limite derecho
            lim_der = umbrales[i]
            
        #Crea una nueva imagen para el objeto
        objetos[i] = copy.copy(im2)
        
        #Recorre los pixeles de la imagen
        for j in range(0,im2.shape[1]):
            for k in range(0,im2.shape[0]):
                if( im2[k,j] > lim_izq and im2[k,j] < lim_der ):
                    (objetos[i])[k][j] = 0
                else:
                    (objetos[i])[k][j] = 255
    
    """
    #Muestra objetos
    for img in objetos:
        cv2.imshow('Objeto', img)
        cv2.waitKey()
    """
    
    #Regresa los objetos
    return objetos