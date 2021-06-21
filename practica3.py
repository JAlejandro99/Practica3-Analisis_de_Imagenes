import cv2
import copy

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
    img=cv2.imread('manzana.jpg')
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
    
    #Binariza la imagen del canal más conveniente con el método de umbralización seleccionado
    adaptativa(g)
    otsu(g)
    
    #Transformada hit or miss
    grises=cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)#Si se pasa la imagen binarizada, no funciona
    
    kernel = np.array((
        [1, 1, 1],
        [0, 1, -1],
        [0, 1, -1]), dtype="int")
    
    trans_img=cv2.morphologyEx(grises, cv2.MORPH_HITMISS, kernel)
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
    
#Método Otsu
def otsu(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Imagen en escala de grises
    ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Metodo Otsu',th2)
    cv2.waitKey(0)
