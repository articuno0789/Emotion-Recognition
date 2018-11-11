#Proyecto: Reconocedor de emociones - Entrenador y clasificador de emociones
 
#Datos de la materia:
#SEMINARIO DE SOLUCION DE PROBLEMAS DE INTELIGENCIA ARTIFICIAL II - I7041
#Profesor VILLASEÑOR PADILLA, CARLOS ALBERTO
#Martes y Jueves de 07:00 - 08:55
#Laboratorio LC02 - DUCT1
#Sección D04 - NRC 140934
#Ciclo escolar 2018 B
#Datos del alumno:
#Castillo Serrano Cristian Michell
#215861738
#Ingeniería en computación (INCO) 
##########################################################################

#-------------------------------Importar paquetes------------------------------
#Importar modelos de clasificación
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, jaccard_similarity_score
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics
from scipy.stats import sem
import cv2
from scipy.ndimage import zoom
from sklearn import datasets
import itertools
import datetime
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from time import time #importamos la función time para capturar tiempos

#------------------------------------------------------------------------------


#--------------------------Variables de configuración--------------------------
archivoDeEntrenamiento = 1
#1) Base de datos "Olivetti faces dataset", esta base se descarga de la WEB, 
    #si no se tiene almacenada en el equipo.
#2) Base de datos Fer2013.
#Determinar si se desea gaurdar frame por frame las predicciones en tiempo real
#realizadas.
#Tipo de reconocimiento 1)Estático, 2)Tiempo real
tipoReco = 2
#Guardar reconocimiento en memoria secundaria
guardarReconocimiento = True
#Tamaño de la imagen del rostro a evaluar y con las que se entrena el modelo.
# El tamaño debe estar especificado en pixeles.
if archivoDeEntrenamiento == 1:
    anchoRostro = 64.0
    altoRostro = 64.0
else:
    anchoRostro = 48.0
    altoRostro = 48.0
#Tamaño de la consola en la cual imprimir los resultados
tamanoConsola = 50
#Utilizar PCA
utilizarPCA = True
_PCA = ""
#Clasificador a utilizar
clasificador = 4
#1) Nearest Neighbors
#2) Linear SVM
#3) RBF SVM
#4) Poly SVM
#5) Gaussian Process
#6) Decision Tree
#7) Random Forest
#8) Neural Net
#9) AdaBoost
#10) NaiveBayes
#11) QDA
nombreClasificador ="";
if clasificador==1:# Nearest Neighbors
    nombreClasificador ="Nearest Neighbors"
elif clasificador==2:#Linear SVM
    nombreClasificador ="Linear SVM"
elif clasificador==3:#RBF SVM
    nombreClasificador ="RBF SVM"
elif clasificador==4:#Poly SVM
    nombreClasificador ="Poly SVM"
elif clasificador==5:#Gaussian Process
    nombreClasificador ="Gaussian Process"
elif clasificador==6:#Decision Tree
    nombreClasificador ="Decision Tree"
elif clasificador==7:# Random Forest
    nombreClasificador ="Random Forest"
elif clasificador==8:#Neural Net
    nombreClasificador ="Neural Net"
elif clasificador==9:#AdaBoost
    nombreClasificador ="AdaBoost"
elif clasificador==10:#NaiveBayes
    nombreClasificador ="NaiveBayes"
elif clasificador==11:#QDA
    nombreClasificador ="QDA"
#------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#-----------------------------Funciones y clases------------------------------
#-----------------------------------------------------------------------------

def crear_directorio(ruta):
    """
    Crea un directorio nuevo de acuerdo al nombre especificado.
    Si existe el directorio deseado borra su contenido y lo crea de nuevo.
    """
    try:
        os.makedirs(ruta)
    except OSError:
        pass

def imprimirTextoCentrado(texto, tamanoTexto, relleno=" "):
    """
    Imprime el texto especificado de manera centra, flanqueando el texto
    con el carácter de relleno especificado.
    """
    print (texto.center(tamanoTexto, relleno)) 

# ===============================================================================
# Funciones para detección de rostros humanos en tiempo real
# from FaceDetectPredict.py
# ===============================================================================

def detectarCaras(frame):
    """
    Se encarga de detectar caras dentro de una imagen.
    """
    cascPath = "../recursos/haarcascades/haarcascade_frontalface_default.xml"
    #cascPath = "../recursos/haarcascades/haarcascade_frontalcatface.xml"
    #cascPath = "../recursos/haarcascades/haarcascade_frontalcatface_extended.xml"
    #cascPath = "../recursos/haarcascades/haarcascade_frontalface_alt.xml"
    #cascPath = "../recursos/haarcascades/haarcascade_frontalface_alt_tree.xml"
    #cascPath = "../recursos/haarcascades/haarcascade_frontalface_alt2.xml"
    
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cara_detectada = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE)
    return gray, cara_detectada


def extraer_caracteristicas_cara(gray, detected_face, offset_coefficients):
    """
    Se encarga de extraer las características de un rostro 
    con respecto a una imagen.
    """
    (x, y, w, h) = detected_face
    horizontal_offset = int(offset_coefficients[0] * w)
    vertical_offset = int(offset_coefficients[1] * h)
    cara_extraida = gray[y + vertical_offset:y + h,
                     x + horizontal_offset:x - horizontal_offset + w]
    nueva_cara_extraida = zoom(cara_extraida, (anchoRostro / cara_extraida.shape[0],
                                               altoRostro / cara_extraida.shape[1]))
    nueva_cara_extraida = nueva_cara_extraida.astype(np.float32)
    nueva_cara_extraida /= float(nueva_cara_extraida.max())
    #print(nueva_cara_extraida)
    if (utilizarPCA):
        nueva_cara_extraida = pca.transform(nueva_cara_extraida.reshape(1, -1))

    return nueva_cara_extraida

def predecir_emocion(cara_extraida):
    """
    Se encarga de evaluar dentro del modelo el patrón de prueba actual 
    y reportar el resultado obtenido.
    """
    prediccion = model.predict(cara_extraida.reshape(1, -1))
    return prediccion

def definir_Titulo_Emocion(frame, resultado_prediccion, x, y):
    #Escribir encima de la cara la emoción reconocida.
    if archivoDeEntrenamiento == 1:
        if resultado_prediccion == 1: #Sonreir
            cv2.putText(frame, "Sonriendo",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
        elif resultado_prediccion == 0: #No sonreir
            cv2.putText(frame, "No sonriendo",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        elif resultado_prediccion == 2: #Sorprendido
            cv2.putText(frame, "Sorprendido",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,000), 5)
        elif resultado_prediccion == 3: #Enojado
            cv2.putText(frame, "Enojado",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (28,84,45), 5)
        elif resultado_prediccion == 4: #Neutral
            cv2.putText(frame, "Neutral",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (33,33,33), 5)
    else:
        if resultado_prediccion == 1: #Disgustado
            cv2.putText(frame, "Disgustado",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
        elif resultado_prediccion == 0: #Enojado
            cv2.putText(frame, "Enojado",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        elif resultado_prediccion == 2: #Temor
            cv2.putText(frame, "Temor",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,000), 5)
        elif resultado_prediccion == 3: #Feliz
            cv2.putText(frame, "Feliz",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (28,84,45), 5)
        elif resultado_prediccion == 4: #Triste
            cv2.putText(frame, "Triste",(x,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (33,33,33), 5)
        elif resultado_prediccion == 5: #Sorprendido
            cv2.putText(frame, "Sorprendido",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (33,33,33), 5)
        elif resultado_prediccion == 6: #Neutral
            cv2.putText(frame, "Neutral",(x,y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (33,33,33), 5)
    return frame
#------------------------------------------------------------------------------

def reconocimiento_tiempo_real():
    global utilizarPCA
    ################# DETECCIÓN DE EMOCIONES EN TIEMPO REAL #################
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Se esta iniciando el reconocimiento facial en tiempo real."
    imprimirTextoCentrado(texto, tamanoConsola)
    imprimirTextoCentrado("", tamanoConsola,"-")
    tipo_extraccion = (0.3, 0.05)#(0.075, 0.05) (0.3, 0.05)
    tituloVentana = "Reconocedor de emociones - " + nombreClasificador
    noTerminarCaptura = True
    contador_caras = 0
    #Establecer el objeto de captura de vídeo
    capturado_Video = cv2.VideoCapture(0)
    while (noTerminarCaptura):
        # Capturar frame-por-frame
        ret, frame = capturado_Video.read()
        # Detectar las caras
        # En esta sección se le pasa a la función el frame capturado por la web-cam
        # entonces regresa una lista con la(s) cara(s) detectadas.
        # Cabe mencionar que se regresa la(s) cara(s) tanto en color como en
        # escala de grises.
        # Las imágenes en escala de grises son las que se utilizarán para realizar
        # las predicciones.
        gray, cara_detectada = detectarCaras(frame)

        #Cuentas el número de caras detectadas en un frame
        indice_caras = 0
        
        #Escribir en pantalla la forma de cerrar la aplicación
        cv2.putText(frame, "Presione 'esc' para salir.",
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 1)

        # Predecir salida para cada cara detectada
        for face in cara_detectada:
            (x, y, w, h) = face
            if w > 100:
                # Dibujar rectángulo alrededor de la cara
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame,(x+int(w/2), y+int(h/2)), 5, (255,0,255), -1)

                # Extraer características cara
                cara_extraida = extraer_caracteristicas_cara(gray, 
                                                             face, 
                                                             tipo_extraccion)
                # Predecir emoción
                resultado_prediccion = predecir_emocion(cara_extraida)
                #Reducir la dimensionalidad de el rostro identificado 
                if utilizarPCA:
                    utilizarPCA = False
                    cara_extraida = extraer_caracteristicas_cara(gray, 
                                                                 face, 
                                                                 tipo_extraccion)
                    utilizarPCA = True
                #Dibujar cara detectada en la esquina superior derecha
                frame[indice_caras * int(anchoRostro): (indice_caras + 1) * int(altoRostro), -int(anchoRostro+1):-1, :] = cv2.cvtColor(cara_extraida * 255, cv2.COLOR_GRAY2RGB)                
                #Escribir encima de la cara la emoción reconocida.
                frame = definir_Titulo_Emocion(frame, resultado_prediccion,
                                               x, y)
                #Contador de incrementos de caras en un frame
                indice_caras += 1
                #Contador de caras global
                contador_caras += 1
                if guardarReconocimiento:
                    cv2.imwrite(os.path.join(ruta , "Frame_" +\
                                             str(contador_caras) +\
                                             ".jpg"), frame)
        #Mostrar en pantalla, el reconocimiento resultante
        cv2.imshow(tituloVentana, frame)
        if cv2.waitKey(10) & 0xFF == 27:
            noTerminarCaptura = False
    # Cuando el proceso termina es necesario cerrar el flujo de datos de video
    capturado_Video.release()
    cv2.destroyAllWindows()

def reconocimiento_estatico(c1, c2, name):
    global utilizarPCA
    ################# DETECCIÓN DE EMOCIONES EN TIEMPO REAL #################
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Se esta iniciando el reconocimiento facial estático."
    imprimirTextoCentrado(texto, tamanoConsola)
    imprimirTextoCentrado("", tamanoConsola,"-")
    tipo_extraccion = (0.3, 0.05)#(0.075, 0.05) (0.3, 0.05)
    imagenOriginal = cv2.imread("../recursos/Imagenes_Pruebas/"\
                                            + str(name) + ".jpg")
    cv2.imshow("Imagen original", imagenOriginal)
    gray, cara_detectada = detectarCaras(imagenOriginal)
    indice_caras = 0
    for face in cara_detectada:
            (x, y, w, h) = face
            if w > 100:
                # Dibujar rectángulo alrededor de la cara
                cv2.rectangle(imagenOriginal, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(imagenOriginal,(x+int(w/2), y+int(h/2)), 5, (255,0,255), -1)

                # Extraer características cara
                cara_extraida = extraer_caracteristicas_cara(gray, 
                                                             face, 
                                                             tipo_extraccion)

                # Predecir emoción
                resultado_prediccion = predecir_emocion(cara_extraida)
                #print(resultado_prediccion)
                
                if utilizarPCA:
                    utilizarPCA = False
                    cara_extraida = extraer_caracteristicas_cara(gray, 
                                                                 face, 
                                                                 tipo_extraccion)
                    utilizarPCA = True
                    
                #Dibujar cara detectada en la esquina superior derecha
                imagenOriginal[indice_caras * int(anchoRostro): (indice_caras + 1) * int(altoRostro), -int(anchoRostro+1):-1, :] = cv2.cvtColor(cara_extraida * 255, cv2.COLOR_GRAY2RGB)                
                indice_caras+= 1
                imagenOriginal = definir_Titulo_Emocion(imagenOriginal,
                                                        resultado_prediccion,
                                                        x, y)
                cv2.imshow("cara_extraida #" + str(indice_caras),
                           cara_extraida)
    tituloVentana = "Reconocedor de emociones sobre imagenes estaticas - " \
    + nombreClasificador
    cv2.imshow(tituloVentana, imagenOriginal)
    cv2.imwrite(os.path.join(ruta , "Frame_1.jpg"), imagenOriginal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################################################################
# =============================================================================

if __name__ == "__main__":

    #Obtener hora y fecha del sistema actual
    x = datetime.datetime.now()
    ruta = ('../source/ResulDetecEmociones/' + str(nombreClasificador) 
    + ' - ReconocimientoEmociones_'+ str(x.day) + "_" + str(x.month) 
    + "_" + str(x.year) + "_" + str(x.hour) + "_" + str(x.minute) + "_ " 
    + str(x.second))
    #Crear directorio en la cual se almacenarán las imágenes de reconocimiento.
    crear_directorio(ruta)
    ruta2 = ('../source/ClasificadoresEntrenados')
    crear_directorio(ruta2)
    
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Bienvenido al reconocedor de emociones"
    imprimirTextoCentrado(texto, tamanoConsola, "#")
    print("Autor: Castillo Serrano Cristian Michell")
    print("Versión: 0.0.1")
    print("Fecha: 28 de octubre de 2018")    
    imprimirTextoCentrado("", tamanoConsola,"-")
    
    #Importar el modelo PCA
    if utilizarPCA:
        _PCA = "-PCA-"
        pca = joblib.load(ruta2 + '/('+ nombreClasificador + ') PCA.joblib') 
    
    #Leer modelo
    #Se lee el modelo almacenado en memoria secundaria.
    model = joblib.load(ruta2 + "/" + nombreClasificador + _PCA + '.joblib') 

    #Realizar reconocimiento de expresiones faciales    
    if(tipoReco ==1):
        reconocimiento_estatico(0.3, 0.0, "Prueba_1") #(0.075, 0.05) (0.3, 0.05)
    else:
        reconocimiento_tiempo_real()