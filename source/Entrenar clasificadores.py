#Proyecto: Reconocedor de emociones - Entrenador de modelos.
 
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
#Tamano de muestras para el conjunto de datos Fer2013
numRegiImportar = 10000
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
#Generar matrices de confusión
generarMatriz = True
#Generar comparativa resultados reales vs. predichos
generarCompar = False
#Utilizar PCA
utilizarPCA = True
numeroCompPrin = 200
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

# ==========================================================================
# Recorre el conjunto de datos incrementando el índice y registra el resultado
# ==========================================================================
class Trainer:
    def __init__(self):
        self.resultados = {}
        self.imgs = rostros.images
        self.indice = 0

    def reset(self):
        print ("||||||||||||||||||||||||||||||||||||||||||||")
        print ("Reiniciando el programa y la clasificación")
        print ("||||||||||||||||||||||||||||||||||||||||||||")
        self.resultados = {}
        self.imgs = rostros.images
        self.indice = 0

    def incrementar_indice_rostro(self):
        if self.indice + 1 >= len(self.imgs):
            return self.indice
        else:
            while str(self.indice) in self.resultados:
                self.indice += 1
            return self.indice

    def salvar_resultado(self, emocion=True):
        resultado = ""
        if(emocion is True):
            resultado = "Sonriendo"
        elif (emocion is False):
            resultado = "No sonriendo"
        print ("Rostro", self.indice + 1, ":", resultado)
        self.resultados[str(self.indice)] = emocion

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y traza la matriz de confusión.
    La normalización se puede aplicar configurando `normalize = True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
 
def crear_directorio(ruta):
    """
    Crea un directorio nuevo de acuerdo al nombre especificado.
    Si existe el directorio deseado borra su contenido y lo crea de nuevo.
    """
    try:
        os.makedirs(ruta)
    except OSError:
        pass

def mostrarMetricaF1(model, x, y_pred, nombreConjunto):
    """
    Se encarga de mostrar las métricas relacionadas con el F1 score, para un
    conjunto de datos determinado.
    """
    texto = "F1 SCORE (" + nombreConjunto + ")"
    imprimirTextoCentrado(texto, tamanoConsola)
    lb = LabelBinarizer()
    y_pred2 = np.array([number[0] for number in lb.fit_transform(y_pred)])
    recall = cross_val_score(model, x, y_pred2, cv=5, scoring='recall')
    print('Recall: ', np.mean(recall), recall)
    precision = cross_val_score(model, x, y_pred2, cv=5, scoring='precision')
    print('Precision: ', np.mean(precision), precision)
    f1 = cross_val_score(model, x, y_pred2, cv=5, scoring='f1')
    print('F1: ', np.mean(f1), f1)

def mostrarReporteClasificacion(y_ori, y_pred, nombreConjunto):
    """
    Se encarga de mostrar la métrica llamada 'Reporte de clasificación'.
    """
    texto = "Classification Report (" + nombreConjunto + ")"
    imprimirTextoCentrado(texto, tamanoConsola)
    print (metrics.classification_report(y_ori, y_pred))
    
def mostrarMetricasGenerales(model, x, y_ori, y_pred, nombreConjunto):
    """
    Se encarga de mostrar las métricas generales resultante de comparar los
    datos reales vs los predichos.
    """
    "MSE (" + nombreConjunto + "): "
    print("MSE (" + nombreConjunto + "): ",
          metrics.mean_squared_error(y_ori, y_pred))
    print("MAE (" + nombreConjunto + "): ",
          metrics.mean_absolute_error(y_ori, y_pred))
    print("EVS (" + nombreConjunto + "): ",
          metrics.explained_variance_score(y_ori, y_pred))
    print("SCORE (" + nombreConjunto + "): ",
          model.score(x, y_ori))
    
def mostrarMetricasClasificacion(model, x, y_ori, y_pred, nombreConjunto):
    """
    Se encarga de mostrar las métricas especificares relacionadas con 
    clasificación como lo es, la matriz de confusión, el reporte de 
    clasificación y el coeficiente de similitud de Jaccard.
    """
    imprimirTextoCentrado("Conjunto " + nombreConjunto, tamanoConsola,"*")
    
    #imprimirTextoCentrado("F1 SCORE", tamanoConsola,"*")
    #mostrarMetricaF1(model, x, y_pred, nombreConjunto)
    
    imprimirTextoCentrado("MATRIZ DE CONFUSION", tamanoConsola,"*")
    confusion_matrix_ = confusion_matrix(y_ori, y_pred);
    print("Matriz de confusion - " + nombreConjunto)
    print(confusion_matrix_)
    
    imprimirTextoCentrado("REPORTE DE CLASIFICACION", tamanoConsola,"*")
    mostrarReporteClasificacion(y_ori, y_pred, nombreConjunto)
    
    imprimirTextoCentrado("JACCARD_SIMILARITY_SCORE", tamanoConsola,"*")
    print("jaccard_similarity_score (" + nombreConjunto + "): ", 
          jaccard_similarity_score(y_ori,y_pred))
    
    return confusion_matrix_

def mostrarGraficacionPrediVsReal(x, y_ori, y_pred, nombreConjunto, nombreModelo):
    """
    Se encarga de realizar una gráfica donde se comparan los valores predichos
    vs los reales.
    """
    fig = plt.figure(num=None, figsize=(14, 5))
    titulo = "Grafica comparativa resultados predichos vs reales (" +\
    str(nombreModelo) + ") (" + str(nombreConjunto) + ")"
    plt.title(titulo, fontsize=22)
    plt.ylabel("Clasificacion", fontsize=18)
    plt.plot(x, y_ori,'ok')
    plt.plot(x, y_pred,'or')
    plt.plot()
    fig.savefig(ruta + "/" + titulo + '.jpg')

def mostrarMatrizConfusionGrafico(confusion_matrix, class_names,
                                  nombreConjunto, nombreModelo, 
                                  precisionNum = 2):
    """
    Se encarga de generar las matrices de confusión graficas del modelo.
    Se crean dos gráficos, donde uno representa el resultado normalizado y otro
    resultado no normalizado.
    """
    #Establecer presisión de los plots
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    fig = plt.figure(num=None, figsize=(14, 5))
    titulo = "Confusion matrix, without normalization (" \
    + str(nombreModelo) + ") (" + str(nombreConjunto) + ")"
    plot_confusion_matrix(confusion_matrix, classes=class_names, title=titulo)
    fig.savefig(ruta + "/" + titulo + '.jpg')
    
    # Plot normalized confusion matrix
    fig = plt.figure(num=None, figsize=(14, 5))
    titulo = "Normalized confusion matrix (" + str(nombreModelo) \
    + ") (" + str(nombreConjunto) + ")"
    plot_confusion_matrix(confusion_matrix,classes=class_names,
                          normalize=True, title=titulo)
    fig.savefig(ruta + "/" + titulo + '.jpg')

def imprimirTextoCentrado(texto, tamanoTexto, relleno=" "):
    """
    Imprime el texto especificado de manera centra, flanqueando el texto
    con el carácter de relleno especificado.
    """
    print (texto.center(tamanoTexto, relleno)) 

def predecir_emocion(cara_extraida):
    """
    Se encarga de evaluar dentro del modelo el patrón de prueba actual 
    y reportar el resultado obtenido.
    """
    prediccion = model.predict(cara_extraida.reshape(1, -1))
    return prediccion

def acomodar_datos_fer2013(csv_f):
    test_set_y =[]
    test_set_x =[]
    counter = 0
    for row in csv_f:  
        #print(counter)
        if(counter == numRegiImportar+2):
            break
        counter+= 1
        if(counter != numRegiImportar+2):
            if str(row[2]) == "Training":
                test_set_y.append(int(row[0]))
                temp_list = []
                for pixel in row[1].split():
                    temp_list.append(int(pixel))
                test_set_x.append(temp_list)
    y = np.asanyarray(test_set_y)
    x = np.asanyarray(test_set_x)
    return x, y
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
    texto = "Bienvenido al reconocedor de emociones - Entrenador de modelos"
    imprimirTextoCentrado(texto, tamanoConsola, "#")
    print("Autor: Castillo Serrano Cristian Michell")
    print("Versión: 0.0.1")
    print("Fecha: 28 de octubre de 2018")    
    imprimirTextoCentrado("", tamanoConsola,"-")
    
    #Leer datos
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Leyendo la base de datos de rostros"
    imprimirTextoCentrado(texto, tamanoConsola)
    print ("\nEste proceso podría tardar unos momentos, por favor sea paciente...")
    
    if(archivoDeEntrenamiento == 1):
        rostros = datasets.fetch_olivetti_faces()
    elif (archivoDeEntrenamiento == 2):
        f = open('../recursos/fer2013.csv')
        rostros = csv.reader(f)
    imprimirTextoCentrado("", tamanoConsola,"-")
    
    if(archivoDeEntrenamiento == 1):
        #Mostrar metadatos del conjunto de rostros
        print("\n")
        imprimirTextoCentrado("", tamanoConsola,"-")
        texto = "Metadatos del conjunto de imágenes a ser clasificado"
        imprimirTextoCentrado(texto, tamanoConsola, "#")
        print ("Palabras llaves en el conjunto de datos: ", rostros.keys())
        print ("Total de las imágenes en el conjunto de imágenes" \
               " (olivetti_faces_dataset):", len(rostros.images))
        #print ("Objetivos para cada imagen: ", rostros.target)
        #print ("Datos de pixeles por imagen: ", rostros.data)
        imprimirTextoCentrado("", tamanoConsola,"-")
        print("\n")

    #Crear modelo
    if clasificador==1:# Nearest Neighbors
        model = KNeighborsClassifier(n_neighbors=5,
                                     algorithm='auto', leaf_size=30,
                                     p=2, metric='minkowski')
    elif clasificador==2:#Linear SVM (Funciona, este es el chido)
        model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto',
                    coef0=0.0, shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False,
                    max_iter=-1, decision_function_shape='ovr',
                    random_state=None)
    elif clasificador==3:#Linear SVM
        model = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                    coef0=0.0, shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False,
                    max_iter=-1, decision_function_shape='ovr',
                    random_state=None)
    elif clasificador==4:#Poly SVM
        model = SVC(C=1.0, kernel='poly', degree=13, gamma='auto',
                    coef0=0.0, shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False,
                    max_iter=-1, decision_function_shape='ovr',
                    random_state=None)
    elif clasificador==5:#Gaussian Process
        model = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b',
                                          n_restarts_optimizer=0,
                                          max_iter_predict=100, warm_start=False,
                                          copy_X_train=True, random_state=None,
                                          multi_class='one_vs_rest', n_jobs=None)
    elif clasificador==6:#Decision Tree
        model = DecisionTreeClassifier(criterion='gini', splitter='best',
                                       max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_features=None, random_state=None,
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_impurity_split=None,
                                       class_weight=None,
                                       presort=False)
    elif clasificador==7:# Random Forest
        model = RandomForestClassifier(n_estimators=5, criterion='gini',
                                       max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_features='auto', max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       min_impurity_split=None, bootstrap=True,
                                       oob_score=False, 
                                       random_state=None, verbose=0,
                                       warm_start=False, class_weight=None)
    elif clasificador==8:#Neural Net
        model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=0.001,
                              max_iter=200, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.0001,
                              validation_fraction=0.1, verbose=False, 
                              warm_start=False)
    elif clasificador==9:#AdaBoost
        model = AdaBoostClassifier(base_estimator=None, n_estimators=50, 
                                   learning_rate=1.0, algorithm='SAMME.R', 
                                   random_state=None)
    elif clasificador==10:#NaiveBayes
        model = GaussianNB(priors=None)
    elif clasificador==11:#QDA (Funciona, pero no muy bien)
        model = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, 
                                              store_covariance=False, tol=0.0001,
                                              store_covariances=None)
            
    if(archivoDeEntrenamiento == 1):
        trainer = Trainer()
        #Cargar los resultados de la clasificación realizada
        results = json.load(open("../resultados/ResultadosClasificacion.xml"))
        trainer.results = results
    
        #Generar la lista de índices en la que se realizó la clasificación
        #Nota: Las imágenes del conjunto de datos se clasificaron de manera 
        #      desordena de forma aleatoria, por lo tanto, hay que organizar las 
        #      de acuerdo al orden en que se clasificaron.
        indices = [int(i) for i in trainer.results]
        #Crear los datos de manera ordena de acuerdo a los índices.
        data = rostros.data[indices, :]
    
        #Crear vector de clasificación o labels
        #Se crea el vector donde se guarda la clasificación a la que pertenece cada
        #imagen del conjunto de datos.
        target = [trainer.results[i] for i in trainer.results]
        target = np.array(target).astype(np.int32)
    else:
        data, target = acomodar_datos_fer2013(rostros)
        
    #Dividir conjunto de datos
    #En esta parte se crean 4 conjuntos
    #Dos conjuntos de entrenamiento, que serán los datos con los que se entrene el modelo.
    #Dos conjuntos de prueba, que serán los datos con los que pruebe el resultado del entrenamiento.
    #En este caso la semilla de generación se deja aleatoria.
    #El tamaño de la generación de los conjuntos se deja por su valor por defecto de 0.25.
    
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Creando conjunto de pruebas y de entrenamiento"
    imprimirTextoCentrado(texto, tamanoConsola)
    imprimirTextoCentrado("", tamanoConsola,"-")
    
    x_train, x_test, y_train, y_test = train_test_split(data, 
                                                        target, 
                                                        test_size=0.25, 
                                                        random_state=0)
    
    #Se utiliza la técnica de Análisis de componentes principales (PCA)
    # para reducir la dimensionalidad del conjunto de datos, eliminando los datos
    # redundantes y manteniendo los datos más significativos.
    tiempo_inicial = time() 
    if utilizarPCA:
        print("\n")
        imprimirTextoCentrado("", tamanoConsola,"-")
        texto = "Realizando el análisis de componentes principales (PCA) con "\
        + str(numeroCompPrin) + " componentes."
        imprimirTextoCentrado(texto, tamanoConsola)
        imprimirTextoCentrado("", tamanoConsola,"-")
        
        pca = PCA(n_components=numeroCompPrin)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        _PCA = "-PCA-"
        joblib.dump(pca, ruta2 + '/('+ nombreClasificador + ') PCA.joblib') 

    #Entrenar modelo
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Se esta realizando el entrenamiento del modelo."
    imprimirTextoCentrado(texto, tamanoConsola)
    print ("\nEste proceso podría tardar unos momentos, por favor sea paciente...")
    imprimirTextoCentrado("", tamanoConsola,"-")
        
    model.fit(x_train, y_train)
    
    tiempo_final = time() 
    tiempo_ejecucion = tiempo_final - tiempo_inicial
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "El tiempo de entrenamiento fue de: " + str(tiempo_ejecucion) \
                                                        + " segundos."
    imprimirTextoCentrado(texto, tamanoConsola)
    imprimirTextoCentrado("", tamanoConsola,"-")
    #Salvar modelo
    #Se realiza el guardado del modelo entrenado en memoria secundaria.
    #Es deseable tener una forma de persistir el modelo para su uso futuro sin 
    #tener que volver a entrenar.
    print("\n")
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Se esta realizando el guardado del modelo."
    imprimirTextoCentrado(texto, tamanoConsola)
    imprimirTextoCentrado("", tamanoConsola,"-")
    
    joblib.dump(model, ruta2 + "/" + nombreClasificador + _PCA + '.joblib') 
    
    #Predecir
    #Realizar las predicciones correspondientes tanto con el conjunto de prueba 
    #como el conjunto de entrenamiento.
    #Estas operaciones calcularan los valores resultantes que estima el modelo
    #para cada conjunto.
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
       
    #Métricas
    #Se realiza el mostrado de las métricas del modelo
    #En esta sección del programa se muestran las métricas que permiten apreciar
    #las características del modelo y datos relevantes para poder evaluar
    #las predicciones del modelo.
    metricas = ("Metricas del modelo " + nombreClasificador).capitalize() 
    imprimirTextoCentrado(metricas, tamanoConsola)
    mostrarMetricasGenerales(model, x_train, y_train, y_pred_train, "train")
    mostrarMetricasGenerales(model, x_test, y_test, y_pred_test, "test")
    imprimirTextoCentrado("", tamanoConsola,"*")
    imprimirTextoCentrado("Metricas importantes para clasificación", 
                          tamanoConsola,"*")
    confusion_matrix_train= mostrarMetricasClasificacion(model, x_train, 
                                                         y_train, y_pred_train,
                                                         "train")
    imprimirTextoCentrado("", tamanoConsola,"#")
    confusion_matrix_test= mostrarMetricasClasificacion(model, x_test, 
                                                        y_test, y_pred_test,
                                                        "test")
    #Crear un iterador de validación cruzada k-fold
    #Nota: Por defecto, la puntuación utilizada es la que se devuelve por el 
    #      método de puntuación del estimador (precisión)
    cv = KFold(n=len(y_train), n_folds=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, x_train, y_train, cv=cv)
    print ("Scores: ", (scores))
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))
    print("*******************************************************************")
    ###########################################################################
    if generarCompar:
        #Graficacion
        #En esta sección se grafican los resultados obtenidos
        #Se grafican la clasificación emocional con respecto 
        # con respecto a las variables independiente que en este caso son los 
        # pixeles de las imágenes de los rostros.
        mostrarGraficacionPrediVsReal(x_train, y_train, y_pred_train, "train",
                                      nombreClasificador)
        mostrarGraficacionPrediVsReal(x_test, y_test, y_pred_test, "test",
                                      nombreClasificador)
    if generarMatriz:
        #Matrices de confusión (Graficas)
        #En esta sección se muestran las matrices de confusión generadas
        #tanto para el conjunto train como para el conjunto test.
        #imprimirTextoCentrado("Matrices de confusión (Graficas)", tamanoConsola,"*")
        if archivoDeEntrenamiento == 1:
            class_names = ["No Sonriendo", "Sonriendo"]
        elif archivoDeEntrenamiento == 2:
                           class_names = ["Enojado", "Disgustado", "Asustado",
                           "Feliz", "Triste", "Sorprendido", "Neutral"]
        mostrarMatrizConfusionGrafico(confusion_matrix_train, class_names,
                                      "train", nombreClasificador)
        mostrarMatrizConfusionGrafico(confusion_matrix_test, class_names,
                                      "test", nombreClasificador)