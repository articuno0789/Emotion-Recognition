#Proyecto: Generar clasificación de emociones - Reconocedor de emociones
 
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

#------------------------------Importar paquetes------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from sklearn import datasets
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
#----------------------------------------------------------------------------

print(__doc__)

procesoTerminado = False

#--------------------------Variables de configuración--------------------------
#Tamaño de la consola en la cual imprimir los resultados
tamanoConsola = 50

#Tipo de grafica al desplegar los resultados finales de la clasificación
tipoDeGrafica = 1
#1) Grafica de barras
#2) Grafica de pastel
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

    def salvar_resultado(self, emocion=1):
        resultado = ""
        if(emocion == 1):
            resultado = "Sonriendo"
        elif (emocion == 0):
            resultado = "No sonriendo"
        elif (emocion == 2):
            resultado = "Sorprendido"
        elif (emocion == 3):
            resultado = "Enojado"
        elif (emocion == 4):
            resultado = "Neutral"
        print ("Rostro", self.indice + 1, ":", resultado)
        self.resultados[str(self.indice)] = emocion

def imprimirTextoCentrado(texto, tamanoTexto, relleno=" "):
    """
    Imprime el texto especificado de manera centra, flanqueando el texto
    con el carácter de relleno especificado.
    """
    print (texto.center(tamanoTexto, relleno)) 

# ==================================================
# Funciones activables en los eventos de los botones
# ==================================================

def iniciar():
    """
    Se reinicia la ventana y la clasificación como en un primer momento.
    """
    global procesoTerminado
    procesoTerminado = False
    trainer.reset() #Reiniciar entrenador
    #Reiniciar variables globales
    global contadorSonriendo, contadorNoSonriendo
    global contadorSorprendido, contadorEnojado, contadorNeutral
    contadorSonriendo = 0
    contadorNoSonriendo = 0
    contadorSorprendido = 0
    contadorEnojado = 0
    contadorNeutral = 0
    actualizar_contador_imagenes()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])

def salir():
    """
    Se detiene la ventana que se está utilizando y se destruyen sus 
    implicaciones.
    """
    root.quit()     # Detener mainloop
    root.destroy()  # Esto es necesario en Windows para prevenir el error
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def fun_sonreir():
    """
    Se encarga de realizar la clasificación del rostro en cuestión de acuerdo
    a la emoción "Sonreír".
    """
    trainer.salvar_resultado(emocion=1)
    trainer.incrementar_indice_rostro()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])
    actualizar_contador_imagenes(countSonri=True)


def fun_no_sonreir():
    """
    Se encarga de realizar la clasificación del rostro en cuestión de acuerdo
    a la emoción "No sonreír".
    """
    trainer.salvar_resultado(emocion=0)
    trainer.incrementar_indice_rostro()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])
    actualizar_contador_imagenes(countNoSonri=True)

def fun_sorprendido():
    """
    Se encarga de realizar la clasificación del rostro en cuestión de acuerdo
    a la emoción "Sorprendido".
    """
    trainer.salvar_resultado(emocion=2)
    trainer.incrementar_indice_rostro()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])
    actualizar_contador_imagenes(countSorpre=True)

def fun_enojado():
    """
    Se encarga de realizar la clasificación del rostro en cuestión de acuerdo
    a la emoción "Enojado".
    """
    trainer.salvar_resultado(emocion=3)
    trainer.incrementar_indice_rostro()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])
    actualizar_contador_imagenes(countEnojo=True)

def fun_neutral():
    """
    Se encarga de realizar la clasificación del rostro en cuestión de acuerdo
    a la emoción "Neutral".
    """
    trainer.salvar_resultado(emocion=4)
    trainer.incrementar_indice_rostro()
    mostrar_rostro_actual(trainer.imgs[trainer.indice])
    actualizar_contador_imagenes(countNeutral=True)

def actualizar_contador_imagenes(countSonri=False, countNoSonri=False, 
                                 countSorpre=False, countEnojo=False,
                                 countNeutral=False):
    """
    Se encarga de actualizar el contador de imágenes. Asimismo, se encarga de
    verificar si el proceso de clasificación a terminado para desplegar
    la grafica de los resultados.
    """
    #Ojo: Aquí se están afectando las variables globales, cuando se aumenta el
    #numero de rostros identificado con esa emoción.
    global contadorSonriendo, contadorNoSonriendo, imageCountString, countString
    global contadorSorprendido, contadorEnojado, contadorNeutral
    if countSonri is True and contadorSonriendo < len(rostros.images):
        contadorSonriendo = contadorSonriendo + 1
    elif countNoSonri is True and contadorNoSonriendo < len(rostros.images):
        contadorNoSonriendo = contadorNoSonriendo + 1
    elif countSorpre is True and contadorSorprendido < len(rostros.images):
        contadorSorprendido = contadorSorprendido + 1
    elif countEnojo is True and contadorEnojado < len(rostros.images):
        contadorEnojado = contadorEnojado + 1
    elif countNeutral is True and contadorNeutral < len(rostros.images):
        contadorNeutral = contadorNeutral + 1
    if contadorSonriendo == len(rostros.images) \
    or contadorNoSonriendo == len(rostros.images)\
    or contadorSorprendido == len(rostros.images)\
    or contadorEnojado == len(rostros.images)\
    or contadorNeutral == len(rostros.images):
        contadorSonriendo = 0
        contadorNoSonriendo = 0
        contadorSorprendido = 0
        contadorEnojado = 0
        contadorNeutral = 0
    # Actualizar contador de imagenes clasificadas y porcentaje
    if trainer.indice+1 < len(rostros.images): 
        contadorPorcentaje = str(float((trainer.indice + 1) * 0.25))
    else:
        contadorPorcentaje = "Finalizada la clasificación"
    
    #Actualizar el porcentaje del proceso de clasificación
    imageCountString = "Progreso de clasificación " + str(trainer.indice+1) + \
    "/" + str(len(rostros.images)) + " (" + contadorPorcentaje + " %)"
    labelVar.set(imageCountString) 
    
    #Actualizar los contadores individuales para cada emoción
    countString = "(Sonreír: " + str(contadorSonriendo) + \
    "   " + "No sonreír: " + str(contadorNoSonriendo)+ \
    "   " + "Sorprendido: " + str(contadorSorprendido)+ \
    "   " + "Enojado: " + str(contadorEnojado)+ \
    "   " + "Neutral: " + str(contadorNeutral) + ")\n"
    countVar.set(countString)
    
def mostrar_grafica_final(graficaFinal):
    """
    Se encarga de actualizar el contador de imágenes. Asimismo, se encarga de
    verificar si el proceso de clasificación a terminado para desplegar
    la gráfica de los resultados.
    """
    ax[1].axis(graficaFinal)
    n_groups = 1
    #Determinar cantidad de elementos de cada emoción
    Sonreir = (sum([trainer.resultados[x] == 1 for x in trainer.resultados]))
    NoSonreir = (sum([trainer.resultados[x] == 0 for x in trainer.resultados]))
    Sorprendido = (sum([trainer.resultados[x] == 2 for x in trainer.resultados]))
    Enojado = (sum([trainer.resultados[x] == 3 for x in trainer.resultados]))
    Neutral = (sum([trainer.resultados[x] == 4 for x in trainer.resultados]))
    indice = np.arange(n_groups)
    
    #Crear grafica
    if tipoDeGrafica == 1: #Grafica de barras
        anchoBarra = 0.1
        ax[1].bar(indice, Sonreir, anchoBarra,color='b',
          label='Sonreír')
        ax[1].bar(indice + anchoBarra, NoSonreir, anchoBarra, color='r',
          label='NoSonreír')
        ax[1].bar(indice + anchoBarra, Sorprendido, anchoBarra, color='y',
          label='Sorprendido')
        ax[1].bar(indice + anchoBarra, Enojado, anchoBarra, color='g',
          label='Enojado')
        ax[1].bar(indice + anchoBarra, Neutral, anchoBarra, color='k',
          label='Neutral')
    elif tipoDeGrafica == 2: #Grafica de pastel
        ax[1].pie(indice, Sonreir, color='b', label='Sonreír')
        ax[1].pie(indice, NoSonreir, color='r', label='NoSonreír')
        ax[1].bar(indice, Sorprendido, color='y', label='Sorprendido')
        ax[1].bar(indice, Enojado, color='g', label='Enojado')
        ax[1].bar(indice, Neutral, color='k', label='Neutral')
    
    # Fijar configuraciones el plot
    ax[1].set_ylim(0, max(Sonreir, NoSonreir)+40)
    ax[1].set_xlabel('Expresiones')
    ax[1].set_ylabel('Numero de rostros')
    ax[1].set_title('Clasificación del conjunto de datos')
    ax[1].legend()
    

def mostrar_salvar_resultados():
    """
    Se encarga mostrar los resultados de la clasificación y esos resultados
    se almacenan en un archivo.
    """
    global procesoTerminado
    procesoTerminado = True
    #Imprimir resultados
    texto = "Resultados de la clasificación"
    imprimirTextoCentrado(texto, tamanoConsola, "#")
    #Guardar resultados
    print (trainer.resultados)
    with open("../resultados/ResultadosClasificacion.xml", 'w') as output:
        #Guardar resultados en formato json
        json.dump(trainer.resultados, output)

def mostrar_rostro_actual(rostro):
    """
    Se encarga de mostrar en pantalla el rostro actual a clasificar.
    """
    ax[0].imshow(rostro, cmap='gray')
    graficaFinal = 'off'
    if trainer.indice+1 == len(rostros.images):
        graficaFinal = 'on'
    if procesoTerminado == False:
        if graficaFinal is 'on':
            mostrar_grafica_final(graficaFinal)
            mostrar_salvar_resultados()
    canvas.draw()
#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    imprimirTextoCentrado("", tamanoConsola,"-")
    texto = "Bienvenido al clasificador del reconocedor de emociones"
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
    rostros = datasets.fetch_olivetti_faces()
    imprimirTextoCentrado("", tamanoConsola,"-")
    
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
    
    #==========================================================================
    #============Creación de la interface gráfica de clasificación=============
    #==========================================================================
    
    # Incrustar los elementos en una tkinter plot y comenzar tkinter plot
    #Nota valores posibles a colocar: WXAgg, GTKAgg, QT4Agg, QT5Agg, TkAgg
    #matplotlib.use('TkAgg')
    
    #Crear y configurar interface principal
    root = Tk.Tk()
    root.wm_title("Reconocedor de emociones")
    root.iconbitmap('../recursos/iconoReconocedor.ico')

    # =======================================
    # Instancias de clase y comenzando a realizar el plot
    # =======================================
    trainer = Trainer()

    # Crear un plot e incrustar en el gráfico tkinter
    f, ax = plt.subplots(1, 2)
    #Iniciar este subplot con la primera imagen del conjunto
    ax[0].imshow(rostros.images[0], cmap='gray')
    ax[1].axis('off')

    # Incrustar la figura anteriormente generada en el canvas de Tkinter
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    # =======================================
    # Declarar instancia de elementos (Botones y etiquetas)
    # =======================================
    
    labelVarEmociones = Tk.StringVar()
    labelEmociones = Tk.Label(master=root, textvariable=labelVarEmociones,
                              font = "Verdana 10 bold")
    categoriaEmociones = "Categorías de emociones"
    labelVarEmociones.set(categoriaEmociones)
    labelEmociones.pack(side=Tk.TOP)
    
    botonSonrreir = Tk.Button(master=root, text='Sonriendo',
                              command=fun_sonreir, fg="blue")
    botonSonrreir.pack(side=Tk.TOP)

    botonNoSonrreir = Tk.Button(master=root, text='No sonriendo',
                                command=fun_no_sonreir, fg="red")
    botonNoSonrreir.pack(side=Tk.TOP)
    
    botonSorprendido = Tk.Button(master=root, text='Sorprendido',
                                 command=fun_sorprendido, fg="yellow")
    botonSorprendido.pack(side=Tk.TOP)
    
    botonEnojado = Tk.Button(master=root, text='Enojado',
                             command=fun_enojado, fg="green")
    botonEnojado.pack(side=Tk.TOP)
    
    botonNeutral = Tk.Button(master=root, text='Neutral',
                             command=fun_neutral, fg="black")
    botonNeutral.pack(side=Tk.TOP)

    # ========== Etiquetas que muestran el proceso de clasificación ===========
    labelTituloProgreso = Tk.StringVar()
    labelTituProce = Tk.Label(master=root, textvariable=labelTituloProgreso,
                              font = "Verdana 10 bold")
    tituloProgreso = "Información de la clasificación realizada"
    labelTituloProgreso.set(tituloProgreso)
    labelTituProce.pack(side=Tk.TOP)

    labelVar = Tk.StringVar()
    label = Tk.Label(master=root, textvariable=labelVar)
    imageCountString = "Progreso en la clasificación: 0/" \
    + str(len(rostros.images)) + "   (0 %)"
    labelVar.set(imageCountString)
    label.pack(side=Tk.TOP)

    countVar = Tk.StringVar()
    contadorSonriendo = 0
    contadorNoSonriendo = 0
    contadorSorprendido = 0
    contadorEnojado = 0
    contadorNeutral = 0
    countLabel = Tk.Label(master=root, textvariable=countVar)
    countString = "(Feliz: 0   Triste: 0   Sorprendido: 0   "\
    "Enojado: 0   Neutral: 0)\n"
    countVar.set(countString)
    countLabel.pack(side=Tk.TOP)
    #==========================================================================

    # ========================== Botones de control ===========================
    botonReinicio = Tk.Button(master=root, text='Reiniciar Proceso',
                              command=iniciar)
    botonReinicio.pack(side=Tk.TOP)

    botonSalida = Tk.Button(master=root, text='Cerrar aplicación',
                            command=salir)
    botonSalida.pack(side=Tk.TOP)
    #==========================================================================

    # ======================== Información del autor ==========================
    authorVar = Tk.StringVar()
    authorLabel = Tk.Label(master=root, textvariable=authorVar)
    authorString = "\n\nPrograma realizado por: " \
                   "\n Castillo Serrano Cristian Michell - 215861738 "
    authorVar.set(authorString)
    authorLabel.pack(side=Tk.BOTTOM)
    #==========================================================================

    #Iniciar la aplicación
    Tk.mainloop() 
