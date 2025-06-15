import numpy as np
import cv2
import pydicom
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Paciente:
    """Clase obligatoria para representar un paciente"""
    def __init__(self, nombre, edad, id_paciente, imagen_asociada):
        self.nombre = nombre
        self.edad = edad
        self.id = id_paciente
        self.imagen_asociada = imagen_asociada  # Matriz 3D reconstruida
    
    def __str__(self):
        return f"Paciente: {self.nombre}, Edad: {self.edad}, ID: {self.id}"

class ProcesadorDICOM:
    """Clase para procesar archivos DICOM"""
    
    @staticmethod
    def cargar_carpeta_dicom(ruta_carpeta):
        """Carga todos los archivos DICOM de una carpeta"""
        archivos_dicom = []
        nombres_archivos = []
        
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith('.dcm'):
                ruta_completa = os.path.join(ruta_carpeta, archivo)
                try:
                    ds = pydicom.dcmread(ruta_completa)
                    archivos_dicom.append(ds)
                    nombres_archivos.append(archivo)
                except Exception as e:
                    print(f"Error al leer {archivo}: {e}")
        
        return archivos_dicom, nombres_archivos
    
    @staticmethod
    def reconstruir_3d(archivos_dicom):
        """Reconstruye imagen 3D a partir de archivos DICOM"""
        if not archivos_dicom:
            return None
        
        # Ordenar por posición de slice si está disponible
        try:
            archivos_dicom.sort(key=lambda x: float(x.SliceLocation))
        except:
            print("No se pudo ordenar por SliceLocation, usando orden original")
        
        # Extraer matrices de pixel
        imagenes = []
        for ds in archivos_dicom:
            imagen = ds.pixel_array
            imagenes.append(imagen)
        
        # Crear matriz 3D
        volumen_3d = np.stack(imagenes, axis=0)
        return volumen_3d
    
    @staticmethod
    def mostrar_cortes(volumen_3d, titulo="Reconstrucción 3D"):
        """Muestra los 3 cortes principales en subplots"""
        if volumen_3d is None:
            print("No hay volumen 3D para mostrar")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Corte transversal (axial) - mitad del volumen
        corte_transversal = volumen_3d[volumen_3d.shape[0]//2, :, :]
        axes[0].imshow(corte_transversal, cmap='gray')
        axes[0].set_title('Corte Transversal (Axial)')
        axes[0].axis('off')
        
        # Corte coronal - mitad en Y
        corte_coronal = volumen_3d[:, volumen_3d.shape[1]//2, :]
        axes[1].imshow(corte_coronal, cmap='gray')
        axes[1].set_title('Corte Coronal')
        axes[1].axis('off')
        
        # Corte sagital - mitad en X
        corte_sagital = volumen_3d[:, :, volumen_3d.shape[2]//2]
        axes[2].imshow(corte_sagital, cmap='gray')
        axes[2].set_title('Corte Sagital')
        axes[2].axis('off')
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def extraer_info_paciente(archivo_dicom):
        """Extrae información del paciente de un archivo DICOM"""
        try:
            nombre = getattr(archivo_dicom, 'PatientName', 'Anonimo')
            edad = getattr(archivo_dicom, 'PatientAge', 'Desconocida')
            id_paciente = getattr(archivo_dicom, 'PatientID', 'ID_Desconocido')
            
            # Convertir a string si es necesario
            if hasattr(nombre, 'family_name'):
                nombre = f"{nombre.family_name} {nombre.given_name}"
            else:
                nombre = str(nombre)
            
            return str(nombre), str(edad), str(id_paciente)
        except Exception as e:
            print(f"Error extrayendo información del paciente: {e}")
            return "Anonimo", "Desconocida", "ID_Desconocido"
    
    @staticmethod
    def trasladar_imagen(imagen, tx, ty):
        """Aplica transformación de traslación usando OpenCV"""
        filas, columnas = imagen.shape[:2]
        matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
        imagen_trasladada = cv2.warpAffine(imagen, matriz_traslacion, (columnas, filas))
        return imagen_trasladada

class ProcesadorImagenes:
    """Clase para procesar imágenes JPG y PNG"""
    
    TIPOS_BINARIZACION = {
        '1': ('Binario', cv2.THRESH_BINARY),
        '2': ('Binario Invertido', cv2.THRESH_BINARY_INV),
        '3': ('Truncado', cv2.THRESH_TRUNC),
        '4': ('To Zero', cv2.THRESH_TOZERO),
        '5': ('To Zero Invertido', cv2.THRESH_TOZERO_INV)
    }
    
    @staticmethod
    def cargar_imagen(ruta):
        """Carga una imagen JPG o PNG"""
        try:
            imagen = cv2.imread(ruta)
            if imagen is None:
                raise ValueError("No se pudo cargar la imagen")
            return imagen
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return None
    
    @staticmethod
    def mostrar_menu_binarizacion():
        """Muestra el menú de opciones de binarización"""
        print("\n=== Opciones de Binarización ===")
        for key, (nombre, _) in ProcesadorImagenes.TIPOS_BINARIZACION.items():
            print(f"{key}. {nombre}")
        print("================================")
    
    @staticmethod
    def binarizar_imagen(imagen, tipo_binarizacion, umbral=127):
        """Binariza la imagen según el tipo seleccionado"""
        # Convertir a escala de grises si está en color
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        tipo_cv = ProcesadorImagenes.TIPOS_BINARIZACION[tipo_binarizacion][1]
        _, imagen_binarizada = cv2.threshold(imagen_gris, umbral, 255, tipo_cv)
        
        return imagen_binarizada
    
    @staticmethod
    def aplicar_morfologia(imagen, kernel_size):
        """Aplica transformación morfológica"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Aplicar apertura (erosión seguida de dilatación)
        imagen_morfologica = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
        return imagen_morfologica
    
    @staticmethod
    def dibujar_forma_con_texto(imagen, forma='circulo', texto="Imagen binarizada", 
                               umbral=127, kernel_size=5):
        """Dibuja una forma (círculo o cuadrado) con texto"""
        # Crear una copia para dibujar
        imagen_resultado = imagen.copy()
        
        # Si la imagen es en escala de grises, convertir a color para el texto
        if len(imagen_resultado.shape) == 2:
            imagen_resultado = cv2.cvtColor(imagen_resultado, cv2.COLOR_GRAY2BGR)
        
        altura, ancho = imagen.shape[:2]
        centro_x, centro_y = ancho // 2, altura // 2
        
        if forma.lower() == 'circulo':
            # Dibujar círculo
            radio = min(ancho, altura) // 4
            cv2.circle(imagen_resultado, (centro_x, centro_y), radio, (0, 255, 0), 3)
            
            # Posición del texto dentro del círculo
            texto_x = centro_x - 80
            texto_y = centro_y - 10
        else:  # cuadrado
            # Dibujar cuadrado
            lado = min(ancho, altura) // 3
            x1 = centro_x - lado // 2
            y1 = centro_y - lado // 2
            x2 = centro_x + lado // 2
            y2 = centro_y + lado // 2
            cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Posición del texto dentro del cuadrado
            texto_x = x1 + 10
            texto_y = y1 + 30
        
        # Agregar texto con información
        texto_completo = f"{texto}"
        texto_umbral = f"Umbral: {umbral}"
        texto_kernel = f"Kernel: {kernel_size}x{kernel_size}"
        
        # Configurar fuente
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.6
        color = (255, 255, 255)  # Blanco
        grosor = 2
        
        # Dibujar textos
        cv2.putText(imagen_resultado, texto_completo, (texto_x, texto_y), 
                   fuente, escala, color, grosor)
        cv2.putText(imagen_resultado, texto_umbral, (texto_x, texto_y + 25), 
                   fuente, escala, color, grosor)
        cv2.putText(imagen_resultado, texto_kernel, (texto_x, texto_y + 50), 
                   fuente, escala, color, grosor)
        
        return imagen_resultado