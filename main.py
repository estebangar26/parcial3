from clases import Paciente, ProcesadorDICOM, ProcesadorImagenes
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Diccionarios globales (fuera de clases como se solicita)
diccionario_dicom = {}
diccionario_pacientes = {}
diccionario_imagenes = {}

# Funciones para el menú principal y opciones

def mostrar_menu_principal():
    """Muestra el menú principal"""
    print("\n" + "="*50)
    print("    PROCESAMIENTO DE IMÁGENES MÉDICAS")
    print("="*50)
    print("a) Procesamiento de archivos DICOM")
    print("b) Ingresar Paciente")
    print("c) Ingresar imágenes JPG/PNG")
    print("d) Transformación geométrica (traslación)")
    print("e) Procesamiento de imágenes JPG/PNG")
    print("f) Salir")
    print("="*50)

def opcion_a_procesar_dicom():
    """Opción a: Procesamiento de archivos DICOM"""
    print("\n=== PROCESAMIENTO DE ARCHIVOS DICOM ===")
    
    ruta_carpeta = input("Ingrese la ruta de la carpeta con archivos DICOM: ")
    
    if not os.path.exists(ruta_carpeta):
        print("Error: La ruta especificada no existe.")
        return
    
    # Cargar archivos DICOM
    archivos_dicom, nombres_archivos = ProcesadorDICOM.cargar_carpeta_dicom(ruta_carpeta)
    
    if not archivos_dicom:
        print("No se encontraron archivos DICOM en la carpeta especificada.")
        return
    
    print(f"Se cargaron {len(archivos_dicom)} archivos DICOM")
    
    # Reconstruir volumen 3D
    volumen_3d = ProcesadorDICOM.reconstruir_3d(archivos_dicom)
    
    if volumen_3d is not None:
        print(f"Volumen 3D reconstruido con dimensiones: {volumen_3d.shape}")
        
        # Mostrar los 3 cortes
        ProcesadorDICOM.mostrar_cortes(volumen_3d, "Reconstrucción 3D - Archivos DICOM")
        
        # Guardar en diccionario
        clave = input("Ingrese una clave para guardar estos archivos DICOM: ")
        diccionario_dicom[clave] = {
            'archivos_dicom': archivos_dicom,
            'volumen_3d': volumen_3d,
            'nombres_archivos': nombres_archivos,
            'ruta_carpeta': ruta_carpeta
        }
        
        print(f"Archivos DICOM guardados con la clave: '{clave}'")
    else:
        print("Error en la reconstrucción 3D")

def opcion_b_ingresar_paciente():
    """Opción b: Ingresar Paciente"""
    print("\n=== INGRESAR PACIENTE ===")
    
    if not diccionario_dicom:
        print("Error: No hay archivos DICOM procesados. Primero ejecute la opción 'a'.")
        return
    
    # Mostrar claves disponibles
    print("Claves DICOM disponibles:")
    for clave in diccionario_dicom.keys():
        print(f"  - {clave}")
    
    clave_dicom = input("Ingrese la clave del DICOM a usar para el paciente: ")
    
    if clave_dicom not in diccionario_dicom:
        print("Error: Clave no encontrada.")
        return
    
    # Obtener datos DICOM
    datos_dicom = diccionario_dicom[clave_dicom]
    archivos_dicom = datos_dicom['archivos_dicom']
    volumen_3d = datos_dicom['volumen_3d']
    
    # Extraer información del primer archivo DICOM
    info_paciente = ProcesadorDICOM.extraer_info_paciente(archivos_dicom[0])
    nombre, edad, id_paciente = info_paciente
    
    print(f"Información extraída del DICOM:")
    print(f"  Nombre: {nombre}")
    print(f"  Edad: {edad}")
    print(f"  ID: {id_paciente}")
    
    # Permitir modificar la información
    modificar = input("¿Desea modificar esta información? (s/n): ").lower()
    if modificar == 's':
        nombre = input(f"Nuevo nombre (actual: {nombre}): ") or nombre
        edad = input(f"Nueva edad (actual: {edad}): ") or edad
        id_paciente = input(f"Nuevo ID (actual: {id_paciente}): ") or id_paciente
    
    # Crear paciente
    paciente = Paciente(nombre, edad, id_paciente, volumen_3d)
    
    # Guardar en diccionario de pacientes
    clave_paciente = input("Ingrese una clave para guardar el paciente: ")
    diccionario_pacientes[clave_paciente] = paciente
    
    # También guardar el DICOM en el diccionario de imágenes
    diccionario_imagenes[clave_dicom] = datos_dicom
    
    print(f"Paciente creado y guardado con la clave: '{clave_paciente}'")
    print(f"DICOM asociado guardado en diccionario de imágenes")

def opcion_c_ingresar_imagenes():
    """Opción c: Ingresar imágenes JPG/PNG"""
    print("\n=== INGRESAR IMÁGENES JPG/PNG ===")
    
    ruta_imagen = input("Ingrese la ruta de la imagen (JPG/PNG): ")
    
    if not os.path.exists(ruta_imagen):
        print("Error: La imagen especificada no existe.")
        return
    
    # Cargar imagen
    imagen = ProcesadorImagenes.cargar_imagen(ruta_imagen)
    
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    print(f"Imagen cargada exitosamente. Dimensiones: {imagen.shape}")
    
    # Guardar en diccionario
    clave = input("Ingrese una clave para guardar la imagen: ")
    diccionario_imagenes[clave] = {
        'imagen': imagen,
        'ruta': ruta_imagen,
        'tipo': 'imagen_comun'
    }
    
    print(f"Imagen guardada con la clave: '{clave}'")

def opcion_d_transformacion_geometrica():
    """Opción d: Transformación geométrica (traslación)"""
    print("\n=== TRANSFORMACIÓN GEOMÉTRICA (TRASLACIÓN) ===")
    
    if not diccionario_dicom:
        print("Error: No hay archivos DICOM procesados. Primero ejecute la opción 'a'.")
        return
    
    # Mostrar claves disponibles
    print("Claves DICOM disponibles:")
    for clave in diccionario_dicom.keys():
        print(f"  - {clave}")
    
    clave_dicom = input("Ingrese la clave del DICOM a transformar: ")
    
    if clave_dicom not in diccionario_dicom:
        print("Error: Clave no encontrada.")
        return
    
    # Obtener un corte del volumen DICOM
    volumen_3d = diccionario_dicom[clave_dicom]['volumen_3d']
    # Usar el corte central
    imagen_original = volumen_3d[volumen_3d.shape[0]//2, :, :].astype(np.uint8)
    
    # Opciones de traslación predefinidas
    print("\nOpciones de traslación:")
    opciones_traslacion = {
        '1': (50, 30),
        '2': (-30, 50),
        '3': (0, 70),
        '4': (100, -50)
    }
    
    for key, (tx, ty) in opciones_traslacion.items():
        print(f"{key}. Traslación X={tx}, Y={ty}")
    
    opcion = input("Seleccione una opción (1-4): ")
    
    if opcion not in opciones_traslacion:
        print("Opción no válida.")
        return
    
    tx, ty = opciones_traslacion[opcion]
    
    # Aplicar traslación
    imagen_trasladada = ProcesadorDICOM.trasladar_imagen(imagen_original, tx, ty)
    
    # Mostrar resultado
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(imagen_original, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(imagen_trasladada, cmap='gray')
    axes[1].set_title(f'Imagen Trasladada (X={tx}, Y={ty})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar imagen trasladada
    nombre_archivo = f"imagen_trasladada_{clave_dicom}_X{tx}_Y{ty}.png"
    cv2.imwrite(nombre_archivo, imagen_trasladada)
    print(f"Imagen trasladada guardada como: {nombre_archivo}")

def opcion_e_procesamiento_imagenes():
    """Opción e: Procesamiento de imágenes JPG/PNG"""
    print("\n=== PROCESAMIENTO DE IMÁGENES JPG/PNG ===")
    
    # Mostrar imágenes disponibles (solo las que no son DICOM)
    imagenes_disponibles = {k: v for k, v in diccionario_imagenes.items() 
                          if v.get('tipo') == 'imagen_comun'}
    
    if not imagenes_disponibles:
        print("Error: No hay imágenes JPG/PNG disponibles. Primero ejecute la opción 'c'.")
        return
    
    print("Imágenes disponibles:")
    for clave in imagenes_disponibles.keys():
        print(f"  - {clave}")
    
    clave_imagen = input("Ingrese la clave de la imagen a procesar: ")
    
    if clave_imagen not in imagenes_disponibles:
        print("Error: Clave no encontrada.")
        return
    
    imagen_original = imagenes_disponibles[clave_imagen]['imagen']
    
    # Menú de binarización
    ProcesadorImagenes.mostrar_menu_binarizacion()
    tipo_binarizacion = input("Seleccione el tipo de binarización (1-5): ")
    
    if tipo_binarizacion not in ProcesadorImagenes.TIPOS_BINARIZACION:
        print("Opción no válida.")
        return
    
    # Umbral para binarización
    umbral = int(input("Ingrese el valor del umbral (0-255, recomendado 127): ") or "127")
    
    # Binarizar imagen
    imagen_binarizada = ProcesadorImagenes.binarizar_imagen(imagen_original, tipo_binarizacion, umbral)
    
    # Transformación morfológica
    kernel_size = int(input("Ingrese el tamaño del kernel para morfología (ej: 5): ") or "5")
    imagen_morfologica = ProcesadorImagenes.aplicar_morfologia(imagen_binarizada, kernel_size)
    
    # Dibujar forma con texto
    forma = input("¿Qué forma desea dibujar? (circulo/cuadrado): ").lower()
    if forma not in ['circulo', 'cuadrado']:
        forma = 'circulo'
    
    tipo_bin_nombre = ProcesadorImagenes.TIPOS_BINARIZACION[tipo_binarizacion][0]
    imagen_final = ProcesadorImagenes.dibujar_forma_con_texto(
        imagen_morfologica, forma, f"Imagen binarizada ({tipo_bin_nombre})", umbral, kernel_size
    )
    
    # Mostrar resultados
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Imagen original
    if len(imagen_original.shape) == 3:
        axes[0,0].imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
    else:
        axes[0,0].imshow(imagen_original, cmap='gray')
    axes[0,0].set_title('Imagen Original')
    axes[0,0].axis('off')
    
    # Imagen binarizada
    axes[0,1].imshow(imagen_binarizada, cmap='gray')
    axes[0,1].set_title(f'Binarizada ({tipo_bin_nombre})')
    axes[0,1].axis('off')
    
    # Imagen con morfología
    axes[1,0].imshow(imagen_morfologica, cmap='gray')
    axes[1,0].set_title(f'Morfología (Kernel {kernel_size}x{kernel_size})')
    axes[1,0].axis('off')
    
    # Imagen final con forma y texto
    if len(imagen_final.shape) == 3:
        axes[1,1].imshow(cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB))
    else:
        axes[1,1].imshow(imagen_final, cmap='gray')
    axes[1,1].set_title(f'Resultado Final ({forma})')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar imagen final
    nombre_archivo = f"imagen_procesada_{clave_imagen}_{tipo_bin_nombre}_{forma}.png"
    cv2.imwrite(nombre_archivo, imagen_final)
    print(f"Imagen procesada guardada como: {nombre_archivo}")

def main():
    """Función principal con el menú"""
    print("¡Bienvenido al Sistema de Procesamiento de Imágenes Médicas!")
    
    while True:
        mostrar_menu_principal()
        opcion = input("\nSeleccione una opción: ").lower()
        
        if opcion == 'a':
            opcion_a_procesar_dicom()
        elif opcion == 'b':
            opcion_b_ingresar_paciente()
        elif opcion == 'c':
            opcion_c_ingresar_imagenes()
        elif opcion == 'd':
            opcion_d_transformacion_geometrica()
        elif opcion == 'e':
            opcion_e_procesamiento_imagenes()
        elif opcion == 'f':
            print("\n¡Gracias por usar el sistema! Hasta luego.")
            break
        else:
            print("Opción no válida. Por favor, seleccione una opción del menú.")
        
        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    # Verificar que se tienen las librerías necesarias
    try:
        import pydicom
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        print("Todas las librerías necesarias están instaladas.")
        main()
    except ImportError as e:
        print(f"Error: Falta instalar una librería: {e}")
        print("Instale las librerías necesarias:")
        print("pip install pydicom opencv-python numpy matplotlib")