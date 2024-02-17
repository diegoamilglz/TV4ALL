import time
import cv2
import numpy
from skimage.feature import texture
from skimage.filters import rank
from skimage import data, feature, filters

#habría que probar combinando al añadir el tamañno del recuadro



# Cargar clasificadores pre-entrenados
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

# Función para extraer características de los ojos
def extract_eye_texture_features(eye):
    # Preprocesamiento de la imagen del ojo (conversión a escala de grises)
    #eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    # Calcular matriz de co-ocurrencia de niveles de gris (GLCM)
    distances = [1]  # Distancia de pixel para calcular la GLCM (1 píxel en este caso)
    angles = [0]     # Dirección de pixel para calcular la GLCM (horizontal en este caso)
    glcm = feature.graycomatrix(eye, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calcular estadísticas de textura a partir de la GLCM
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]  # Homogeneidad
    energy = feature.graycoprops(glcm, 'energy')[0, 0]            # Energía
    correlation = feature.graycoprops(glcm, 'correlation')[0, 0]  # Correlación

    return homogeneity, energy, correlation
# Capturar video desde la cámara
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # Extraer características de textura del ojo
            homogeneity, energy, correlation = extract_eye_texture_features(eye)
            
          # Clasificar si el ojo está abierto o cerrado (ejemplo simplificado)
            if homogeneity > 0.25:  # Umbral arbitrario
                eye_status = "Abierto"
            else:
                eye_status = "Cerrado"
            
            # Dibujar rectángulo alrededor del ojo y mostrar estado
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(roi_color, eye_status, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.putText(roi_color, homogeneity, (ex, ey + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            intervalo_tiempo = 1  # Por ejemplo, cada 5 segundos

            print(correlation)   
                
    cv2.imshow('Eye Status', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
