import cv2

# Cargamos el clasificador pre-entrenado para la detección de caras y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascadel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
eye_cascader = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
#eye_cascadel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicializamos la captura de vídeo desde la cámara
cap = cv2.VideoCapture(1)

# Factor de zoom
zoom_factor = 0.4  # Ajusta este valor según sea necesario. Con este valor, aproximadamente se logran detectar ambos ojos a 3.5~ metros
#condiciones de buena iluminación (con sol), a la misma altura (77 cm)

while True:
    # Capturamos un fotograma desde la cámara
    ret, frame = cap.read()
    
    # Obtenemos las dimensiones originales del fotograma
    height, width = frame.shape[:2]

    # Calculamos las nuevas dimensiones después del zoom
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)

    # Redimensionamos el fotograma para aplicar el zoom
    zoom_frame = cv2.resize(frame, (new_width, new_height))

    # Convertimos el fotograma a escala de grises
    gray = cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2GRAY)

    # Detectamos las caras en el fotograma
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) #clave, editar para afinar lo que se puede hacer con el minipc (potencia de procesado/fiabilidad detección)

    # Por cada cara detectada, intentamos detectar los ojos
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = zoom_frame[y:y+h, x:x+w]
        left_eye = eye_cascadel.detectMultiScale(roi_gray)
        right_eye = eye_cascadel.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in left_eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
        for (ex,ey,ew,eh) in right_eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(zoom_frame,"Parpadea?",(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostramos el fotograma con los rectángulos dibujados alrededor de los ojos
    cv2.imshow('Eye Detection', zoom_frame)

    # Esperamos a que se presione la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos
cap.release()
cv2.destroyAllWindows()