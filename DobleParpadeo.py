import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
#from findpeaks import findpeaks
from scipy.signal import find_peaks
from scipy.signal import medfilt
import pickle




def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter,fpsdisp):
     aux_image = np.zeros(frame.shape, np.uint8)
     output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)
     cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)
     cv2.rectangle(output, (202, 0), (265, 50), (255, 0, 0),2)
     cv2.putText(output, "Num. Parpadeos:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
     cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)
     cv2.putText(output, fpsdisp, (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
     
     return output

def eye_aspect_ratio(coordinates):
     d_A = np.linalg.norm(np.array(coordinates[3]) - np.array(coordinates[13]))
     d_B = np.linalg.norm(np.array(coordinates[5]) - np.array(coordinates[11]))
     d_C = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[8]))
     #print("Distancia horizontal:", d_C)
     return (d_A + d_B) / (2 * d_C), d_C

def calculate_threshold(d_C):
     """
     Calculate the EAR threshold based on the horizontal distance of the eye in pixels (d_C).
     """
     # Define the ranges for d_C and EAR_THRESH.
     d_C_range = [35, 7]
     EAR_THRESH_range = [0.29, 0.19]

     # Calculate the proportion of d_C within its range.
     proportion = (d_C - d_C_range[0]) / (d_C_range[1] - d_C_range[0])

     # Use the proportion to calculate EAR_THRESH within its range.
     EAR_THRESH = EAR_THRESH_range[0] + proportion * (EAR_THRESH_range[1] - EAR_THRESH_range[0])

     # Ensure EAR_THRESH stays within its range.
     EAR_THRESH = max(min(EAR_THRESH, EAR_THRESH_range[0]), EAR_THRESH_range[1])

     return EAR_THRESH

def rising_edge(data, thresh):
     sign = data >= thresh
     pos = np.where(np.convolve(sign, [1, -1]) == 1)
     return pos

def representar_grafico(pts_ear_array, pts_ear_array_smooth, line1, line2):
     # Create a new figure and two subplots: one for the original data and one for the smoothed data
     global figure
     pts_or = np.linspace(0, 1, len(pts_ear_array))
     pts_smooth = np.linspace(0, 1, len(pts_ear_array_smooth))
     # pts_ear_array = pts_ear_array.flatten()
     # pts_ear_array_smooth = pts_ear_array_smooth.flatten()
     if line1 == []:
          plt.style.use("ggplot")
          plt.ion()
          figure, (ax1, ax2) = plt.subplots(2, 1)
          #line1, = ax1.plot([], [], 'b-', label='original data')
          line1, = ax1.plot(pts_or, pts_ear_array, 'b-', label='original data')
          #line2, = ax2.plot([], [], 'r-', label='smoothed data')
          line2, = ax2.plot(pts_smooth, pts_ear_array_smooth, 'r-', label='smoothed data')
          # Add a legend to each subplot
          ax1.legend()
          ax2.legend()
          ax1.set_xlim(0, 1)
          ax1.set_ylim(-0.45, -0.1)
          ax2.set_xlim(0, 1)
          ax2.set_ylim(-0.45, -0.1)
     else:
          line1.set_ydata(pts_ear_array)
          line2.set_ydata(pts_ear_array_smooth)
          figure.canvas.draw()
          figure.canvas.flush_events()
          if cv2.waitKey(5) & 0xFF == 27:
               cv2.destroyAllWindows()

     return line1, line2


# Inicializa la captura de video

cap = cv2.VideoCapture(0) #distancia máxima 1.80 metros mao meno

# DEFINIENDO LAS VARIABLES QUE SE UTILIZARÁN 
# Definir el máximo número de valores que deseas almacenar
MAX_VALORES = 1000
mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] #[33, 160, 158, 133, 153, 144] Simpler version
index_right_eye = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382] #[362, 385, 387, 263, 373, 380] Simpler version
#EAR_THRESH = 0.27 #variable, si está mas cerca o mas lejos deberia variar 0.19
# Definir el número de frames consecutivos que deben tener un valor de EAR inferior a EAR_THRESH para considerar que un usuario está parpadeando de forma intencional
NUM_FRAMES = 15
aux_counter = 0
blink_counter = 0
last_activationframe=0
activationframe=False
segundoParpadeo=False
framesBWblinks=0
# Inicializa una lista vacía para almacenar los valores de ear
line1 = []
line2 = []
pts_ear = deque(maxlen=64)
i = 0
a=0
num_blinks=0
# Parámetros para mostrar los FPS en pantalla
fc=0
display_time=3
FPS=0
start_time=time.time()
# Parámetros para suavizado de la señal
window_size = 5
# Parámetros para representar gráficos
peak_detected = False

with mp_face_mesh.FaceMesh(
     static_image_mode=False,
     max_num_faces=1) as face_mesh:
     with open('ear_values.pkl', 'ab') as f:
          while True:
               ret, frame = cap.read()
               #Display FPS
               fc+=1
               TIME = time.time() - start_time
               if (TIME) >= display_time:
                    FPS = fc / (TIME)
                    fc = 0
                    start_time = time.time()
               
               fps_disp = "FPS: "+str(FPS)[:5]
               
               
               if ret == False:
                    break
               frame = cv2.flip(frame, 1)
               height, width, _ = frame.shape
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               results = face_mesh.process(frame_rgb)

               coordinates_left_eye = []
               coordinates_right_eye = []

               if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                         for index in index_left_eye:
                              x = int(face_landmarks.landmark[index].x * width)
                              y = int(face_landmarks.landmark[index].y * height)
                              coordinates_left_eye.append([x, y])
                              cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                              cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                         for index in index_right_eye:
                              x = int(face_landmarks.landmark[index].x * width)
                              y = int(face_landmarks.landmark[index].y * height)
                              coordinates_right_eye.append([x, y])
                              cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                              cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                    ear_left_eye, distancia_horizontal_l = eye_aspect_ratio(coordinates_left_eye)
                    ear_right_eye, distancia_horizontal_r = eye_aspect_ratio(coordinates_right_eye)
                    ear = (ear_left_eye + ear_right_eye)/2
                    EAR_THRESH = (calculate_threshold(distancia_horizontal_l) + calculate_threshold(distancia_horizontal_r))/2
                    #print("ear_thresh:", EAR_THRESH)
                    

                    #Retardo forzado de activación
                    if activationframe is True:
                         last_activationframe+=1
                         if last_activationframe >7:
                              activationframe=False
                              last_activationframe=0

                    #temporizador de activaciones
                    if segundoParpadeo is True:
                         framesBWblinks+=1
                         if framesBWblinks>62:
                              framesBWblinks=0
                              segundoParpadeo=False
                              print("Segundo parpadeo no realizado")

                    # Ojos cerrados
                    if ear < EAR_THRESH:
                         aux_counter += 1
                         dormido = False
                    else:
                         if (NUM_FRAMES+30)> aux_counter >= NUM_FRAMES:
                         # elif aux_counter <= (NUM_FRAMES+30):
                              blink_counter =blink_counter
                              # print("Parpadeo exitoso")
                         elif aux_counter > (NUM_FRAMES+30):
                              # print("Ojo cerrado por mucho tiempo")
                              aux_counter = 0
                         # else:
                         #      dormido = True
                         aux_counter = 0
                         
                              
                    frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter,fps_disp)
                    pickle.dump(ear, f)
                    pts_ear.append(ear)
                    # Convierte la lista de valores de EAR en un array de numpy
                    pts_ear_array = -np.array(pts_ear)
                    

                    # Espera a tener suficientes valores antes de buscar picos, para evitar falsos positivos
                    if len(pts_ear) > 63: #cambiar si se quiere representar graficamente de forma continua
                         # Encuentra picos en los valores de EAR usando la función find_peaks
                         #line1 = plotting_ear(pts_ear, line1)
                         pts_ear_array_smooth = medfilt(pts_ear_array, kernel_size=3)
                         #pts_ear_array_smooth = np.convolve(pts_ear_array, np.ones(window_size)/window_size, mode='valid') 
                         # line1, line2 = representar_grafico(pts_ear_array, pts_ear_array_smooth, line1, line2) 
                         # Display the plot
                         peaks, diccionario = find_peaks(pts_ear_array_smooth,distance=40,  width=15, prominence=0.030)#  , distance=30, threshold=0.0015, wlen=15, width=15) #esto es lo que hay que modificar
                         # Imprime los índices de los picos encontrados (si hay alguno)
                         if peaks.size > 0:
                              if any(peaks>30) and not peak_detected:
                              #     DIFERENTE PARA SIN GRÁFICAS
                                   # if not peak_detected and len(peaks) > 1:
                                   #      print("Varios picos consecutivos. peaks:", peaks)
                                   #      #podría ser necesario cambiarlo o tenerlo en cuenta
                                   # else:
                                   #print("PICO DETECTADO")
                                   #print("Picos encontrados en los índices:", peaks)
                                   #print("Prominencias de los picos:", diccionario['prominences'])
                                   #print("Bases izquierdas de los picos:", diccionario['left_bases'])
                                   #print("Bases derechas de los picos:", diccionario['right_bases'], "\n")
                                   # print("Threshold", diccionario['left_thresholds'], diccionario['right_thresholds'], "\n")
                                   if activationframe is False:
                                        if segundoParpadeo ==True:
                                             segundoParpadeo=False
                                             framesBWblinks=0
                                             num_blinks+=1
                                             blink_counter+=1
                                             print("Segundo parpadeo detectado.Activación enviada")

                                             print("Numero de activaciones: ", num_blinks,"\n")
                                        else:
                                             segundoParpadeo=True
                                             print("Primer parpadeo detectado")
                                        activationframe =True
                                   peak_detected = True
                              elif peak_detected and any(peaks<=30):
                                   #print(" RESETEO SEÑAL ENVIADA. Peaks: ", peaks)
                                   #aquí es simplemente para resetear el trigger de peak_detected
                                   peak_detected = False         

               cv2.imshow("Frame", frame)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()

with open('ear_values.pkl', 'rb') as f:
    ear_values = []
    while True:
        try:
            ear_values.append(-pickle.load(f))
        except EOFError:
            break
# Plot pts_ear and pts_ear_array_smooth in the same figure
plt.figure()

# Plot ear_values in red
ear_values = ear_values[-(len(pts_ear_array_smooth)):]
plt.plot(ear_values, color='red', label='pts_ear')

# Plot pts_ear_array_smooth in blue
plt.plot(pts_ear_array_smooth, color='blue', label='pts_ear_array_smooth')

plt.title('pts_ear and pts_ear_array_smooth')
plt.legend()  # Add a legend to distinguish the two lines
plt.show()

# Close all plots
plt.close('all')
# # Plot pts_ear
# plt.figure()
# ear_values = ear_values[-100:]
# plt.plot(ear_values)
# plt.title('pts_ear')

# # Plot pts_ear_array_smooth
# plt.figure()
# plt.plot(pts_ear_array_smooth)
# plt.title('pts_ear_array_smooth')
# plt.show()

# # Close all plots
# plt.close('all')