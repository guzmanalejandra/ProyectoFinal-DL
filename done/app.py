import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Carga tu modelo
modelito = load_model('modelito2.h5')

# Diccionario de emociones y contador de emociones
emotion_dict = {0: "Angry", 1: "Disgust", 2: "neutral", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emotion_counter = {emotion: 0 for emotion in emotion_dict.values()}

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def plot_emotions(emotion_counter):
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())

    plt.bar(emotions, counts, color='blue')
    plt.xlabel('Emotions')
    plt.ylabel('Counts')
    plt.title('Emotion Distribution')
    plt.show()

# Colores para las emociones
emotion_colors = {
    "Angry": (0, 0, 255),     # Rojo
    "Disgust": (0, 255, 0),   # Verde
    "Neutral": (200, 200, 200),    # Púrpura
    "Happy": (0, 255, 255),   # Amarillo
    "Sad": (255, 0, 0),       # Azul
    "Surprise": (255, 255, 0),# Cian
    "Neutral": (200, 200, 200)# Gris
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clasificador de caras
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de caras
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Preprocesamiento para el modelo
        final_image = cv2.resize(roi_color, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        # Predicción de emociones
        Predictions = modelito.predict(final_image)
        emotion_index = np.argmax(Predictions)
        emotion_label = emotion_dict[emotion_index]

        # Elegir el color basado en la emoción
        color = emotion_colors[emotion_label]

        # Dibujar rectángulo y texto con el color correspondiente a la emoción
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Contar las emociones
        emotion_counter[emotion_label] += 1

    # Mostrar instrucciones en la pantalla
    cv2.putText(frame, "Presiona 'q' para salir", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar el frame con la detección y la emoción
    cv2.imshow('Emotion Detector', frame)

    # Terminar con la tecla 'q'
    if cv2.waitKey(2) & 0xFF == ord('q'):
        plot_emotions(emotion_counter)
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
