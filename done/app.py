import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carga tu modelo
modelito = load_model('modelito2.h5')

# Diccionario de emociones
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

        # Mostrar la emoción en la pantalla
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Imprimir la emoción en la consola
        print("Detected Emotion: ", emotion_label)
    # Mostrar el frame con la detección y la emoción
    cv2.imshow('Emotion Detector', frame)

    # Terminar con la tecla 'q'
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

