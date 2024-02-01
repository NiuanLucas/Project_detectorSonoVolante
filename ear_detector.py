# ear_detector.py

import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear, play_sound
from config import get_ear_threshold, get_tempo_alerta

# Descrição:
# Este script detecta sonolência ao volante usando a técnica EAR (Eye Aspect Ratio).
# O EAR é um indicador que mede a abertura dos olhos, e uma diminuição repentina no EAR
# pode indicar que o motorista está ficando sonolento.

def detectar_sonolencia_ear(video_source=0):
    # Inicialização do detector de face e preditor facial
    detector, predictor = initialize_detector()

    # Inicialização do vídeo
    video = initialize_video(video_source)

    # Obtenção do limiar EAR a partir do arquivo de configuração
    ear_threshold = get_ear_threshold()

    # Obtenção do tempo de alerta a partir do arquivo de configuração
    tempo_alerta = get_tempo_alerta()

    # Obtém a taxa de quadros por segundo (FPS) do vídeo
    fps = video.get(cv2.CAP_PROP_FPS)

    # Inicialização de variáveis
    contador_fechados = 0
    alerta_sono = False
    tempo_acumulado_fechado = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        ear = 0  # Garante que ear tenha um valor inicial em cada frame

        for face in faces:
            shape = predictor(gray, face)
            pontos_olho_esquerdo = [
                (shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
            pontos_olho_direito = [(shape.part(n).x, shape.part(n).y)
                                   for n in range(42, 48)]
            ear_esquerdo = calcular_ear(pontos_olho_esquerdo)
            ear_direito = calcular_ear(pontos_olho_direito)
            ear = (ear_esquerdo + ear_direito) / 2.0

            # Adicionando indicadores visuais para os olhos
            for pontos in [pontos_olho_esquerdo, pontos_olho_direito]:
                hull = cv2.convexHull(np.array(pontos))
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

            if ear < ear_threshold:
                contador_fechados += 1/fps
                tempo_acumulado_fechado += 1/fps
            else:
                if contador_fechados >= tempo_alerta:
                    alerta_sono = True
                    play_sound(True)
                contador_fechados = 0

        # Adicionando fundo com opacidade para os textos
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        # Fundo para texto de estado de sono
        texto_estado = "Sono Detectado!" if alerta_sono else "Sono Nao Detectado"
        cv2.rectangle(frame, (10, 20), (400, 120), (0, 0, 0), -1)
        cv2.putText(frame, texto_estado, (20, 50), fonte, 0.7,
                    (0, 0, 255) if alerta_sono else (0, 255, 0), 2)

        # Fundo para texto do valor EAR
        texto_ear = f"EAR: {ear:.2f}"
        cv2.putText(frame, texto_ear, (20, 80), fonte, 0.7, (255, 255, 255), 2)

        # Fundo para texto do tempo com olhos fechados
        texto_tempo_fechado = f"Tempo com olhos fechados: {tempo_acumulado_fechado:.2f}s"
        cv2.putText(frame, texto_tempo_fechado, (20, 110),
                    fonte, 0.7, (255, 255, 255), 2)

        cv2.imshow("Deteccao de Sono ao Volante - EAR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_sonolencia_ear()
