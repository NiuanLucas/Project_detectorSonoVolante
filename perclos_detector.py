# perclos_detector.py

import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear, play_sound
from config import get_ear_threshold, get_tempo_alerta

# Descrição:
# Este script detecta sonolência ao volante usando a técnica PERCLOS (Percentage of Eye Closure).
# O PERCLOS mede a porcentagem de tempo que os olhos estão fechados durante um determinado período,
# e um valor acima do limiar pode indicar que o motorista está ficando sonolento.


def detectar_sonolencia_perclos(video_source=0):
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
    frames_fechados = 0
    total_frames = 0
    alerta_sono = False

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Chama a função para reproduzir o som de alerta
        play_sound(alerta_sono)

        # Inicialização da variável perclos no início do loop
        perclos = 0  # Garante que perclos tenha um valor antes de ser usado

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            pontos = [(shape.part(n).x, shape.part(n).y)
                      for n in range(36, 48)]
            ear = (calcular_ear(pontos[0:6]) +
                   calcular_ear(pontos[6:12])) / 2.0

            if ear < ear_threshold:
                frames_fechados += 1

        total_frames += 1

        if total_frames % fps == 0:  # A cada segundo, verifica o PERCLOS
            perclos = (frames_fechados / total_frames) * 100
            alerta_sono = perclos > 20  # Supondo que 20% seja o limiar de alerta de sonolência

            # Resetando contadores a cada segundo para análise contínua
            frames_fechados = 0
            total_frames = 0

        # Adicionando métricas e indicadores visuais na tela
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        texto_perclos = f"PERCLOS: {perclos:.2f}%"
        texto_frames = f"Frames Fechados/Total: {frames_fechados}/{total_frames}"
        estado_sono = "Sono Detectado!" if alerta_sono else "Sono Nao Detectado"

        # Fundo para textos
        cv2.rectangle(frame, (10, 20), (450, 100), (0, 0, 0), -1)
        cv2.putText(frame, estado_sono, (10, 40), fonte, 0.7,
                    (0, 0, 255) if alerta_sono else (0, 255, 0), 2)
        cv2.putText(frame, texto_perclos, (10, 60),
                    fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, texto_frames, (10, 80),
                    fonte, 0.7, (255, 255, 255), 2)

        cv2.imshow("Deteccao de Sono ao Volante - PERCLOS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_sonolencia_perclos()
