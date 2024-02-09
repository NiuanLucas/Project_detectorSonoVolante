# perclos_detector.py
# Importa as bibliotecas necessárias para operações de processamento de imagem e detecção.
import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear
from utils import SoundPlayer
from config import get_ear_threshold, get_tempo_alerta, get_sound_file_path
import time

# Descrição:
# Este script detecta sonolência ao volante usando a técnica PERCLOS (Percentage of Eye Closure).
# O PERCLOS mede a porcentagem de tempo que os olhos estão fechados durante um determinado período,
# e um valor acima do limiar pode indicar que o motorista está ficando sonolento.

# Define a função principal que detecta sonolência usando a métrica PERCLOS.

DROWSINESS_PERCENTAGE = 30
RESET_PERCLOS_MEASUREMENT = 60

# Retorna o caminho para o arquivo de som
sound_player = SoundPlayer(get_sound_file_path())


def draw_eyes_points(frame, shape, cor_rgb):
    # Converte a cor de RGB para BGR, pois o OpenCV usa BGR
    cor_bgr = (cor_rgb[2], cor_rgb[1], cor_rgb[0])
    # Loop pelos pontos dos olhos (de 36 a 47, incluindo ambos os olhos).
    for n in range(36, 48):
        # Coordenadas do ponto.
        x = shape.part(n).x
        y = shape.part(n).y
        # Desenha uma pequena bolinha (círculo) em cada ponto com a cor especificada.
        cv2.circle(frame, (x, y), 2, cor_bgr, -1)


def detect_drowsiness_perclos(video_source=0):
    # Inicializa o detector de faces e o preditor de pontos faciais.
    detector, predictor = initialize_detector()

    # Inicializa a captura de vídeo.
    video = initialize_video(video_source)

    # Obtém o valor do limiar EAR do arquivo de configuração.
    ear_threshold = get_ear_threshold()

    # Obtém o valor do tempo de alerta do arquivo de configuração.
    tempo_alerta = get_tempo_alerta()

    # Inicializa contadores para calcular o PERCLOS.
    frames_fechados = 0
    total_frames = 0
    frame_count = 0  # Correção: Inicializa frame_count para calcular o FPS
    start_time = time.time()
    alerta_sono_frame = False
    alerta_sono = False
    perclos = 0
    fps = 0
    average_perclos = 0
    perclos_values = []  # Lista para armazenar os valores de PERCLOS

    # Loop para processar cada frame do vídeo.
    while True:
        # Lê o próximo frame do vídeo.
        ret, frame = video.read()
        # Se não houver frame, encerra o loop.
        if not ret:
            break

        # Reproduz ou para o som de alerta baseado no estado atual de alerta de sonolência.
        sound_player.play_sound(alerta_sono)

        # Converte o frame para escala de cinza para facilitar a detecção de faces.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces no frame.
        faces = detector(gray)

        # Loop para processar cada face detectada.
        for face in faces:
            # Identifica os pontos faciais na face detectada.
            shape = predictor(gray, face)
            # Exemplo de chamada da função com a cor vermelha em formato RGB
            cor_rgb_vermelho = (220, 255, 50)  # Vermelho
            draw_eyes_points(frame, shape, cor_rgb_vermelho)
            # Calcula o EAR para ambos os olhos, somando e dividindo por 2 para obter a média.
            ear = (calcular_ear([(shape.part(n).x, shape.part(n).y) for n in range(36, 42)]) +
                   calcular_ear([(shape.part(n).x, shape.part(n).y) for n in range(42, 48)])) / 2.0

            # Se o EAR estiver abaixo do limiar, incrementa o contador de frames com olhos fechados.
            if ear < ear_threshold:
                frames_fechados += 1

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time

            if total_frames > 0:
                perclos = (frames_fechados / total_frames) * 100
                # Adiciona o valor atual de PERCLOS à lista
                perclos_values.append(perclos)
                alerta_sono_frame = perclos > DROWSINESS_PERCENTAGE

                # Calcula a média de PERCLOS com base nos valores armazenados
                average_perclos = sum(perclos_values) / len(perclos_values)
                alerta_sono = average_perclos > DROWSINESS_PERCENTAGE

                # Resetar a lista a cada 60 medições para começar um novo conjunto de dados
                if len(perclos_values) >= RESET_PERCLOS_MEASUREMENT:
                    perclos_values = []

                frames_fechados = 0
                total_frames = 0
        else:
            total_frames += 1

        # Adiciona informações visuais ao vídeo para feedback ao usuário.
        fonte = cv2.FONT_HERSHEY_COMPLEX
        estado_sono = "Sono Detectado!" if alerta_sono_frame else "Sono Nao Detectado"

        cv2.rectangle(frame, (10, 20), (260, 90), (0, 0, 0), -1)
        cv2.putText(frame, estado_sono, (10, 50), fonte, 0.7,
                    (0, 0, 255) if alerta_sono_frame else (0, 255, 0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}%",
                    (10, 80), fonte, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (10, 350), (460, 480), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Frames Fechados/Total: {frames_fechados}/{total_frames}", (10, 380), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Media PERCLOS: {average_perclos:.2f}%",
                    (10, 410), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Medicoes PERCLOS: {len(perclos_values)}",
                    (10, 440), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Video FPS: {fps:.2f}",
                    (10, 470), fonte, 0.7, (255, 255, 255), 2)

        # Exibe o frame processado.
        cv2.imshow("Deteccao de Sono ao Volante - PERCLOS", frame)

        # Permite a interrupção do loop pressionando 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera o dispositivo de captura e fecha todas as janelas abertas.
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_drowsiness_perclos()
