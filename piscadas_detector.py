# piscadas_detector.py

import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear
from utils import SoundPlayer
from config import get_ear_threshold, get_tempo_alerta, get_sound_file_path

# Descrição:
# Este script detecta sonolência ao volante contando o número de piscadas curtas dos olhos (blinks).
# O tempo médio de piscadas é calculado e, se for maior que um limiar, um alerta de sonolência é acionado.

# Define uma variável global para armazenar a média da duração das piscadas ao longo do tempo.
media_duracao_piscadas = 0

# Retorna o caminho para o arquivo de som
sound_player = SoundPlayer(get_sound_file_path())

# Define a função principal do script.


def detectar_sonolencia_piscadas(video_source=0):
    # Inicializa o detector de faces e o preditor de pontos faciais.
    detector, predictor = initialize_detector()
    # Inicializa a captura de vídeo pela câmera ou arquivo, conforme especificado.
    video = initialize_video(video_source)
    # Obtém o limiar do EAR, abaixo do qual um olho é considerado fechado.
    ear_threshold = get_ear_threshold()

    # Inicializa contadores para a análise das piscadas.
    piscadas = 0
    tempo_total_piscadas = 0
    total_duracao_piscadas = 0

    # Loop principal para processar cada frame do vídeo.
    while True:
        # Lê um frame do vídeo.
        ret, frame = video.read()
        # Se não conseguir ler o frame, encerra o loop.
        if not ret:
            break

        # Converte o frame para escala de cinza, facilitando a detecção de faces.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecta faces no frame convertido.
        faces = detector(gray)

        # Processa cada face detectada no frame.
        for face in faces:
            # Identifica os pontos faciais importantes para os olhos.
            shape = predictor(gray, face)
            pontos_olho_esquerdo = [
                (shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
            pontos_olho_direito = [(shape.part(n).x, shape.part(n).y)
                                   for n in range(42, 48)]

            # Calcula o EAR médio para ambos os olhos.
            ear_esquerdo = calcular_ear(pontos_olho_esquerdo)
            ear_direito = calcular_ear(pontos_olho_direito)
            ear = (ear_esquerdo + ear_direito) / 2.0

            # Verifica se o EAR está abaixo do limiar, indicando que os olhos estão fechados.
            if ear < ear_threshold and tempo_total_piscadas == 0:
                # Marca o início do tempo de uma piscada.
                tempo_total_piscadas = cv2.getTickCount()
            elif ear >= ear_threshold and tempo_total_piscadas > 0:
                # Calcula a duração da piscada ao abrir os olhos e acumula o total.
                duracao = (cv2.getTickCount() -
                           tempo_total_piscadas) / cv2.getTickFrequency()
                total_duracao_piscadas += duracao
                piscadas += 1
                # Reseta o contador para a próxima piscada.
                tempo_total_piscadas = 0

        # Calcula a média da duração das piscadas para evitar divisão por zero.
        media_duracao_piscadas = total_duracao_piscadas / piscadas if piscadas > 0 else 0

        # Adiciona informações visuais ao frame para feedback.
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        # Fundo para textos
        cv2.rectangle(frame, (10, 20), (550, 120), (0, 0, 0), -1)
        # Exibe o número total de piscadas e a duração média das piscadas.
        cv2.putText(frame, f"Piscadas: {piscadas}",
                    (10, 50), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Duração Média das Piscadas: {media_duracao_piscadas:.2f}s", (
            10, 80), fonte, 0.7, (255, 255, 255), 2)

        # Determina e exibe se há sonolência baseada na duração média das piscadas.
        alerta = "Sonolência Detectada!" if media_duracao_piscadas > 0.4 else "Sonolência Não Detectada"
        cv2.putText(frame, alerta, (10, 110), fonte, 0.7, (0, 0, 255)
                    if media_duracao_piscadas > 0.4 else (0, 255, 0), 2)

        # Exibe o frame processado.
        cv2.imshow("Detecção de Sono ao Volante - Piscadas", frame)

        # Permite sair do loop pressionando 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera recursos e fecha janelas abertas.
    video.release()
    cv2.destroyAllWindows()


# Permite executar a detecção diretamente, chamando a função definida.
if __name__ == "__main__":
    detectar_sonolencia_piscadas()
    # Executa o som de alerta se a média da duração das piscadas indicar sonolência.
    sound_player.play_sound(media_duracao_piscadas > 0.4)
