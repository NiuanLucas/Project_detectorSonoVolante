# ear_detector.py
# Importa as bibliotecas necessárias para o funcionamento do script.
import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear
from utils import SoundPlayer
from config import get_ear_threshold, get_tempo_alerta, get_sound_file_path

# Descrição:
# Este script detecta sonolência ao volante usando a técnica EAR (Eye Aspect Ratio).
# O EAR é um indicador que mede a abertura dos olhos, e uma diminuição repentina no EAR
# pode indicar que o motorista está ficando sonolento.

# Retorna o caminho para o arquivo de som
sound_player = SoundPlayer(get_sound_file_path())

# Define a função principal que detecta sonolência baseada no EAR.


def detectar_sonolencia_ear(video_source=0):
    # Inicializa o detector de faces e o preditor de pontos faciais.
    detector, predictor = initialize_detector()

    # Inicializa a captura de vídeo.
    video = initialize_video(video_source)

    # Lê o valor do limiar EAR do arquivo de configuração.
    ear_threshold = get_ear_threshold()

    # Lê o valor do tempo de alerta do arquivo de configuração.
    tempo_alerta = get_tempo_alerta()

    # Obtém a taxa de quadros por segundo (FPS) do vídeo.
    fps = video.get(cv2.CAP_PROP_FPS)

    # Inicializa variáveis para controle de detecção.
    contador_fechados = 0
    alerta_sono = False
    tempo_acumulado_fechado = 0

    # Loop para processar cada frame do vídeo.
    while True:
        # Lê o próximo frame do vídeo.
        ret, frame = video.read()
        # Se não houver frame, encerra o loop.
        if not ret:
            break

        # Converte o frame para escala de cinza para detecção de faces.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecta faces no frame.
        faces = detector(gray)
        # Inicializa a variável EAR para este frame.
        ear = 0

        # Processa cada face detectada.
        for face in faces:
            # Identifica os pontos faciais na face detectada.
            shape = predictor(gray, face)
            # Extrai os pontos dos olhos esquerdo e direito.
            pontos_olho_esquerdo = [
                (shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
            pontos_olho_direito = [(shape.part(n).x, shape.part(n).y)
                                   for n in range(42, 48)]
            # Calcula o EAR para ambos os olhos.
            ear_esquerdo = calcular_ear(pontos_olho_esquerdo)
            ear_direito = calcular_ear(pontos_olho_direito)
            # Calcula a média do EAR dos dois olhos.
            ear = (ear_esquerdo + ear_direito) / 2.0

            # Desenha um contorno convexo ao redor dos olhos detectados para visualização.
            for pontos in [pontos_olho_esquerdo, pontos_olho_direito]:
                hull = cv2.convexHull(np.array(pontos))
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

            # Verifica se o EAR está abaixo do limiar de sonolência.
            if ear < ear_threshold:
                # Incrementa o contador de frames com olhos fechados.
                contador_fechados += 1/fps
                # Acumula o tempo com olhos fechados.
                tempo_acumulado_fechado += 1/fps
            else:
                # Se o contador exceder o tempo de alerta, ativa o alerta de sonolência.
                if contador_fechados >= tempo_alerta:
                    alerta_sono = True
                    # Toca o som de alerta.

                    sound_player.play_sound(True)
                # Reseta o contador de frames com olhos fechados.
                contador_fechados = 0

        # Adiciona uma interface gráfica para indicar o estado de detecção.
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        texto_estado = "Sono Detectado!" if alerta_sono else "Sono Não Detectado"
        cv2.rectangle(frame, (10, 20), (400, 120), (0, 0, 0), -1)
        cv2.putText(frame, texto_estado, (20, 50), fonte, 0.7,
                    (0, 0, 255) if alerta_sono else (0, 255, 0), 2)
        texto_ear = f"EAR: {ear:.2f}"
        cv2.putText(frame, texto_ear, (20, 80), fonte, 0.7, (255, 255, 255), 2)
        texto_tempo_fechado = f"Tempo com olhos fechados: {tempo_acumulado_fechado:.2f}s"
        cv2.putText(frame, texto_tempo_fechado, (20, 110),
                    fonte, 0.7, (255, 255, 255), 2)

        # Exibe o frame processado.
        cv2.imshow("Deteccao de Sono ao Volante - EAR", frame)

        # Permite sair do loop pressionando 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera o dispositivo de captura e fecha todas as janelas.
    video.release()
    cv2.destroyAllWindows()


# Permite que o script seja executado diretamente.
if __name__ == "__main__":
    detectar_sonolencia_ear()
