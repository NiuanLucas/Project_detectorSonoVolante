import cv2
import numpy as np
from utils import initialize_video, initialize_detector, calcular_ear, play_sound
from config import get_ear_threshold, get_tempo_alerta


def detectar_sonolencia_piscadas(video_source=0):
    detector, predictor = initialize_detector()
    video = initialize_video(video_source)
    ear_threshold = get_ear_threshold()

    piscadas = 0
    tempo_total_piscadas = 0
    total_duracao_piscadas = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        play_sound(media_duracao_piscadas > 0.4)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            pontos_olho_esquerdo = [
                (shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
            pontos_olho_direito = [(shape.part(n).x, shape.part(n).y)
                                   for n in range(42, 48)]
            ear_esquerdo = calcular_ear(pontos_olho_esquerdo)
            ear_direito = calcular_ear(pontos_olho_direito)
            ear = (ear_esquerdo + ear_direito) / 2.0

            if ear < ear_threshold and tempo_total_piscadas == 0:
                tempo_total_piscadas = cv2.getTickCount()  # Marca o início da piscada
            elif ear >= ear_threshold and tempo_total_piscadas > 0:
                duracao = (cv2.getTickCount() -
                           tempo_total_piscadas) / cv2.getTickFrequency()
                total_duracao_piscadas += duracao
                piscadas += 1
                tempo_total_piscadas = 0  # Reseta o contador para a próxima piscada

        media_duracao_piscadas = total_duracao_piscadas / piscadas if piscadas > 0 else 0

        # Adicionando informações na tela
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (10, 20), (550, 120),
                      (0, 0, 0), -1)  # Fundo para textos
        cv2.putText(frame, f"Piscadas: {piscadas}",
                    (10, 50), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Duracao Media das Piscadas: {media_duracao_piscadas:.2f}s", (
            10, 80), fonte, 0.7, (255, 255, 255), 2)

        alerta = "Sonolencia Detectada!" if media_duracao_piscadas > 0.4 else "Sonolencia Nao Detectada"
        cv2.putText(frame, alerta, (10, 110), fonte, 0.7, (0, 0, 255)
                    if media_duracao_piscadas > 0.4 else (0, 255, 0), 2)

        cv2.imshow("Deteccao de Sono ao Volante - Piscadas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Sai do loop se 'q' for pressionado
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_sonolencia_piscadas()