import cv2
import dlib
import pygame
from scipy.spatial import distance as dist
from config import get_predictor_path, get_sound_file_path, get_ear_threshold


def initialize_video(video_source=0):
    """Inicializa a captura de vídeo."""
    video = cv2.VideoCapture(video_source)
    if not video.isOpened():
        raise Exception("Erro ao abrir a câmera.")
    return video


def initialize_detector():
    """Inicializa o detector de faces e o preditor de pontos faciais."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(get_predictor_path())
    return detector, predictor


def calcular_ear(pontos):
    """Calcula o Eye Aspect Ratio (EAR) para estimar a abertura dos olhos."""
    A = dist.euclidean(pontos[1], pontos[5])
    B = dist.euclidean(pontos[2], pontos[4])
    C = dist.euclidean(pontos[0], pontos[3])
    return (A + B) / (2.0 * C)


def play_sound(alerta_sono):
    """Controla a reprodução do som de alerta."""
    pygame.mixer.init()
    pygame.mixer.music.load(get_sound_file_path())
    if alerta_sono and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # Toca em loop
    elif not alerta_sono and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
