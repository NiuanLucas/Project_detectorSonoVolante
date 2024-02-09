# Importa as bibliotecas necessárias para operação do script.
import cv2  # Usada para operações de captura e processamento de vídeo.
import dlib  # Usada para detecção de faces e pontos faciais.
import pygame  # Usada para reprodução de som.
# Usada para cálculo de distâncias euclidianas.
from scipy.spatial import distance as dist
# Importa funções de configuração para acessar parâmetros específicos.
from config import get_predictor_path, get_sound_file_path, get_ear_threshold

# Inicializa a captura de vídeo a partir de uma fonte (por padrão, a webcam principal).


def initialize_video(video_source=0):
    video = cv2.VideoCapture(video_source)  # Cria um objeto VideoCapture.
    if not video.isOpened():  # Verifica se a captura foi inicializada corretamente.
        # Lança uma exceção se não conseguir abrir.
        raise Exception("Erro ao abrir a câmera.")
    return video  # Retorna o objeto de captura de vídeo.

# Inicializa o detector de faces e o preditor de pontos faciais usando a biblioteca dlib.


def initialize_detector():
    detector = dlib.get_frontal_face_detector()  # Cria um detector de faces.
    # Carrega o preditor de pontos faciais usando o caminho fornecido pela função de configuração.
    predictor = dlib.shape_predictor(get_predictor_path())
    return detector, predictor  # Retorna o detector e o preditor.

# Calcula o Eye Aspect Ratio (EAR) para estimar a abertura dos olhos com base nos pontos fornecidos.


def calcular_ear(pontos):
    """
    Calcula o Eye Aspect Ratio (EAR) para um olho, baseado nas coordenadas dos seus pontos.

    Argumentos:
    pontos -- Lista de tuplas contendo as coordenadas (x, y) dos seis pontos ao redor de um olho.

    Retorna:
    EAR -- O Eye Aspect Ratio calculado para o olho.
    """

    # Primeiro, calculamos as distâncias verticais entre os pontos superiores e inferiores das pálpebras.
    # Distância vertical superior: entre o ponto 2 e o ponto 6.
    distancia_vertical_superior = dist.euclidean(pontos[1], pontos[5])

    # Distância vertical inferior: entre o ponto 3 e o ponto 5.
    distancia_vertical_inferior = dist.euclidean(pontos[2], pontos[4])

    # A média das distâncias verticais é calculada somando as duas distâncias e dividindo por 2.
    media_distancias_verticais = (
        distancia_vertical_superior + distancia_vertical_inferior) / 2.0

    # Em seguida, calculamos a distância horizontal entre os cantos do olho (pontos 1 e 4).
    distancia_horizontal = dist.euclidean(pontos[0], pontos[3])

    # Finalmente, calculamos o EAR como a razão da média das distâncias verticais pela distância horizontal.
    # Isso nos dá uma medida proporcional de quão abertos ou fechados estão os olhos.
    EAR = media_distancias_verticais / distancia_horizontal

    return EAR

# Controla a reprodução do som de alerta usando a biblioteca pygame.


class SoundPlayer:
    def __init__(self, sound_file_path):
        pygame.mixer.init()
        self.sound_file_path = sound_file_path
        self.som_iniciado = False

    def play_sound(self, alerta_sono):
        # Carrega o arquivo de som de alerta se não estiver ocupado ou se o som ainda não foi carregado
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.sound_file_path)

        # Se um alerta de sono é ativo e o som não está sendo reproduzido,
        if alerta_sono and not self.som_iniciado:
            pygame.mixer.music.play()  # Inicia a reprodução do som.
            self.som_iniciado = True  # Atualiza a variável para indicar que o som foi iniciado
        elif not alerta_sono:
            self.som_iniciado = False  # Reseta a variável quando não há alerta de sono

        # Se o som terminou de ser reproduzido, resete a variável som_iniciado
        if not pygame.mixer.music.get_busy():
            self.som_iniciado = False
