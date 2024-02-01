from configparser import ConfigParser

# Inicializa o objeto ConfigParser para ler o arquivo 'config.ini'
config = ConfigParser()
config.read('config.ini')

# Obtém o caminho do preditor facial no arquivo de configuração


def get_predictor_path():
    return config.get('settings', 'PREDICTOR_PATH')

# Obtém o caminho do arquivo de som no arquivo de configuração


def get_sound_file_path():
    return config.get('settings', 'SOUND_FILE_PATH')

# Obtém o valor do limiar EAR (Eye Aspect Ratio) no arquivo de configuração


def get_ear_threshold():
    return config.getfloat('settings', 'EAR_THRESHOLD')

# Obtém o valor do tempo de alerta no arquivo de configuração


def get_tempo_alerta():
    return config.getfloat('settings', 'TEMPO_ALERTA')
