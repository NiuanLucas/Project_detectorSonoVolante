from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


def get_predictor_path():
    return config.get('settings', 'PREDICTOR_PATH')


def get_sound_file_path():
    return config.get('settings', 'SOUND_FILE_PATH')


def get_ear_threshold():
    return config.getfloat('settings', 'EAR_THRESHOLD')


def get_tempo_alerta():
    return config.getfloat('settings', 'TEMPO_ALERTA')
