# Importando os módulos dos detectores
import ear_detector
import perclos_detector
import piscadas_detector


def main():
    print("Selecione a técnica de detecção de sonolência:")
    print("1: EAR (Eye Aspect Ratio)")
    print("2: PERCLOS (Percentage of Eyelid Closure)")
    print("3: Análise de Frequência e Duração das Piscadas")
    escolha = input("Digite o número da técnica escolhida (1, 2, ou 3): ")

    if escolha == '1':
        ear_detector.detectar_sonolencia_ear()
    elif escolha == '2':
        perclos_detector.detectar_sonolencia_perclos()
    elif escolha == '3':
        piscadas_detector.detectar_sonolencia_piscadas()
    else:
        print("Escolha inválida. Por favor, selecione 1, 2, ou 3.")


if __name__ == "__main__":
    main()
