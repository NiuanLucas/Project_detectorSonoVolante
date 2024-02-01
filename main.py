# Descrição Geral:
# Este é um programa de detecção de sonolência ao volante que utiliza três técnicas diferentes:
# 1. EAR (Eye Aspect Ratio): Mede a abertura dos olhos para identificar sonolência.
# 2. PERCLOS (Percentage of Eyelid Closure): Calcula a porcentagem de tempo com os olhos fechados.
# 3. Análise de Frequência e Duração das Piscadas: Avalia a frequência e duração das piscadas dos olhos.

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
        # Chama a detecção de sonolência usando EAR
        ear_detector.detectar_sonolencia_ear()
    elif escolha == '2':
        # Chama a detecção de sonolência usando PERCLOS
        perclos_detector.detectar_sonolencia_perclos()
    elif escolha == '3':
        # Chama a detecção de sonolência usando análise de piscadas
        piscadas_detector.detectar_sonolencia_piscadas()
    else:
        print("Escolha inválida. Por favor, selecione 1, 2, ou 3.")


if __name__ == "__main__":
    main()
