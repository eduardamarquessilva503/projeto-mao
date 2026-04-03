import cv2
import pickle
import time

from detector import HandDetector
from utils import normalize_landmarks


MODEL_PATH = "modelo.pkl"


frase = ""
ultima_predicao = ""
contador_estabilidade = 0
ultima_letra_adicionada = ""
ultimo_tempo_adicao = 0


def click(event, x, y, flags, param):
    global frase
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 < x < 630 and 70 < y < 110:
            frase = ""


def main():
    global frase, ultima_predicao, contador_estabilidade
    global ultima_letra_adicionada, ultimo_tempo_adicao

    try:
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
    except FileNotFoundError:
        print("Erro: modelo.pkl não encontrado. Execute o treinar.py primeiro.")
        return

    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", click)

    while True:
        ret, img = cap.read()
        if not ret:
            print("Erro ao capturar frame da câmera.")
            break

        img, hands = detector.find_hands(img)

        letra_atual = "-"

        if hands:
            hand = hands[0]
            dados = normalize_landmarks(hand)

            if dados:
                previsao = modelo.predict([dados])
                letra_atual = previsao[0]

                if letra_atual == ultima_predicao:
                    contador_estabilidade += 1
                else:
                    contador_estabilidade = 0

                ultima_predicao = letra_atual

                tempo_agora = time.time()

                if (
                    contador_estabilidade >= 8 and
                    letra_atual != ultima_letra_adicionada and
                    (tempo_agora - ultimo_tempo_adicao) > 0.8
                ):
                    frase += letra_atual
                    ultima_letra_adicionada = letra_atual
                    ultimo_tempo_adicao = tempo_agora
                    contador_estabilidade = 0

                cv2.putText(img, f"Letra: {letra_atual}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(img, (0, 0), (640, 60), (50, 50, 50), -1)
        cv2.putText(img, f"FRASE: {frase}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(img, (500, 70), (630, 110), (0, 0, 255), -1)
        cv2.putText(img, "LIMPAR", (505, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Camera", img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(" "):
            frase += " "
        elif key == ord("b") and len(frase) > 0:
            frase = frase[:-1]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()