import cv2
import csv
import os

from detector import HandDetector
from utils import normalize_landmarks


CSV_PATH = "dados.csv"


def criar_cabecalho_se_necessario():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["letra"] + [f"p{i}" for i in range(42)]
            writer.writerow(header)


def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    letra = input("Digite a letra/classe que deseja coletar: ").strip().upper()
    if not letra:
        print("Erro: letra inválida.")
        cap.release()
        return

    criar_cabecalho_se_necessario()

    print("Coleta iniciada.")
    print("Pressione 'c' para capturar uma amostra.")
    print("Pressione ESC para sair.")

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        while True:
            ret, img = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break

            img, hands = detector.find_hands(img)

            cv2.putText(img, f"Classe: {letra}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, "Pressione C para salvar", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF

            if hands:
                hand = hands[0]
                dados = normalize_landmarks(hand)

                if key == ord("c"):
                    writer.writerow([letra] + dados)
                    print(f"Amostra salva para a classe {letra}")

            cv2.imshow("Coleta de Dados", img)

            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()