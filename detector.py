import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            pontos = []

            for lm in hand_lms.landmark:
                pontos.append([lm.x, lm.y])

            return pontos

        return None

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hands = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS
                )

                pontos = []
                for lm in hand_lms.landmark:
                    pontos.append([lm.x, lm.y])

                hands.append(pontos)

        return img, hands