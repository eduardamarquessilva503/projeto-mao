def normalize_landmarks(hand):
    """
    Normaliza os 21 landmarks da mão:
    - desloca para a origem usando o primeiro ponto
    - escala pelo maior valor absoluto
    Retorna uma lista achatada: [x1, y1, x2, y2, ...]
    """
    if not hand or len(hand) == 0:
        return []

    base_x, base_y = hand[0]
    normalized = []

    for x, y in hand:
        normalized.append(x - base_x)
        normalized.append(y - base_y)

    max_value = max(abs(v) for v in normalized) if normalized else 1.0
    if max_value == 0:
        max_value = 1.0

    normalized = [v / max_value for v in normalized]
    return normalized