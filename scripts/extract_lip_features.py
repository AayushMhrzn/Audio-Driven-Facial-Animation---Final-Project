import numpy as np

# ===============================
# Landmark indices 
# ===============================
UPPER_MID = 5
LOWER_MID = 15
LEFT_CORNER = 0
RIGHT_CORNER = 10

UPPER_LEFT = [0, 1, 2, 3, 4]
UPPER_RIGHT = [6, 7, 8, 9, 10]
LOWER_LEFT = [0, 19, 18, 17, 16]
LOWER_RIGHT = [14, 13, 12, 11, 10]

# ===============================
# Stability constants
# ===============================
EPS = 1e-6
LIP_ROUNDING_MAX = 3.0   # 👈 Option A clamp (IMPORTANT)


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def extract_features(lips):
    """
    lips: (20, 2) centered & normalized landmarks
    returns: dict of geometric mouth features
    """

    upper_mid = lips[UPPER_MID]
    lower_mid = lips[LOWER_MID]

    left_corner = lips[LEFT_CORNER]
    right_corner = lips[RIGHT_CORNER]

    # -------------------------------------------------
    # 1️⃣ Mouth opening (vertical distance)
    # -------------------------------------------------
    mouth_open = euclidean(upper_mid, lower_mid)

    # -------------------------------------------------
    # 2️⃣ Mouth width (horizontal distance)
    # -------------------------------------------------
    mouth_width = euclidean(left_corner, right_corner)

    # -------------------------------------------------
    # 3️⃣ Upper lip raise
    # (average Y displacement of upper lip curve)
    # -------------------------------------------------
    upper_lip_raise = np.mean(
        lips[UPPER_LEFT + UPPER_RIGHT][:, 1]
    )

    # -------------------------------------------------
    # 4️⃣ Lip rounding
    # (ratio of width to opening, CLAMPED)
    # -------------------------------------------------
    lip_rounding = mouth_width / (mouth_open + EPS)
    lip_rounding = min(lip_rounding, LIP_ROUNDING_MAX)  # 👈 FIX

    return {
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "upper_lip_raise": upper_lip_raise,
        "lip_rounding": lip_rounding
    }
