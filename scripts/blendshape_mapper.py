#blendshape_mapper.py
def clamp(x):
    return max(0.0, min(1.0, float(x)))


def map_to_gltf_blendshapes(norm_feat):
    """
    norm_feat: normalized features in [0,1]
    Output keys MUST match glTF morph target names
    """

    mouth_open = norm_feat["mouth_open"]
    mouth_width = norm_feat["mouth_width"]
    lip_rounding = norm_feat["lip_rounding"]
    upper_lip = norm_feat["upper_lip_raise"]

    # Derived behaviors
    lower_lip = mouth_open * 0.8
    frown = max(0.0, (1.0 - mouth_width) * 0.5)

    return {
        "open": clamp(mouth_open),
        "wide": clamp(mouth_width),
        "narrow": clamp(lip_rounding), #"narrow": clamp(lip_rounding * (1.0 - mouth_open))

        "upper_up": clamp(upper_lip),
        "lower_down": clamp(lower_lip),
        "frown": clamp(frown),

        # Explicitly zero unused ones
        "upper_down": 0.0,
        "lower_up": 0.0,
        "wink": 0.0
    }
