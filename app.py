# ============================================================
# ACCELERATION DETECTION
# ============================================================

def classify_acceleration(
    gestational_age,
    amplitude_bpm,
    duration_sec,
    onset_to_peak_sec
):
    """
    NICHD/ACOG acceleration classification
    """

    # 10 dakika veya üzeri = baseline değişikliği
    if duration_sec >= 600:
        return "Baseline değişikliği", False

    # 2–10 dk = prolonged acceleration
    if 120 <= duration_sec < 600:
        return "Uzamış akselerasyon", True

    # Abrupt olması gerekir
    if onset_to_peak_sec >= 30:
        return "Akselerasyon kriterlerini karşılamıyor", False

    if gestational_age >= 32:
        if amplitude_bpm >= 15 and 15 <= duration_sec < 120:
            return "15×15 akselerasyon", True

    else:
        if amplitude_bpm >= 10 and 10 <= duration_sec < 120:
            return "10×10 akselerasyon", True

    return "Akselerasyon kriterlerini karşılamıyor", False


# ============================================================
# DECELERATION DETECTION
# ============================================================

def classify_deceleration(
    amplitude_drop,
    duration_sec,
    onset_to_nadir_sec,
    contraction_present=False,
    onset_relation=None,
    nadir_relation=None,
    recovery_relation=None
):
    """
    NICHD/ACOG deceleration classification.

    onset_relation / nadir_relation / recovery_relation:
    Early veya late ayrımı için kontraksiyon ilişkisi.
    """

    # --------------------------------------------------------
    # 10 dakika ve üzeri = baseline değişikliği
    # --------------------------------------------------------

    if duration_sec >= 600:
        return "Bazal FHR değişikliği"

    # --------------------------------------------------------
    # PROLONGED DECELERATION
    # ≥15 bpm, 2–10 dakika
    # --------------------------------------------------------

    if amplitude_drop >= 15 and 120 <= duration_sec < 600:
        return "Uzamış deselerasyon"

    # --------------------------------------------------------
    # VARIABLE DECELERATION
    # Abrupt: onset → nadir <30 sn
    # ≥15 bpm
    # ≥15 sn ve <2 dk
    # --------------------------------------------------------

    if (
        amplitude_drop >= 15
        and 15 <= duration_sec < 120
        and onset_to_nadir_sec < 30
    ):
        return "Değişken deselerasyon"

    # --------------------------------------------------------
    # GRADUAL DECELERATION
    # onset → nadir ≥30 sn
    # Early / Late ayrımı kontraksiyona göre
    # --------------------------------------------------------

    if contraction_present and onset_to_nadir_sec >= 30:

        # EARLY
        if (
            onset_relation == "Kontraksiyonla birlikte"
            and nadir_relation == "Kontraksiyon tepe noktasında"
            and recovery_relation == "Kontraksiyonla birlikte"
        ):
            return "Erken deselerasyon"

        # LATE
        if (
            onset_relation == "Kontraksiyondan sonra"
            and nadir_relation == "Kontraksiyon tepe noktasından sonra"
            and recovery_relation == "Kontraksiyon bittikten sonra"
        ):
            return "Geç deselerasyon"

        return "Gradual deselerasyon – early/late için zaman ilişkisini kontrol et"

    return "Tanımlanamayan / kriterleri karşılamayan deselerasyon"
