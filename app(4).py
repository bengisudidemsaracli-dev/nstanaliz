import os
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab


# ============================================================
# UYGULAMA AYARLARI
# ============================================================

st.set_page_config(
    page_title="NST / CTG Analyzer",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 NST / CTG Analyzer")
st.caption("Bilkent/hastane dijital NST çıktısı için deneysel klinik karar destek prototipi")

st.warning(
    "⚠️ Bu uygulama deneysel bir prototiptir. Görüntüden elde edilen otomatik ölçümler "
    "ve yönetim önerileri klinisyenin traseyi doğrudan değerlendirmesinin yerine geçmez. "
    "Kalibrasyon veya çizgi yakalama güvenilir değilse sonuç kullanılmamalıdır."
)

# Hastane çıktısı için preset:
# 2 küçük kare = 1 dakika  -> 1 küçük kare = 30 saniye
SECONDS_PER_SMALL_SQUARE = 30.0

# FHR panelinde ana yatay grid aralığı
BPM_PER_MAJOR_GRID = 20.0

# Verilen hastane export örneğinin referans geometrisi
REFERENCE_W = 572.0
REFERENCE_H = 559.0
REFERENCE_SMALL_SQUARE_PX = 15.0
REFERENCE_MAJOR_GRID_PX = 37.0
REFERENCE_Y160 = 132.0

# Panel / sinyal sınırları (örnek export oranlarından)
FHR_TOP_RATIO = 0.10
FHR_BOTTOM_RATIO = 0.57
TRACE_X_LEFT_RATIO = 0.085
TRACE_X_RIGHT_RATIO = 0.975


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file).convert("RGB"))


def get_hospital_calibration(image):
    h, w, _ = image.shape

    small_square_px = REFERENCE_SMALL_SQUARE_PX * (w / REFERENCE_W)
    major_grid_px = REFERENCE_MAJOR_GRID_PX * (h / REFERENCE_H)
    y160_global = REFERENCE_Y160 * (h / REFERENCE_H)

    return {
        "small_square_px": small_square_px,
        "seconds_per_pixel": SECONDS_PER_SMALL_SQUARE / small_square_px,
        "major_grid_px": major_grid_px,
        "bpm_per_pixel": BPM_PER_MAJOR_GRID / major_grid_px,
        "y160_global": y160_global,
    }


def crop_fhr_panel(image):
    h, w, _ = image.shape

    y1 = int(h * FHR_TOP_RATIO)
    y2 = int(h * FHR_BOTTOM_RATIO)
    x1 = int(w * TRACE_X_LEFT_RATIO)
    x2 = int(w * TRACE_X_RIGHT_RATIO)

    return image[y1:y2, x1:x2], y1, x1


def purple_trace_mask(rgb):
    """
    Hastane exportundaki mor/magenta FHR çizgisini gri gridden ayırır.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Mor/magenta ana maske
    mask1 = (
        (h >= 125) &
        (h <= 179) &
        (s >= 35) &
        (v >= 45)
    )

    # Bazı exportlarda mor çizgi daha soluk olabilir.
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    mask2 = (
        (r - g >= 10) &
        (b - g >= 3) &
        (s >= 20)
    )

    mask = (mask1 | mask2).astype(np.uint8) * 255

    # Küçük kopuklukları birleştir.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def extract_trace_from_mask(mask):
    """
    Her x kolonunda mor trace'e ait y değerini seçer.
    Süreklilik kullanılarak yazı/legend gibi gürültü azaltılır.
    """
    h, w = mask.shape
    trace = np.full(w, np.nan)

    previous_y = None

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        # Aynı kolondaki bitişik pikselleri gruplandır.
        groups = []
        current = [ys[0]]

        for y in ys[1:]:
            if y - current[-1] <= 2:
                current.append(y)
            else:
                groups.append(current)
                current = [y]

        groups.append(current)
        centers = np.array([np.mean(g) for g in groups])

        if previous_y is None:
            # FHR'nin fizyolojik olarak daha olası olduğu orta-alt kısmı tercih et.
            target = h * 0.40
            selected = centers[np.argmin(np.abs(centers - target))]
        else:
            selected = centers[np.argmin(np.abs(centers - previous_y))]

            # Çok büyük ani piksel sıçramalarını çizgi olarak kabul etme.
            if abs(selected - previous_y) > max(25, h * 0.12):
                continue

        trace[x] = selected
        previous_y = selected

    valid = ~np.isnan(trace)
    quality = float(np.mean(valid))

    if np.sum(valid) >= 10:
        trace = np.interp(
            np.arange(w),
            np.where(valid)[0],
            trace[valid]
        )

    return trace, quality


def fallback_dark_trace(fhr_crop):
    """
    Mor segmentasyon başarısız olursa ikinci yöntem.
    Gri grid ile karışabileceği için yalnızca fallback olarak kullanılır.
    """
    gray = cv2.cvtColor(fhr_crop, cv2.COLOR_RGB2GRAY)
    dark = gray < 105

    h, w = dark.shape
    trace = np.full(w, np.nan)
    previous_y = None

    y_min = int(h * 0.10)
    y_max = int(h * 0.88)

    for x in range(w):
        ys = np.where(dark[y_min:y_max, x])[0] + y_min

        if len(ys) == 0:
            continue

        if previous_y is None:
            target = h * 0.40
            selected = ys[np.argmin(np.abs(ys - target))]
        else:
            selected = ys[np.argmin(np.abs(ys - previous_y))]
            if abs(selected - previous_y) > 12:
                continue

        trace[x] = selected
        previous_y = selected

    valid = ~np.isnan(trace)
    quality = float(np.mean(valid))

    if np.sum(valid) >= 10:
        trace = np.interp(
            np.arange(w),
            np.where(valid)[0],
            trace[valid]
        )

    return trace, quality


def convert_trace_to_bpm(trace_y, fhr_top_global, calibration, manual_offset_bpm=0.0):
    global_y = trace_y + fhr_top_global

    bpm = (
        160.0
        + (calibration["y160_global"] - global_y) * calibration["bpm_per_pixel"]
        + manual_offset_bpm
    )

    return bpm


def robust_smooth(signal, window=3):
    if window <= 1:
        return signal.copy()

    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def calculate_baseline(fhr):
    """
    Baseline için kaba prototip yaklaşımı:
    önce fizyolojik aralıktaki medyanı alır, sonra ±25 bpm dışındaki
    belirgin excursionları çıkarıp tekrar medyan hesaplar.
    """
    valid = fhr[np.isfinite(fhr) & (fhr >= 70) & (fhr <= 210)]

    if len(valid) < 20:
        return None

    first = np.median(valid)
    trimmed = valid[np.abs(valid - first) <= 25]

    if len(trimmed) < 20:
        trimmed = valid

    return float(np.median(trimmed))


def calculate_variability(fhr, baseline):
    """
    Raster görüntüden yaklaşık variability tahmini.
    Klinik görsel değerlendirmeyi ikame etmez.
    """
    valid = fhr[
        np.isfinite(fhr) &
        (np.abs(fhr - baseline) <= 15)
    ]

    if len(valid) < 30:
        return None, "Değerlendirilemedi"

    amplitude = float(np.percentile(valid, 95) - np.percentile(valid, 5))

    if amplitude <= 2:
        label = "Absent"
    elif amplitude <= 5:
        label = "Minimal"
    elif amplitude <= 25:
        label = "Moderate"
    else:
        label = "Marked"

    return amplitude, label


def find_regions(mask):
    regions = []
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(mask) - 1))

    return regions


def detect_accelerations_morphologic(
    fhr,
    baseline,
    seconds_per_pixel,
    amplitude_threshold=15.0,
    duration_threshold=15.0,
):
    """
    Akselerasyonu yalnızca baseline+eşik üzerinde geçirilen süre ile değil,
    onset -> peak -> baseline'a dönüş morfolojisiyle yaklaşık olarak bulur.

    Bu hâlâ raster görüntü için deneysel bir algoritmadır; klinik trase
    yorumunun yerine geçmez.
    """
    signal = np.asarray(fhr, dtype=float)
    n = len(signal)
    if n < 5 or not np.isfinite(baseline):
        return []

    # Onset/return için bazale yakın kabul edilen tolerans.
    return_tolerance = 5.0
    above = signal >= baseline + return_tolerance
    candidate_regions = find_regions(above)
    events = []

    # Yakın iki parçayı, aradaki kısa bazale yaklaşma nedeniyle bölünmüşse birleştir.
    max_gap_px = max(1, int(round(5.0 / seconds_per_pixel)))
    merged = []
    for region in candidate_regions:
        if not merged:
            merged.append(list(region))
        elif region[0] - merged[-1][1] - 1 <= max_gap_px:
            merged[-1][1] = region[1]
        else:
            merged.append(list(region))

    for core_start, core_end in merged:
        # Peak gerçekten gerekli amplitüde ulaşmalı.
        core = signal[core_start:core_end + 1]
        peak_rel = int(np.nanargmax(core))
        peak_idx = core_start + peak_rel
        peak = float(signal[peak_idx])
        amplitude = peak - baseline

        if amplitude < amplitude_threshold:
            continue

        # Onset: peak öncesinde bazale (±5 bpm) son temas.
        onset = core_start
        i = peak_idx
        while i > 0:
            if signal[i] <= baseline + return_tolerance:
                onset = i
                break
            i -= 1

        # Return: peak sonrasında bazale (±5 bpm) ilk dönüş.
        end = core_end
        i = peak_idx
        while i < n - 1:
            if signal[i] <= baseline + return_tolerance:
                end = i
                break
            i += 1

        duration = (end - onset + 1) * seconds_per_pixel

        # NICHD terminolojisinde akselerasyon <2 dk olmalıdır;
        # 2-10 dk prolonged acceleration olarak değerlendirilir.
        if duration < duration_threshold or duration >= 120.0:
            continue

        events.append({
            "start": int(onset),
            "peak_index": int(peak_idx),
            "end": int(end),
            "duration": float(duration),
            "amplitude": float(amplitude),
            "extreme": peak,
        })

    # Aynı peak çevresinde oluşan örtüşen eventleri tekilleştir.
    deduped = []
    for event in sorted(events, key=lambda x: x["start"]):
        if not deduped or event["start"] > deduped[-1]["end"]:
            deduped.append(event)
        elif event["amplitude"] > deduped[-1]["amplitude"]:
            deduped[-1] = event

    return deduped


def detect_excursions(
    fhr,
    baseline,
    seconds_per_pixel,
    direction="down",
    amplitude_threshold=15.0,
    duration_threshold=15.0,
):
    """Deselerasyon için kaba eşik taraması; morfolojik tip klinisyen doğrulamasındadır."""
    if direction == "up":
        mask = fhr >= baseline + amplitude_threshold
    else:
        mask = fhr <= baseline - amplitude_threshold

    regions = find_regions(mask)
    events = []

    for start, end in regions:
        duration = (end - start + 1) * seconds_per_pixel
        if duration < duration_threshold:
            continue

        segment = fhr[start:end + 1]
        if direction == "up":
            extreme = float(np.max(segment))
            amplitude = float(extreme - baseline)
        else:
            extreme = float(np.min(segment))
            amplitude = float(baseline - extreme)

        events.append({
            "start": start,
            "end": end,
            "duration": float(duration),
            "amplitude": amplitude,
            "extreme": extreme,
        })

    return events

def nst_assessment(accelerations, gestational_age, total_minutes):
    if gestational_age >= 32:
        amp_thr, dur_thr = 15.0, 15.0
        rule_name = "15×15"
    else:
        amp_thr, dur_thr = 10.0, 10.0
        rule_name = "10×10"

    valid = [
        a for a in accelerations
        if a["amplitude"] >= amp_thr and a["duration"] >= dur_thr
    ]

    if len(valid) >= 2:
        return (
            "Reaktivite kriteri karşılandı",
            valid,
            f"Bu görüntü kesitinde ≥2 adet {rule_name} akselerasyon saptandı."
        )

    if total_minutes >= 20:
        return (
            "Reaktivite kriteri karşılanmadı",
            valid,
            f"Yaklaşık {total_minutes:.1f} dakikalık görüntüde ≥2 adet {rule_name} akselerasyon saptanmadı."
        )

    return (
        "Süre yetersiz",
        valid,
        f"Görüntü yaklaşık {total_minutes:.1f} dakika. Reaktivite yokluğu için tam kayıt süresi değerlendirilmelidir."
    )


def classify_ctg(
    baseline,
    variability,
    recurrent_late,
    recurrent_variable,
    sinusoidal
):
    # Category III
    if sinusoidal:
        return "III"

    if variability == "Absent":
        if recurrent_late or recurrent_variable or baseline < 110:
            return "III"

    # Category I
    if (
        110 <= baseline <= 160
        and variability == "Moderate"
        and not recurrent_late
        and not recurrent_variable
        and not sinusoidal
    ):
        return "I"

    # Category II = I ve III dışında kalanlar
    return "II"


def pattern_summary(
    baseline,
    variability,
    early,
    recurrent_late,
    recurrent_variable,
    prolonged,
    sinusoidal
):
    parts = []

    if baseline < 110:
        parts.append("bradikardi")
    elif baseline > 160:
        parts.append("taşikardi")
    else:
        parts.append("normal bazal hız")

    parts.append(f"{variability.lower()} variabilite")

    if early:
        parts.append("early deselerasyon")
    if recurrent_late:
        parts.append("rekürren late deselerasyon")
    if recurrent_variable:
        parts.append("rekürren variable deselerasyon")
    if prolonged:
        parts.append("prolonged deselerasyon")
    if sinusoidal:
        parts.append("sinüzoidal patern")

    return ", ".join(parts)


def acog_pattern_recommendations(
    category,
    baseline,
    variability,
    early,
    recurrent_late,
    recurrent_variable,
    prolonged,
    sinusoidal,
    tachysystole,
    oxytocin,
    propess,
    cytotec,
    hypotension,
    maternal_hypoxia,
):
    recs = []

    if category == "I":
        recs.append("Category I: rutin intrapartum FHR izlemi uygundur.")
        recs.append("Mevcut FHR paterni nedeniyle spesifik intrauterin resüsitatif girişim gerekmez.")
        if early:
            recs.append("Early deselerasyonlar tek başına fetal hipoksemi/asidemi göstergesi değildir; klinik bağlamda rutin izlem sürdürülür.")
        return recs

    recs.append("Maternal vital bulguları, uterin aktiviteyi, indüksiyon/augmentasyon ajanlarını ve FHR değişiminin zamanlamasını birlikte yeniden değerlendir.")

    if baseline > 160:
        recs.append("Fetal taşikardi: maternal ateş/enfeksiyon, dehidratasyon, ilaç etkileri ve diğer maternal-fetal nedenleri değerlendir.")

    if baseline < 110:
        recs.append("Bradikardi: maternal hipotansiyon, umbilikal kord prolapsusu, ablasyo, uterin rüptür ve diğer akut nedenleri hızla değerlendir.")
        recs.append("Uygun maternal pozisyon değişikliğini değerlendir.")

    if variability == "Minimal":
        recs.append("Minimal variabilite: uyku siklusu, ilaç etkileri ve hipoksemi/asidemi olasılığını klinik bağlamla birlikte değerlendir; seri trase değerlendirmesi yap.")

    if variability == "Absent":
        recs.append("Absent variabiliteyi eşlik eden bazal hız ve deselerasyonlarla birlikte acil olarak yeniden değerlendir.")

    if variability == "Marked":
        recs.append("Marked variabilite Category II kapsamındadır; devamlılığını ve eşlik eden FHR özelliklerini seri olarak izle.")

    if recurrent_variable:
        recs.append("Rekürren variable deselerasyonlarda umbilikal kord kompresyonunu değerlendir ve maternal pozisyon değişikliğini düşün.")
        recs.append("Persistan rekürren variable deselerasyonlarda uygun klinik durumda amnioinfüzyon düşünülebilir.")

    if recurrent_late:
        recs.append("Rekürren late deselerasyonlarda uteroplasental perfüzyonu etkileyen nedenleri değerlendir; maternal hipotansiyon varsa düzelt ve pozisyon değişikliğini düşün.")

    if prolonged:
        recs.append("Prolonged deselerasyonda akut maternal/obstetrik nedenleri hızla değerlendir; kord prolapsusu, ablasyo, uterin rüptür ve hipotansiyon gibi geri döndürülebilir nedenleri dışla/düzelt.")
        recs.append("Patern düzelmiyorsa doğum için hazırlık yap ve doğum zamanlamasını maternal-fetal duruma göre değerlendir.")

    if tachysystole:
        recs.append("Taşisistoli mevcutsa uterin aktiviteyi azaltmaya yönelik girişimleri değerlendir.")
        if oxytocin:
            recs.append("Oksitosin kullanılıyorsa azaltılması veya durdurulmasını değerlendir.")
        if cytotec:
            recs.append("Misoprostol sonrası taşisistoli/FHR anomalisi varsa ek doz öncesi yeniden klinik değerlendirme yap ve kurum indüksiyon protokolünü uygula.")
        if propess:
            recs.append("Dinoprostone vajinal insert ile taşisistoli veya olumsuz FHR paterni varsa insertin devamı/çıkarılması açısından ürün ve kurum protokolüne göre acil yeniden değerlendirme yap.")

    if category in ("II", "III") and (oxytocin or propess or cytotec):
        recs.append("İndüksiyon/augmentasyon ajanlarının uterin aktivite ve FHR değişikliği ile zaman ilişkisini değerlendir.")

    if hypotension:
        recs.append("Maternal hipotansiyonu düzelt ve uteroplasental perfüzyonu yeniden değerlendir.")

    if maternal_hypoxia:
        recs.append("Maternal hipoksemi mevcutsa maternal oksijenasyonu düzelt.")
    else:
        recs.append("Maternal oksijen satürasyonu normalse yalnızca Category II/III FHR paterni nedeniyle rutin maternal oksijen verme.")

    if sinusoidal:
        recs.append("Sinüzoidal patern Category III'tür; fetal anemi ve diğer ciddi etiyolojiler açısından acil değerlendirme gerekir.")

    if category == "III":
        recs.append("Category III: hızlı bedside değerlendirme ve intrauterin resüsitatif girişimleri başlat.")
        if oxytocin:
            recs.append("Oksitosini durdur.")
        if tachysystole:
            recs.append("Eşlik eden taşisistoliyi hızla tedavi et.")
        recs.append("Başlangıç girişimlerine rağmen Category III patern düzelmiyorsa doğumu hızlandır; yöntem ve zamanlamayı maternal-fetal durum ve uygulanabilirliğe göre belirle.")

    return recs


def make_plot(fhr, baseline, valid_accels, decels, seconds_per_pixel):
    time_min = np.arange(len(fhr)) * seconds_per_pixel / 60.0

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(time_min, fhr, linewidth=1.1, label="Dijitalize FHR")
    ax.axhline(baseline, linestyle="--", label=f"Baseline {baseline:.0f} bpm")

    for event in valid_accels:
        ax.axvspan(
            event["start"] * seconds_per_pixel / 60,
            event["end"] * seconds_per_pixel / 60,
            alpha=0.12
        )
        if "peak_index" in event:
            peak_t = event["peak_index"] * seconds_per_pixel / 60
            ax.scatter([peak_t], [event["extreme"]], s=28, zorder=5)

    for event in decels:
        ax.axvspan(
            event["start"] * seconds_per_pixel / 60,
            event["end"] * seconds_per_pixel / 60,
            alpha=0.12
        )

    ax.set_xlabel("Zaman (dk)")
    ax.set_ylabel("FHR (bpm)")
    ax.set_ylim(80, 200)
    ax.legend()
    ax.grid(alpha=0.2)

    return fig


# ============================================================
# PDF
# ============================================================

def register_pdf_font():
    """
    PDF icin Turkce karakter destekleyen fontu uygulamanin calistigi
    ortamdan bulur. Streamlit Cloud'da ReportLab ile birlikte gelen
    Bitstream Vera fontlarini kullanir; boylece Arial/DejaVu kurulu
    olmasa bile ğ, ü, ş, ı, ö, ç, İ karakterleri bozulmaz.
    """
    reportlab_fonts = os.path.join(os.path.dirname(reportlab.__file__), "fonts")

    regular_candidates = [
        os.path.join(reportlab_fonts, "Vera.ttf"),
        r"C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        os.path.join(reportlab_fonts, "VeraBd.ttf"),
        r"C:\\Windows\\Fonts\\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    normal = None
    bold = None

    for path in regular_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("NSTFont", path))
                normal = "NSTFont"
                break
            except Exception:
                pass

    for path in bold_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("NSTFontBold", path))
                bold = "NSTFontBold"
                break
            except Exception:
                pass

    if normal is None:
        raise RuntimeError("Turkce karakter destekleyen PDF fontu bulunamadi.")
    if bold is None:
        bold = normal

    return normal, bold


def create_pdf(
    patient_name,
    patient_age,
    gestational_age,
    gravida,
    parity,
    living,
    propess,
    propess_time,
    cytotec,
    cytotec_dose,
    cytotec_time,
    oxytocin,
    tachysystole,
    hypotension,
    maternal_hypoxia,
    baseline,
    variability,
    variability_amp,
    accel_count,
    decel_count,
    nst_status,
    category,
    pattern_text,
    recommendations,
    quality,
    total_minutes,
):
    buffer = BytesIO()
    normal_font, bold_font = register_pdf_font()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=38,
        bottomMargin=38,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NSTTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "NSTHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "NSTBody",
        parent=styles["BodyText"],
        fontName=normal_font,
        fontSize=9,
        leading=12,
    )

    story = [
        Paragraph("NST / CTG DEĞERLENDİRME RAPORU", title_style),
        Paragraph(datetime.now().strftime("%d.%m.%Y %H:%M"), body_style),
        Spacer(1, 8),
    ]

    story.append(Paragraph("Hasta Bilgileri", heading_style))
    patient_rows = [
        ["Hasta", patient_name if patient_name.strip() else "-"],
        ["Yaş", str(patient_age)],
        ["Gebelik haftası", f"{gestational_age:.1f}"],
        ["Gravida / Parite / Yaşayan", f"G{gravida} / P{parity} / Y{living}"],
    ]

    t = Table(patient_rows, colWidths=[160, 290])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), normal_font),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    story.append(Paragraph("İndüksiyon / Maternal-Uterin Durum", heading_style))
    induction_rows = [
        ["ProPess", f"Evet - {propess_time}" if propess else "Hayır"],
        ["Cytotec", f"Evet - {cytotec_dose} mcg - {cytotec_time}" if cytotec else "Hayır"],
        ["Oksitosin", "Evet" if oxytocin else "Hayır"],
        ["Taşisistoli", "Var" if tachysystole else "Yok"],
        ["Maternal hipotansiyon", "Var" if hypotension else "Yok"],
        ["Maternal hipoksemi", "Var" if maternal_hypoxia else "Yok"],
    ]

    t = Table(induction_rows, colWidths=[180, 270])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), normal_font),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    story.append(Paragraph("FHR / NST Analizi", heading_style))
    analysis_rows = [
        ["Bazal FHR", f"{baseline:.0f} bpm"],
        ["Variabilite", variability],
        ["Yaklaşık variabilite amplitüdü", f"{variability_amp:.1f} bpm" if variability_amp is not None else "Değerlendirilemedi"],
        ["Uygun akselerasyon", str(accel_count)],
        ["≥15 bpm / ≥15 sn kaba deselerasyon", str(decel_count)],
        ["NST", nst_status],
        ["FHR kategorisi", f"CATEGORY {category}"],
        ["Patern", pattern_text],
    ]

    t = Table(analysis_rows, colWidths=[180, 270])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), normal_font),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    story.append(Paragraph("Paterne Özgü Yaklaşım", heading_style))
    for rec in recommendations:
        story.append(Paragraph("- " + rec, body_style))
        story.append(Spacer(1, 3))

    story.append(Paragraph("Teknik Bilgiler", heading_style))
    story.append(Paragraph(f"FHR çizgi yakalama oranı: %{quality * 100:.1f}", body_style))
    story.append(Paragraph(f"Görüntülenen yaklaşık kayıt süresi: {total_minutes:.1f} dk", body_style))
    story.append(Paragraph("Hastane preset'i: 2 küçük kare = 1 dakika; ana FHR yatay grid aralığı = 20 bpm.", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "UYARI: Bu rapor deneysel bir görüntü analizi prototipi tarafından üretilmiştir. "
        "Raster görüntüden bazal hız/variabilite/event ölçümü hata içerebilir. "
        "Klinik karar, orijinal trase ve hastanın doğrudan klinik değerlendirmesine dayanmalıdır.",
        body_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# ARAYÜZ
# ============================================================

st.header("1️⃣ Hasta Bilgileri")

c1, c2, c3 = st.columns(3)

with c1:
    patient_name = st.text_input("Hasta adı / protokol (opsiyonel)")
    patient_age = st.number_input("Hasta yaşı", 10, 60, 30)

with c2:
    gestational_age = st.number_input(
        "Gestasyonel hafta",
        min_value=20.0,
        max_value=42.0,
        value=39.0,
        step=0.1,
    )
    gravida = st.number_input("Gravida", 1, 20, 1)

with c3:
    parity = st.number_input("Parite", 0, 20, 0)
    living = st.number_input("Yaşayan", 0, 20, 0)


st.header("2️⃣ NST / CTG Görüntüsü")

uploaded_file = st.file_uploader(
    "Hastane NST ekran görüntüsünü yükleyin",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Yüklenen NST / CTG", use_container_width=True)

    fhr_crop, fhr_top, fhr_left = crop_fhr_panel(image)

    with st.expander("Analiz edilen FHR alanını göster"):
        st.image(fhr_crop, use_container_width=True)


    st.header("3️⃣ FHR Patern Doğrulaması")

    st.info(
        "Bazal FHR, yaklaşık variabilite ve excursionlar görüntüden hesaplanır. "
        "Late/variable/early/prolonged ve sinüzoidal patern bu sürümde klinisyen tarafından doğrulanır. "
        "Bu seçimler Category I/II/III hesabına girer."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        early = st.checkbox("Early deselerasyon")
        recurrent_late = st.checkbox("Rekürren late deselerasyon")

    with c2:
        recurrent_variable = st.checkbox("Rekürren variable deselerasyon")
        prolonged = st.checkbox("Prolonged deselerasyon")

    with c3:
        sinusoidal = st.checkbox("Sinüzoidal patern")


    st.header("4️⃣ Maternal / Uterin Durum")

    c1, c2, c3 = st.columns(3)

    with c1:
        tachysystole = st.checkbox("Taşisistoli")
        oxytocin = st.checkbox("Oksitosin kullanılıyor")

    with c2:
        propess = st.checkbox("ProPess (dinoprostone) uygulanmış")
        propess_time = st.text_input("ProPess uygulama saati", value="", disabled=not propess)

    with c3:
        cytotec = st.checkbox("Cytotec (misoprostol) verilmiş")
        cytotec_dose = st.number_input(
            "Cytotec dozu (mcg)",
            min_value=0,
            max_value=1000,
            value=25,
            step=25,
            disabled=not cytotec,
        )
        cytotec_time = st.text_input("Son Cytotec uygulama saati", value="", disabled=not cytotec)

    c1, c2 = st.columns(2)

    with c1:
        hypotension = st.checkbox("Maternal hipotansiyon")

    with c2:
        maternal_hypoxia = st.checkbox("Maternal hipoksemi")


    st.header("5️⃣ Hastane Kalibrasyonu")

    st.success("Preset aktif: 2 küçük kare = 1 dakika • 1 küçük kare = 30 sn • ana FHR yatay grid = 20 bpm")

    manual_calibration = st.checkbox(
        "Bazal FHR otomatik kalibrasyonu yanlışsa manuel düzeltme aç",
        value=False,
    )

    if manual_calibration:
        manual_offset_bpm = st.slider(
            "FHR kalibrasyon düzeltmesi (bpm)",
            min_value=-40,
            max_value=40,
            value=0,
            step=1,
            help="Örneğin otomatik bazal 150, gerçek bazal 140 ise -10 seçin.",
        )
    else:
        manual_offset_bpm = 0


    if st.button("🫀 NST / CTG'yi Analiz Et", type="primary", use_container_width=True):

        calibration = get_hospital_calibration(image)

        purple_mask = purple_trace_mask(fhr_crop)
        trace_y, quality = extract_trace_from_mask(purple_mask)
        extraction_method = "Mor FHR çizgi segmentasyonu"

        if quality < 0.45:
            trace_y_fallback, fallback_quality = fallback_dark_trace(fhr_crop)

            if fallback_quality > quality:
                trace_y = trace_y_fallback
                quality = fallback_quality
                extraction_method = "Koyu çizgi fallback"

        if quality < 0.35:
            st.error(
                f"❌ FHR çizgisi güvenilir şekilde çıkarılamadı (yakalama %{quality*100:.0f}). "
                "Bu görüntü için otomatik sonuç üretmiyorum."
            )
            st.stop()

        fhr = convert_trace_to_bpm(
            trace_y,
            fhr_top,
            calibration,
            manual_offset_bpm=manual_offset_bpm,
        )

        fhr = robust_smooth(fhr, window=3)

        # Aşırı uçları rapor öncesi kontrol et
        plausible_fraction = np.mean((fhr >= 70) & (fhr <= 210))

        if plausible_fraction < 0.80:
            st.error(
                "❌ BPM kalibrasyonu fizyolojik aralıkla uyumsuz görünüyor. "
                "Manuel kalibrasyon düzeltmesini açın veya görüntü formatını kontrol edin."
            )
            st.stop()

        baseline = calculate_baseline(fhr)

        if baseline is None:
            st.error("❌ Bazal FHR hesaplanamadı.")
            st.stop()

        variability_amp, variability = calculate_variability(fhr, baseline)

        # Zaman kalibrasyonu tüm export görüntüsündeki aynı piksel ölçeğini kullanır.
        seconds_per_pixel = calibration["seconds_per_pixel"]

        if gestational_age >= 32:
            acc_amp, acc_dur = 15.0, 15.0
        else:
            acc_amp, acc_dur = 10.0, 10.0

        accelerations = detect_accelerations_morphologic(
            fhr,
            baseline,
            seconds_per_pixel,
            amplitude_threshold=acc_amp,
            duration_threshold=acc_dur,
        )

        decelerations = detect_excursions(
            fhr,
            baseline,
            seconds_per_pixel,
            direction="down",
            amplitude_threshold=15.0,
            duration_threshold=15.0,
        )

        total_minutes = len(fhr) * seconds_per_pixel / 60.0

        nst_status, valid_accels, nst_detail = nst_assessment(
            accelerations,
            gestational_age,
            total_minutes,
        )

        category = classify_ctg(
            baseline,
            variability,
            recurrent_late,
            recurrent_variable,
            sinusoidal,
        )

        pattern_text = pattern_summary(
            baseline,
            variability,
            early,
            recurrent_late,
            recurrent_variable,
            prolonged,
            sinusoidal,
        )

        recommendations = acog_pattern_recommendations(
            category,
            baseline,
            variability,
            early,
            recurrent_late,
            recurrent_variable,
            prolonged,
            sinusoidal,
            tachysystole,
            oxytocin,
            propess,
            cytotec,
            hypotension,
            maternal_hypoxia,
        )

        st.divider()
        st.header("📊 ANALİZ SONUCU")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bazal FHR", f"{baseline:.0f} bpm")
        c2.metric("Variabilite", variability)
        c3.metric("Uygun akselerasyon", len(valid_accels))
        c4.metric("Kaba deselerasyon", len(decelerations))

        st.subheader("NST")
        if nst_status == "Reaktivite kriteri karşılandı":
            st.success("✅ " + nst_status)
        elif nst_status == "Süre yetersiz":
            st.info("ℹ️ " + nst_status)
        else:
            st.warning("⚠️ " + nst_status)

        st.write(nst_detail)

        with st.expander("📈 Otomatik akselerasyon ayrıntıları"):
            if not valid_accels:
                st.write("Uygun akselerasyon saptanmadı.")
            else:
                for i, acc in enumerate(valid_accels, 1):
                    st.write(
                        f"**Akselerasyon {i}:** "
                        f"amplitüd {acc['amplitude']:.1f} bpm • "
                        f"onset-return süresi {acc['duration']:.1f} sn • "
                        f"peak {acc['extreme']:.0f} bpm"
                    )

        st.subheader("İntrapartum FHR Kategorisi")
        if category == "I":
            st.success("🟢 CATEGORY I")
        elif category == "II":
            st.warning("🟡 CATEGORY II")
        else:
            st.error("🔴 CATEGORY III")

        st.subheader("🔎 Patern")
        st.write(pattern_text)

        st.subheader("🩺 Paterne Özgü Yaklaşım")
        for rec in recommendations:
            st.write("• " + rec)

        st.caption(
            "Öneri motoru üç-kategorili intrapartum FHR sınıflamasını ve paterne özgü "
            "değerlendirmeyi esas alır. Category II tek başına tek bir müdahale anlamına gelmez."
        )

        st.subheader("📈 Dijitalize edilmiş FHR")
        fig = make_plot(
            fhr,
            baseline,
            valid_accels,
            decelerations,
            seconds_per_pixel,
        )
        st.pyplot(fig)

        with st.expander("🔧 Teknik kalite / kalibrasyon"):
            st.write(f"Çizgi çıkarma yöntemi: **{extraction_method}**")
            st.write(f"FHR çizgi yakalama oranı: **%{quality*100:.1f}**")
            st.write(f"1 küçük kare: **{calibration['small_square_px']:.1f} px ≈ 30 sn**")
            st.write(f"Ana yatay FHR grid: **{calibration['major_grid_px']:.1f} px ≈ 20 bpm**")
            st.write(f"Görüntülenen yaklaşık süre: **{total_minutes:.1f} dk**")
            st.write(f"Manuel FHR offset: **{manual_offset_bpm:+d} bpm**")

        pdf = create_pdf(
            patient_name,
            patient_age,
            gestational_age,
            gravida,
            parity,
            living,
            propess,
            propess_time,
            cytotec,
            cytotec_dose,
            cytotec_time,
            oxytocin,
            tachysystole,
            hypotension,
            maternal_hypoxia,
            baseline,
            variability,
            variability_amp,
            len(valid_accels),
            len(decelerations),
            nst_status,
            category,
            pattern_text,
            recommendations,
            quality,
            total_minutes,
        )

        st.divider()
        st.subheader("📄 Rapor")

        st.download_button(
            "📄 PDF Raporunu İndir",
            data=pdf,
            file_name="NST_CTG_Raporu_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.caption(
            "Önemli: 'kaba deselerasyon' sayısı yalnızca ≥15 bpm / ≥15 sn eşik excursion taramasıdır; "
            "late/early/variable morfolojisi bu sürümde klinisyen doğrulamasına dayanır."
        )
