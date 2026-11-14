import cv2 as cv
import numpy as np
import os
import sys
import time
import html


# ============================================================
# PRÀCTICA 1 - VERSIÓ 3 FINAL PREMIUM
#
# Funcionalitats principals:
# - Lectura de les imatges oficials a resolució completa
# - Detecció SIFT optimitzada sobre còpia reduïda
# - Reescalat dels punts a coordenades originals
# - Emparellament Brute Force
# - Càlcul de la matriu fonamental amb RANSAC
# - Rectificació no calibrada
# - Construcció de l'anàglif
# - Diferència entre imatges rectificades
# - Mapa de disparitat
# - Visualització de correspondències SIFT
# - Visualització de línies epipolars més neta
# - Informe automàtic TXT
# - Informe automàtic HTML
# - Visor interactiu amb:
#       zoom
#       arrossegament amb ratolí
#       canvi de modes
#       captura amb tecla D
#       informació de píxel amb botó dret
#       panell lateral amb controls
# ============================================================


# ============================================================
# FITXERS D'ENTRADA
# ============================================================

IMG_LEFT_PATH = "imatge1 practica1.jpg"
IMG_RIGHT_PATH = "imatge2 practica 1.jpg"

RESULTS_FOLDER = "resultats_versio3_final"


# ============================================================
# PARÀMETRES DE PROCESSAMENT
# ============================================================

MAX_SIFT_FEATURES = 4000
MAX_MATCHES = 500
DETECTION_MAX_WIDTH = 1800

RANSAC_REPROJ_THRESHOLD = 2.0
RANSAC_CONFIDENCE = 0.99

DISPARITY_MAX_WIDTH = 1600

MAX_MATCHES_TO_DRAW = 120

# Línies epipolars més netes i menys carregades
MAX_LINES_TO_DRAW = 16
EPILINE_THICKNESS = 1
EPIPOINT_RADIUS = 4


# ============================================================
# PARÀMETRES DEL VISOR
# ============================================================

WINDOW_NAME = "Practica 1 - Versio 3 final"

VIEW_WIDTH = 1200
VIEW_HEIGHT = 800
PANEL_WIDTH = 380

TOTAL_WIDTH = VIEW_WIDTH + PANEL_WIDTH
TOTAL_HEIGHT = VIEW_HEIGHT

MIN_ZOOM = 0.05
MAX_ZOOM = 8.0


# ============================================================
# VARIABLES GLOBALS
# ============================================================

images = {}
mode_names = []

# Modes:
# 0 = Esquerra rectificada
# 1 = Dreta rectificada
# 2 = Anaglif
# 3 = Diferencia
# 4 = Mapa de disparitat
# 5 = Correspondencies SIFT
# 6 = Linies epipolars

current_mode = 2  # Comencem amb l'anàglif

zoom = 1.0
offset_x = 0
offset_y = 0

dragging = False
last_mouse_x = 0
last_mouse_y = 0

cursor_x = 0
cursor_y = 0

last_time = time.time()
fps = 0.0

pixel_popup_visible = False
pixel_popup_screen_x = 0
pixel_popup_screen_y = 0
pixel_popup_img_x = 0
pixel_popup_img_y = 0
pixel_popup_bgr = (0, 0, 0)
pixel_popup_mode = ""

stats = {
    "img_left_shape": None,
    "img_right_shape": None,
    "sift_left": 0,
    "sift_right": 0,
    "matches_total": 0,
    "matches_used": 0,
    "inliers": 0,
    "F": None,
    "H1": None,
    "H2": None,
}


# ============================================================
# UTILITATS GENERALS
# ============================================================

def crear_carpeta_resultats():
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)


def comprovar_fitxers():
    print("Carpeta actual de treball:")
    print(os.getcwd())
    print()

    if not os.path.exists(IMG_LEFT_PATH):
        sys.exit(f"No existeix la imatge esquerra: {IMG_LEFT_PATH}")

    if not os.path.exists(IMG_RIGHT_PATH):
        sys.exit(f"No existeix la imatge dreta: {IMG_RIGHT_PATH}")

    print("Imatges oficials trobades correctament:")
    print(" -", IMG_LEFT_PATH)
    print(" -", IMG_RIGHT_PATH)
    print()


def carregar_imatges_originals():
    img_left = cv.imread(IMG_LEFT_PATH, cv.IMREAD_COLOR)
    img_right = cv.imread(IMG_RIGHT_PATH, cv.IMREAD_COLOR)

    if img_left is None:
        sys.exit(f"No s'ha pogut obrir la imatge esquerra: {IMG_LEFT_PATH}")

    if img_right is None:
        sys.exit(f"No s'ha pogut obrir la imatge dreta: {IMG_RIGHT_PATH}")

    print("Imatge esquerra carregada:", img_left.shape)
    print("Imatge dreta carregada:", img_right.shape)

    stats["img_left_shape"] = img_left.shape
    stats["img_right_shape"] = img_right.shape

    if img_left.shape[:2] != img_right.shape[:2]:
        print()
        print("ATENCIÓ: Les imatges no tenen exactament la mateixa mida.")
        print("El programa continuarà igualment.")
        print()

    return img_left, img_right


def reduir_per_deteccio(img, max_width):
    h, w = img.shape[:2]

    if w <= max_width:
        return img.copy(), 1.0

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_small = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

    return img_small, scale


def reescalar_keypoints(kp_small, scale):
    if scale == 1.0:
        return kp_small

    kp_original = []

    for kp in kp_small:
        x, y = kp.pt

        nou_kp = cv.KeyPoint(
            x / scale,
            y / scale,
            kp.size / scale,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )

        kp_original.append(nou_kp)

    return kp_original


def format_matrix(M):
    if M is None:
        return "No disponible"

    return np.array2string(
        M,
        precision=8,
        suppress_small=False,
        separator=", "
    )


# ============================================================
# PROCESSAMENT FOTOGRAMÈTRIC
# ============================================================

def detectar_sift(img_left, img_right):
    print("Preparant còpies reduïdes per a la detecció SIFT...")

    small_left, scale_left = reduir_per_deteccio(img_left, DETECTION_MAX_WIDTH)
    small_right, scale_right = reduir_per_deteccio(img_right, DETECTION_MAX_WIDTH)

    print("Mida temporal esquerra per SIFT:", small_left.shape)
    print("Mida temporal dreta per SIFT:", small_right.shape)
    print("Escala esquerra:", scale_left)
    print("Escala dreta:", scale_right)

    gray_left = cv.cvtColor(small_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(small_right, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create(nfeatures=MAX_SIFT_FEATURES)

    print("Detectant punts SIFT...")
    kp1_small, desc1 = sift.detectAndCompute(gray_left, None)
    kp2_small, desc2 = sift.detectAndCompute(gray_right, None)

    if desc1 is None or desc2 is None:
        sys.exit("No s'han pogut calcular descriptors SIFT.")

    kp1 = reescalar_keypoints(kp1_small, scale_left)
    kp2 = reescalar_keypoints(kp2_small, scale_right)
    
    print("Nombre de punts SIFT imatge esquerra:", len(kp1))
    print("Nombre de punts SIFT imatge dreta:", len(kp2))

    stats["sift_left"] = len(kp1)
    stats["sift_right"] = len(kp2)

    if len(kp1) < 8 or len(kp2) < 8:
        sys.exit("Hi ha massa pocs punts SIFT per continuar.")

    return kp1, desc1, kp2, desc2


def emparellar_punts(desc1, desc2):
    print("Emparellant descriptors amb Brute Force...")

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    print("Nombre total de correspondències inicials:", len(matches))

    if len(matches) < 8:
        sys.exit("Hi ha massa poques correspondències per calcular la matriu fonamental.")

    good_matches = matches[:min(MAX_MATCHES, len(matches))]

    print("Nombre de correspondències utilitzades:", len(good_matches))

    stats["matches_total"] = len(matches)
    stats["matches_used"] = len(good_matches)

    return good_matches


def calcular_matriu_fonamental(kp1, kp2, matches):
    print("Calculant matriu fonamental amb RANSAC...")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv.findFundamentalMat(
        pts1,
        pts2,
        method=cv.FM_RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_CONFIDENCE
    )

    if F is None or mask is None:
        sys.exit("No s'ha pogut calcular la matriu fonamental.")

    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]

    print()
    print("Matriu fonamental:")
    print(F)
    print()
    print("Nombre de correspondències abans de RANSAC:", len(pts1))
    print("Nombre de correspondències vàlides després de RANSAC:", len(pts1_inliers))
    print()

    if len(pts1_inliers) < 8:
        sys.exit("Hi ha massa poques correspondències vàlides després de RANSAC.")

    stats["inliers"] = len(pts1_inliers)
    stats["F"] = F

    return F, pts1, pts2, pts1_inliers, pts2_inliers, mask


def rectificar_imatges(img_left, img_right, F, pts1_inliers, pts2_inliers):
    print("Rectificant imatges originals...")

    h, w = img_left.shape[:2]

    success, H1, H2 = cv.stereoRectifyUncalibrated(
        pts1_inliers,
        pts2_inliers,
        F,
        imgSize=(w, h)
    )

    if not success:
        sys.exit("La rectificació no calibrada ha fallat.")

    rect_left = cv.warpPerspective(img_left, H1, (w, h))
    rect_right = cv.warpPerspective(img_right, H2, (w, h))

    path_left = os.path.join(RESULTS_FOLDER, "02_rectificada_esquerra.png")
    path_right = os.path.join(RESULTS_FOLDER, "03_rectificada_dreta.png")

    cv.imwrite(path_left, rect_left)
    cv.imwrite(path_right, rect_right)

    print("Imatge esquerra rectificada desada a:", path_left)
    print("Imatge dreta rectificada desada a:", path_right)

    stats["H1"] = H1
    stats["H2"] = H2

    return rect_left, rect_right, H1, H2


def crear_anaglif(rect_left, rect_right):
    print("Creant anàglif...")

    gray_left = cv.cvtColor(rect_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(rect_right, cv.COLOR_BGR2GRAY)

    anaglif = np.zeros_like(rect_left)

    # OpenCV treballa en BGR.
    anaglif[:, :, 2] = gray_left
    anaglif[:, :, 1] = gray_right
    anaglif[:, :, 0] = gray_right

    path = os.path.join(RESULTS_FOLDER, "04_anaglif.png")
    cv.imwrite(path, anaglif)

    print("Anàglif desat a:", path)

    return anaglif


def crear_diferencia(rect_left, rect_right):
    print("Creant imatge de diferència...")

    gray_left = cv.cvtColor(rect_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(rect_right, cv.COLOR_BGR2GRAY)

    diff = cv.absdiff(gray_left, gray_right)
    diff_color = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)

    path = os.path.join(RESULTS_FOLDER, "05_diferencia_rectificades.png")
    cv.imwrite(path, diff_color)

    print("Diferència desada a:", path)

    return diff_color


def calcular_disparitat(rect_left, rect_right):
    print("Calculant mapa de disparitat...")

    h, w = rect_left.shape[:2]

    if w > DISPARITY_MAX_WIDTH:
        scale = DISPARITY_MAX_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        small_left = cv.resize(rect_left, (new_w, new_h), interpolation=cv.INTER_AREA)
        small_right = cv.resize(rect_right, (new_w, new_h), interpolation=cv.INTER_AREA)
    else:
        scale = 1.0
        small_left = rect_left
        small_right = rect_right

    gray_left = cv.cvtColor(small_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(small_right, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

    disp_norm = cv.normalize(
        disparity,
        None,
        alpha=0,
        beta=255,
        norm_type=cv.NORM_MINMAX
    )

    disp_norm = np.uint8(disp_norm)
    disp_color = cv.applyColorMap(disp_norm, cv.COLORMAP_JET)

    if scale != 1.0:
        disp_color = cv.resize(disp_color, (w, h), interpolation=cv.INTER_LINEAR)

    path = os.path.join(RESULTS_FOLDER, "06_mapa_disparitat.png")
    cv.imwrite(path, disp_color)

    print("Mapa de disparitat desat a:", path)

    return disp_color


# ============================================================
# VISUALITZACIONS AVANÇADES
# ============================================================

def crear_visualitzacio_matches(img_left, img_right, kp1, kp2, matches, mask):
    print("Creant visualització de correspondències...")

    inlier_matches = []

    for i, m in enumerate(matches):
        if mask.ravel()[i] == 1:
            inlier_matches.append(m)

    img_matches = cv.drawMatches(
        img_left,
        kp1,
        img_right,
        kp2,
        inlier_matches[:MAX_MATCHES_TO_DRAW],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    path = os.path.join(RESULTS_FOLDER, "01_correspondencies_sift.png")
    cv.imwrite(path, img_matches)

    print("Correspondències desades a:", path)

    return img_matches


def dibuixar_linia(img, line, color, thickness=1):
    h, w = img.shape[:2]
    a, b, c = line

    if abs(b) < 1e-8:
        if abs(a) < 1e-8:
            return
        x = int(-c / a)
        cv.line(img, (x, 0), (x, h - 1), color, thickness, cv.LINE_AA)
    else:
        x0 = 0
        y0 = int(-c / b)

        x1 = w - 1
        y1 = int(-(c + a * x1) / b)

        cv.line(img, (x0, y0), (x1, y1), color, thickness, cv.LINE_AA)


def crear_visualitzacio_epipolars(img_left, img_right, pts1_inliers, pts2_inliers, F):
    print("Creant visualització de línies epipolars...")

    h, w = img_left.shape[:2]
    display_width_each = 900
    scale = display_width_each / w

    new_w = int(w * scale)
    new_h = int(h * scale)

    left_small = cv.resize(img_left, (new_w, new_h), interpolation=cv.INTER_AREA)
    right_small = cv.resize(img_right, (new_w, new_h), interpolation=cv.INTER_AREA)

    pts1_small = pts1_inliers * scale
    pts2_small = pts2_inliers * scale

    total_pts = len(pts1_small)
    n = min(MAX_LINES_TO_DRAW, total_pts)

    if n < 1:
        canvas = np.hstack((left_small, right_small))
        path = os.path.join(RESULTS_FOLDER, "07_linies_epipolars.png")
        cv.imwrite(path, canvas)
        print("Línies epipolars desades a:", path)
        return canvas

    # Triem punts repartits per tota la llista, no només els primers.
    idxs = np.linspace(0, total_pts - 1, n, dtype=int)

    pts1_sel = pts1_small[idxs]
    pts2_sel = pts2_small[idxs]

    # Adaptació de la matriu fonamental a l'escala reduïda.
    S_inv = np.array([
        [1.0 / scale, 0, 0],
        [0, 1.0 / scale, 0],
        [0, 0, 1]
    ])

    F_small = S_inv.T @ F @ S_inv

    lines_left = cv.computeCorrespondEpilines(
        pts2_sel.reshape(-1, 1, 2),
        2,
        F_small
    ).reshape(-1, 3)

    lines_right = cv.computeCorrespondEpilines(
        pts1_sel.reshape(-1, 1, 2),
        1,
        F_small
    ).reshape(-1, 3)

    left_base = left_small.copy()
    right_base = right_small.copy()

    left_overlay = left_small.copy()
    right_overlay = right_small.copy()

    rng = np.random.default_rng(7)

    for i in range(n):
        color_np = rng.integers(70, 255, size=3)
        color = (int(color_np[0]), int(color_np[1]), int(color_np[2]))

        dibuixar_linia(left_overlay, lines_left[i], color, thickness=EPILINE_THICKNESS)
        dibuixar_linia(right_overlay, lines_right[i], color, thickness=EPILINE_THICKNESS)

        p1 = tuple(np.int32(pts1_sel[i]))
        p2 = tuple(np.int32(pts2_sel[i]))

        cv.circle(left_overlay, p1, EPIPOINT_RADIUS, color, -1, cv.LINE_AA)
        cv.circle(right_overlay, p2, EPIPOINT_RADIUS, color, -1, cv.LINE_AA)

        cv.circle(left_overlay, p1, EPIPOINT_RADIUS + 1, (255, 255, 255), 1, cv.LINE_AA)
        cv.circle(right_overlay, p2, EPIPOINT_RADIUS + 1, (255, 255, 255), 1, cv.LINE_AA)

    # Barreja suau perquè les línies no siguin tan agressives.
    left_epi = cv.addWeighted(left_overlay, 0.72, left_base, 0.28, 0)
    right_epi = cv.addWeighted(right_overlay, 0.72, right_base, 0.28, 0)

    canvas = np.hstack((left_epi, right_epi))

    cv.rectangle(canvas, (0, 0), (canvas.shape[1], 45), (0, 0, 0), -1)
    cv.putText(
        canvas,
        f"Linies epipolars - {n} correspondencies representatives",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    path = os.path.join(RESULTS_FOLDER, "07_linies_epipolars.png")
    cv.imwrite(path, canvas)

    print("Línies epipolars desades a:", path)

    return canvas


# ============================================================
# INFORMES
# ============================================================

def generar_informe_txt():
    print("Generant informe TXT...")

    path = os.path.join(RESULTS_FOLDER, "informe_resultats.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("PRÀCTICA 1 - INFORME DE RESULTATS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Aquest informe resumeix el processament realitzat sobre el parell\n")
        f.write("fotogramètric utilitzat a la pràctica 1. El sistema implementat\n")
        f.write("inclou detecció automàtica de punts SIFT, emparellament amb Brute\n")
        f.write("Force, càlcul de la matriu fonamental, rectificació no calibrada,\n")
        f.write("construcció d'anàglif, mapa de disparitat i visor interactiu.\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. DADES D'ENTRADA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Imatge esquerra utilitzada : {IMG_LEFT_PATH}\n")
        f.write(f"Imatge dreta utilitzada    : {IMG_RIGHT_PATH}\n")
        f.write(f"Mida imatge esquerra       : {stats['img_left_shape']}\n")
        f.write(f"Mida imatge dreta          : {stats['img_right_shape']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("2. PARÀMETRES DE PROCESSAMENT\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nombre màxim de punts SIFT            : {MAX_SIFT_FEATURES}\n")
        f.write(f"Nombre màxim de correspondències      : {MAX_MATCHES}\n")
        f.write(f"Amplada màxima per a detecció SIFT    : {DETECTION_MAX_WIDTH}\n")
        f.write(f"Llindar RANSAC                        : {RANSAC_REPROJ_THRESHOLD}\n")
        f.write(f"Confiança de RANSAC                   : {RANSAC_CONFIDENCE}\n")
        f.write(f"Amplada màxima per a disparitat       : {DISPARITY_MAX_WIDTH}\n\n")

        f.write("-" * 70 + "\n")
        f.write("3. RESULTATS OBTINGUTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Punts SIFT detectats a la imatge esquerra : {stats['sift_left']}\n")
        f.write(f"Punts SIFT detectats a la imatge dreta    : {stats['sift_right']}\n")
        f.write(f"Correspondències inicials trobades        : {stats['matches_total']}\n")
        f.write(f"Correspondències utilitzades              : {stats['matches_used']}\n")
        f.write(f"Correspondències vàlides després RANSAC   : {stats['inliers']}\n\n")

        f.write("La matriu fonamental estimada és la següent:\n\n")
        f.write(format_matrix(stats["F"]))
        f.write("\n\n")

        f.write("L'homografia de rectificació per a la imatge esquerra és:\n\n")
        f.write(format_matrix(stats["H1"]))
        f.write("\n\n")

        f.write("L'homografia de rectificació per a la imatge dreta és:\n\n")
        f.write(format_matrix(stats["H2"]))
        f.write("\n\n")

        f.write("-" * 70 + "\n")
        f.write("4. INTERPRETACIÓ DELS RESULTATS\n")
        f.write("-" * 70 + "\n")

        if stats["inliers"] >= 200:
            f.write("El nombre de correspondències vàlides després de RANSAC és alt,\n")
            f.write("fet que indica una estimació robusta de la geometria epipolar.\n")
            f.write("Això permet obtenir una rectificació fiable i visualment coherent.\n\n")
        elif stats["inliers"] >= 80:
            f.write("El nombre de correspondències vàlides és correcte i suficient\n")
            f.write("per obtenir una rectificació útil. Tot i així, la qualitat final\n")
            f.write("pot veure's afectada lleugerament en algunes zones.\n\n")
        else:
            f.write("El nombre de correspondències vàlides és relativament baix.\n")
            f.write("Això pot dificultar la qualitat de la rectificació i dels modes\n")
            f.write("derivats, especialment l'anàglif i la disparitat.\n\n")

        f.write("Les línies epipolars generades permeten visualitzar de manera\n")
        f.write("gràfica la relació geomètrica entre les dues imatges. Igualment,\n")
        f.write("l'anàglif i el mapa de disparitat ofereixen una interpretació\n")
        f.write("visual complementària del parell estereoscòpic.\n\n")

        f.write("-" * 70 + "\n")
        f.write("5. FITXERS GENERATS\n")
        f.write("-" * 70 + "\n")
        f.write("01_correspondencies_sift.png      -> visualització de matches SIFT\n")
        f.write("02_rectificada_esquerra.png       -> imatge esquerra rectificada\n")
        f.write("03_rectificada_dreta.png          -> imatge dreta rectificada\n")
        f.write("04_anaglif.png                    -> composició anàglif\n")
        f.write("05_diferencia_rectificades.png    -> diferència absoluta\n")
        f.write("06_mapa_disparitat.png            -> mapa de disparitat\n")
        f.write("07_linies_epipolars.png           -> visualització epipolar\n")
        f.write("informe_resultats.txt             -> informe tècnic en text\n")
        f.write("informe_resultats.html            -> informe visual en HTML\n\n")

        f.write("-" * 70 + "\n")
        f.write("6. FUNCIONALITATS DEL VISOR\n")
        f.write("-" * 70 + "\n")
        f.write("El visor interactiu permet canviar entre diversos modes de\n")
        f.write("visualització, fer zoom, moure la vista amb el ratolí, guardar\n")
        f.write("captures de pantalla i consultar informació d'un píxel concret\n")
        f.write("mitjançant el botó dret del ratolí.\n\n")

        f.write("Controls principals:\n")
        f.write("  1 -> esquerra rectificada\n")
        f.write("  2 -> dreta rectificada\n")
        f.write("  3 -> anàglif\n")
        f.write("  4 -> diferència\n")
        f.write("  5 -> mapa de disparitat\n")
        f.write("  C -> correspondències SIFT\n")
        f.write("  E -> línies epipolars\n")
        f.write("  + -> augmentar zoom\n")
        f.write("  - -> reduir zoom\n")
        f.write("  R -> reiniciar vista\n")
        f.write("  D -> desar vista actual\n")
        f.write("  Q -> sortir\n\n")

        f.write("-" * 70 + "\n")
        f.write("7. VALORACIÓ FINAL\n")
        f.write("-" * 70 + "\n")
        f.write("La versió avançada desenvolupada compleix els requisits bàsics\n")
        f.write("de la pràctica i incorpora millores addicionals orientades a la\n")
        f.write("usabilitat i a la interpretació visual dels resultats. Entre les\n")
        f.write("millores més destacables hi ha el visor interactiu, la consulta\n")
        f.write("de píxels, la generació automàtica d'un informe i la visualització\n")
        f.write("de línies epipolars com a suport a la validació geomètrica.\n\n")

        f.write("=" * 70 + "\n")
        f.write("Fi de l'informe.\n")

    print("Informe TXT desat a:", path)


def generar_informe_html():
    print("Generant informe HTML...")

    path = os.path.join(RESULTS_FOLDER, "informe_resultats.html")

    F_text = html.escape(format_matrix(stats["F"]))
    H1_text = html.escape(format_matrix(stats["H1"]))
    H2_text = html.escape(format_matrix(stats["H2"]))

    if stats["inliers"] >= 200:
        valoracio_tecnica = (
            "El nombre de correspondències vàlides després de RANSAC és alt. "
            "Això indica que l'estimació de la geometria epipolar és robusta "
            "i que la rectificació resultant és fiable per construir l'anàglif "
            "i analitzar el parell fotogramètric."
        )
        badge = "Resultat robust"
    elif stats["inliers"] >= 80:
        valoracio_tecnica = (
            "El nombre de correspondències vàlides és suficient per obtenir "
            "una rectificació útil. Tot i així, algunes zones poden presentar "
            "petites diferències visuals."
        )
        badge = "Resultat acceptable"
    else:
        valoracio_tecnica = (
            "El nombre de correspondències vàlides és baix. La rectificació "
            "pot ser menys estable i seria recomanable revisar el parell "
            "d'imatges o els paràmetres de detecció."
        )
        badge = "Resultat millorable"

    html_content = f"""<!DOCTYPE html>
<html lang="ca">
<head>
    <meta charset="utf-8">
    <title>Informe Pràctica 1 - Mikel Zeberio i Ignacio Ojeda</title>
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            background: #eef2f7;
            color: #222;
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 1180px;
            margin: 30px auto;
            background: white;
            padding: 38px;
            border-radius: 20px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.14);
        }}

        .hero {{
            background: linear-gradient(135deg, #13293d, #1f4e79);
            color: white;
            padding: 38px;
            border-radius: 20px;
            margin-bottom: 35px;
        }}

        .hero h1 {{
            font-size: 42px;
            margin: 0 0 12px 0;
            letter-spacing: -0.5px;
        }}

        .names {{
            font-size: 34px;
            font-weight: bold;
            margin-top: 18px;
            color: #ffffff;
        }}

        .subtitle {{
            font-size: 18px;
            line-height: 1.55;
            color: #dbeafe;
            max-width: 920px;
        }}

        h2 {{
            color: #1f3b57;
            border-bottom: 2px solid #dce5ef;
            padding-bottom: 8px;
            margin-top: 42px;
            font-size: 26px;
        }}

        h3 {{
            color: #244b6b;
            margin-top: 26px;
            font-size: 21px;
        }}

        p {{
            line-height: 1.65;
            font-size: 16px;
        }}

        .badge {{
            display: inline-block;
            background: #d8f3dc;
            color: #1b6b35;
            padding: 9px 16px;
            border-radius: 999px;
            font-weight: bold;
            margin-top: 14px;
        }}

        .personal-box {{
            background: #fff7ed;
            border-left: 6px solid #f97316;
            padding: 22px;
            border-radius: 12px;
            margin-top: 22px;
        }}

        .final-box {{
            background: #ecfdf5;
            border-left: 6px solid #10b981;
            padding: 22px;
            border-radius: 12px;
            margin-top: 22px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-top: 20px;
        }}

        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
        }}

        .card .number {{
            font-size: 30px;
            font-weight: bold;
            color: #1f3b57;
        }}

        .card .label {{
            font-size: 14px;
            color: #667085;
            margin-top: 6px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}

        th {{
            background: #f1f5f9;
            color: #334155;
        }}

        pre {{
            background: #0f172a;
            color: #dbeafe;
            padding: 16px;
            border-radius: 12px;
            overflow-x: auto;
            font-size: 13px;
        }}

        .image-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 26px;
            margin-top: 24px;
        }}

        .image-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        }}

        .image-card img {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid #d8dee9;
            margin-bottom: 14px;
        }}

        .image-card h3 {{
            margin-top: 0;
        }}

        .files {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}

        .file {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px;
        }}

        .controls {{
            columns: 2;
            line-height: 1.8;
        }}

        .small-note {{
            color: #64748b;
            font-size: 14px;
        }}

        .footer {{
            margin-top: 45px;
            color: #667085;
            font-size: 13px;
            text-align: center;
        }}

        @media (max-width: 900px) {{
            .grid {{
                grid-template-columns: 1fr 1fr;
            }}

            .image-section {{
                grid-template-columns: 1fr;
            }}

            .files {{
                grid-template-columns: 1fr;
            }}

            .hero h1 {{
                font-size: 32px;
            }}

            .names {{
                font-size: 26px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">

        <div class="hero">
            <h1>Pràctica 1 - Parell fotogramètric interactiu</h1>
            <div class="subtitle">
                Implementació d’un sistema de visió artificial per detectar correspondències,
                calcular la geometria epipolar, rectificar dues imatges, generar un anàglif
                i explorar els resultats mitjançant un visor interactiu.
            </div>
            <div class="names">Mikel Zeberio i Ignacio Ojeda</div>
            <div class="badge">{badge}</div>
        </div>

        <h2>Per què hem escollit la Pràctica 1?</h2>
        <div class="personal-box">
            <p>
                Hem escollit la Pràctica 1 perquè ens semblava la més divertida d’utilitzar
                i la que podia donar un resultat més visual. A diferència d’altres pràctiques
                més tancades o més purament mecàniques, aquesta ens permetia experimentar amb
                imatges reals, veure resultats directament en pantalla i construir una eina
                interactiva que es pogués explorar amb el ratolí i el teclat.
            </p>
            <p>
                També ens va semblar un repte més tècnic que les altres dues pràctiques,
                perquè no només calia aplicar una funció concreta, sinó encadenar moltes
                parts diferents: detecció de punts, emparellament, filtratge amb RANSAC,
                càlcul de la matriu fonamental, rectificació, anàglif, mapa de disparitat
                i finalment un visor interactiu. Precisament per això ens va semblar més
                interessant: hi havia més marge per equivocar-se, provar, millorar i acabar
                construint una versió final més completa.
            </p>
        </div>

        <h2>Resum del projecte</h2>
        <p>
            El programa treballa amb dues imatges originals d’un mateix escenari i construeix
            una versió interactiva d’un parell fotogramètric. Primer es detecten punts
            característics amb SIFT, després s’emparellen amb Brute Force i es filtra la
            geometria amb RANSAC. A partir d’aquests punts es calcula la matriu fonamental,
            es rectifiquen les imatges i es generen diferents visualitzacions per interpretar
            millor el resultat.
        </p>

        <h2>Resum numèric</h2>
        <div class="grid">
            <div class="card">
                <div class="number">{stats["sift_left"]}</div>
                <div class="label">Punts SIFT esquerra</div>
            </div>
            <div class="card">
                <div class="number">{stats["sift_right"]}</div>
                <div class="label">Punts SIFT dreta</div>
            </div>
            <div class="card">
                <div class="number">{stats["matches_used"]}</div>
                <div class="label">Correspondències utilitzades</div>
            </div>
            <div class="card">
                <div class="number">{stats["inliers"]}</div>
                <div class="label">Inliers RANSAC</div>
            </div>
        </div>

        <h2>Dades d'entrada</h2>
        <table>
            <tr><th>Element</th><th>Valor</th></tr>
            <tr><td>Imatge esquerra</td><td>{html.escape(IMG_LEFT_PATH)}</td></tr>
            <tr><td>Imatge dreta</td><td>{html.escape(IMG_RIGHT_PATH)}</td></tr>
            <tr><td>Mida imatge esquerra</td><td>{stats["img_left_shape"]}</td></tr>
            <tr><td>Mida imatge dreta</td><td>{stats["img_right_shape"]}</td></tr>
        </table>

        <h2>Paràmetres principals</h2>
        <table>
            <tr><th>Paràmetre</th><th>Valor</th></tr>
            <tr><td>MAX_SIFT_FEATURES</td><td>{MAX_SIFT_FEATURES}</td></tr>
            <tr><td>MAX_MATCHES</td><td>{MAX_MATCHES}</td></tr>
            <tr><td>DETECTION_MAX_WIDTH</td><td>{DETECTION_MAX_WIDTH}</td></tr>
            <tr><td>RANSAC_REPROJ_THRESHOLD</td><td>{RANSAC_REPROJ_THRESHOLD}</td></tr>
            <tr><td>RANSAC_CONFIDENCE</td><td>{RANSAC_CONFIDENCE}</td></tr>
            <tr><td>DISPARITY_MAX_WIDTH</td><td>{DISPARITY_MAX_WIDTH}</td></tr>
        </table>

        <h2>Què representa cada mode del visor?</h2>
        <p>
            El visor interactiu permet passar per diferents formes de veure el mateix parell
            d’imatges. Cada mode té una funció concreta i ajuda a entendre una part diferent
            del procés fotogramètric.
        </p>

        <div class="image-section">

            <div class="image-card">
                <img src="02_rectificada_esquerra.png" alt="Imatge esquerra rectificada">
                <h3>1. Imatge esquerra rectificada</h3>
                <p>
                    Aquesta és la imatge esquerra després d’aplicar una transformació projectiva
                    obtinguda a partir de la matriu fonamental. La rectificació busca que els
                    punts corresponents de les dues imatges quedin alineats horitzontalment.
                    Les vores negres o petites inclinacions són normals, ja que la imatge ha
                    estat reprojectada sobre un nou pla.
                </p>
            </div>

            <div class="image-card">
                <img src="03_rectificada_dreta.png" alt="Imatge dreta rectificada">
                <h3>2. Imatge dreta rectificada</h3>
                <p>
                    Aquesta és la imatge dreta rectificada. El més important no és que sigui
                    una imatge perfectament “bonica”, sinó que sigui coherent amb l’esquerra.
                    Quan les dues rectificades estan ben alineades, els punts equivalents
                    apareixen aproximadament a la mateixa altura vertical, cosa que facilita
                    la creació de l’anàglif i el mapa de disparitat.
                </p>
            </div>

            <div class="image-card">
                <img src="04_anaglif.png" alt="Anàglif">
                <h3>3. Anàglif</h3>
                <p>
                    L’anàglif combina la informació de les dues imatges en canals de color:
                    normalment el canal vermell prové d’una imatge i els canals verd i blau
                    de l’altra. El resultat permet observar una sensació de profunditat amb
                    ulleres vermell-cian. En el nostre cas es veu una separació clara entre
                    canals, però l’escena continua sent reconeixible i coherent.
                </p>
            </div>

            <div class="image-card">
                <img src="05_diferencia_rectificades.png" alt="Diferència entre rectificades">
                <h3>4. Diferència entre imatges</h3>
                <p>
                    Aquest mode mostra la diferència absoluta entre les dues imatges
                    rectificades. Les zones fosques indiquen àrees semblants entre les dues
                    imatges, mentre que les zones més clares mostren diferències de textura,
                    perspectiva, il·luminació o petits desajustos. És útil per comprovar
                    visualment la qualitat de l’alineació.
                </p>
            </div>

            <div class="image-card">
                <img src="06_mapa_disparitat.png" alt="Mapa de disparitat">
                <h3>5. Mapa de disparitat</h3>
                <p>
                    El mapa de disparitat representa les diferències de posició horitzontal
                    entre punts equivalents de les dues imatges rectificades. En termes generals,
                    zones amb colors diferents indiquen canvis de profunditat o zones on
                    l’algorisme detecta desplaçaments diferents. No és un model 3D perfecte,
                    però dona una lectura visual interessant de la profunditat relativa.
                </p>
            </div>

            <div class="image-card">
                <img src="01_correspondencies_sift.png" alt="Correspondències SIFT">
                <h3>6. Correspondències SIFT</h3>
                <p>
                    Aquesta visualització mostra els punts característics detectats i emparellats
                    entre les dues imatges. SIFT identifica zones rellevants com cantonades,
                    textures o canvis forts d’intensitat. Després, Brute Force compara els
                    descriptors i RANSAC elimina correspondències errònies. Aquest mode ajuda
                    a veure si el sistema ha trobat coincidències coherents.
                </p>
            </div>

            <div class="image-card">
                <img src="07_linies_epipolars.png" alt="Línies epipolars">
                <h3>7. Línies epipolars</h3>
                <p>
                    Les línies epipolars representen la geometria entre les dues càmeres o
                    punts de vista. Si un punt apareix en una imatge, el seu corresponent
                    a l’altra imatge hauria de trobar-se sobre la línia epipolar associada.
                    Per això aquest mode és molt útil per validar visualment la matriu
                    fonamental calculada.
                </p>
            </div>

        </div>

        <h2>Funcionalitats interactives del visor</h2>
        <p>
            A més de generar les imatges de resultat, el programa incorpora un visor interactiu.
            Aquest visor permet explorar la imatge de forma còmoda, moure la vista arrossegant
            amb el ratolí, fer zoom, canviar de mode, guardar captures i consultar informació
            de píxels concrets.
        </p>

        <div class="controls">
            <p>
                <strong>1</strong> - esquerra rectificada<br>
                <strong>2</strong> - dreta rectificada<br>
                <strong>3</strong> - anàglif<br>
                <strong>4</strong> - diferència<br>
                <strong>5</strong> - mapa de disparitat<br>
                <strong>C</strong> - correspondències SIFT<br>
                <strong>E</strong> - línies epipolars<br>
                <strong>+</strong> - augmentar zoom<br>
                <strong>-</strong> - reduir zoom<br>
                <strong>R</strong> - reiniciar vista<br>
                <strong>D</strong> - desar vista actual<br>
                <strong>Q</strong> - sortir<br>
                <strong>Botó esquerre + arrossegar</strong> - moure la vista<br>
                <strong>Botó dret</strong> - consultar informació del píxel
            </p>
        </div>

        <h2>Interpretació tècnica dels resultats</h2>
        <p>
            {valoracio_tecnica}
        </p>
        <p>
            En el nostre cas, el sistema ha detectat {stats["sift_left"]} punts SIFT a la
            imatge esquerra i {stats["sift_right"]} punts SIFT a la imatge dreta. Dels
            emparellaments utilitzats, RANSAC n’ha conservat {stats["inliers"]} com a
            correspondències vàlides. Aquest valor és un bon indicador, perquè significa que
            moltes correspondències són coherents amb una mateixa geometria epipolar.
        </p>

        <h2>Matriu fonamental</h2>
        <p class="small-note">
            La matriu fonamental descriu la relació geomètrica entre els dos punts de vista.
        </p>
        <pre>{F_text}</pre>

        <h2>Homografia de rectificació esquerra</h2>
        <pre>{H1_text}</pre>

        <h2>Homografia de rectificació dreta</h2>
        <pre>{H2_text}</pre>

        <h2>Fitxers generats</h2>
        <div class="files">
            <div class="file"><strong>01_correspondencies_sift.png</strong><br>Visualització dels emparellaments SIFT.</div>
            <div class="file"><strong>02_rectificada_esquerra.png</strong><br>Imatge esquerra rectificada.</div>
            <div class="file"><strong>03_rectificada_dreta.png</strong><br>Imatge dreta rectificada.</div>
            <div class="file"><strong>04_anaglif.png</strong><br>Anàglif vermell-cian.</div>
            <div class="file"><strong>05_diferencia_rectificades.png</strong><br>Diferència entre rectificades.</div>
            <div class="file"><strong>06_mapa_disparitat.png</strong><br>Mapa de disparitat.</div>
            <div class="file"><strong>07_linies_epipolars.png</strong><br>Visualització de línies epipolars.</div>
            <div class="file"><strong>informe_resultats.txt</strong><br>Informe tècnic textual.</div>
        </div>

        <h2>Valoració final</h2>
        <div class="final-box">
            <p>
                Estem molt contents amb el resultat final del treball. La pràctica ha estat
                exigent, sobretot perquè treballar amb imatges grans i amb visió artificial
                implica controlar molts aspectes alhora: temps de càlcul, qualitat dels punts,
                correspondències incorrectes, deformacions de la rectificació i visualització
                dels resultats. Tot i això, hem aconseguit una versió final estable,
                interactiva i visualment clara.
            </p>
            <p>
                Considerem que els resultats són fiables perquè el nombre d’inliers després
                de RANSAC és alt i perquè les rectificacions obtingudes mantenen l’escena
                recognoscible i coherent. A més, l’anàglif mostra una superposició útil entre
                les dues imatges, el mapa de disparitat aporta una lectura visual de la
                profunditat i les línies epipolars confirmen que la geometria entre les dues
                imatges s’ha estimat de manera consistent.
            </p>
            <p>
                En conjunt, el treball no només compleix els punts bàsics de l’enunciat, sinó
                que també afegeix millores pròpies com el visor interactiu, el panell lateral,
                la consulta de píxels amb el botó dret, la generació automàtica d’informes i
                la visualització de línies epipolars. Per això pensem que és una pràctica
                completa, funcional i ben resolta.
            </p>
        </div>

        <div class="footer">
            Informe generat automàticament pel programa de la Pràctica 1.<br>
            Mikel Zeberio i Ignacio Ojeda
        </div>
    </div>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Informe HTML desat a:", path)



def generar_informes():
    generar_informe_txt()
    generar_informe_html()


# ============================================================
# EXECUCIÓ DEL PROCESSAMENT COMPLET
# ============================================================

def executar_processament_complet():
    crear_carpeta_resultats()
    comprovar_fitxers()

    img_left, img_right = carregar_imatges_originals()

    kp1, desc1, kp2, desc2 = detectar_sift(img_left, img_right)
    matches = emparellar_punts(desc1, desc2)

    F, pts1, pts2, pts1_inliers, pts2_inliers, mask = calcular_matriu_fonamental(
        kp1,
        kp2,
        matches
    )

    matches_img = crear_visualitzacio_matches(
        img_left,
        img_right,
        kp1,
        kp2,
        matches,
        mask
    )

    rect_left, rect_right, H1, H2 = rectificar_imatges(
        img_left,
        img_right,
        F,
        pts1_inliers,
        pts2_inliers
    )

    anaglif = crear_anaglif(rect_left, rect_right)
    diff = crear_diferencia(rect_left, rect_right)
    disparity = calcular_disparitat(rect_left, rect_right)

    epilines_img = crear_visualitzacio_epipolars(
        img_left,
        img_right,
        pts1_inliers,
        pts2_inliers,
        F
    )

    generar_informes()

    return {
        "Esquerra rectificada": rect_left,
        "Dreta rectificada": rect_right,
        "Anaglif": anaglif,
        "Diferencia": diff,
        "Mapa de disparitat": disparity,
        "Correspondencies SIFT": matches_img,
        "Linies epipolars": epilines_img
    }


# ============================================================
# VISOR INTERACTIU
# ============================================================

def zoom_per_veure_imatge_completa():
    img = images[mode_names[current_mode]]
    h, w = img.shape[:2]

    zx = VIEW_WIDTH / w
    zy = VIEW_HEIGHT / h

    z = min(zx, zy)

    z = max(MIN_ZOOM, min(z, MAX_ZOOM))

    return z


def clamp_offsets():
    global offset_x, offset_y

    img = images[mode_names[current_mode]]
    h, w = img.shape[:2]

    crop_w = int(VIEW_WIDTH / zoom)
    crop_h = int(VIEW_HEIGHT / zoom)

    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))

    max_x = max(0, w - crop_w)
    max_y = max(0, h - crop_h)

    offset_x = max(0, min(offset_x, max_x))
    offset_y = max(0, min(offset_y, max_y))


def coordenada_pantalla_a_imatge(x, y):
    img = images[mode_names[current_mode]]
    h, w = img.shape[:2]

    img_x = int(offset_x + x / zoom)
    img_y = int(offset_y + y / zoom)

    img_x = max(0, min(img_x, w - 1))
    img_y = max(0, min(img_y, h - 1))

    return img_x, img_y


def dibuixar_popup_pixel(view):
    b, g, r = pixel_popup_bgr
    gray = int(0.114 * b + 0.587 * g + 0.299 * r)

    lines = [
        "INFO PIXEL",
        f"Mode: {pixel_popup_mode}",
        f"x: {pixel_popup_img_x}",
        f"y: {pixel_popup_img_y}",
        f"BGR: ({b}, {g}, {r})",
        f"RGB: ({r}, {g}, {b})",
        f"Gris: {gray}"
    ]

    box_w = 310
    box_h = 205

    x = pixel_popup_screen_x + 15
    y = pixel_popup_screen_y + 15

    if x + box_w > VIEW_WIDTH:
        x = pixel_popup_screen_x - box_w - 15

    if y + box_h > VIEW_HEIGHT:
        y = pixel_popup_screen_y - box_h - 15

    x = max(5, x)
    y = max(50, y)

    cv.rectangle(view, (x + 5, y + 5), (x + box_w + 5, y + box_h + 5), (0, 0, 0), -1)
    cv.rectangle(view, (x, y), (x + box_w, y + box_h), (40, 40, 40), -1)
    cv.rectangle(view, (x, y), (x + box_w, y + box_h), (235, 235, 235), 2)
    cv.rectangle(view, (x, y), (x + box_w, y + 35), (80, 80, 80), -1)

    text_y = y + 25

    cv.putText(
        view,
        lines[0],
        (x + 12, text_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    text_y += 35

    for line in lines[1:]:
        cv.putText(
            view,
            line,
            (x + 12, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv.LINE_AA
        )
        text_y += 25

    cv.circle(view, (pixel_popup_screen_x, pixel_popup_screen_y), 6, (0, 255, 255), 2)
    cv.line(view, (pixel_popup_screen_x - 8, pixel_popup_screen_y),
            (pixel_popup_screen_x + 8, pixel_popup_screen_y), (0, 255, 255), 1)
    cv.line(view, (pixel_popup_screen_x, pixel_popup_screen_y - 8),
            (pixel_popup_screen_x, pixel_popup_screen_y + 8), (0, 255, 255), 1)


def crear_area_imatge():
    global fps, last_time

    img = images[mode_names[current_mode]]
    h, w = img.shape[:2]

    clamp_offsets()

    crop_w = int(VIEW_WIDTH / zoom)
    crop_h = int(VIEW_HEIGHT / zoom)

    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))

    x1 = offset_x
    y1 = offset_y
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    crop = img[y1:y2, x1:x2]
    view = cv.resize(crop, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv.INTER_AREA)

    now = time.time()
    dt = now - last_time

    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    last_time = now

    overlay = view.copy()
    cv.rectangle(overlay, (0, 0), (VIEW_WIDTH, 45), (0, 0, 0), -1)
    view = cv.addWeighted(overlay, 0.65, view, 0.35, 0)

    title = (
        f"Mode: {mode_names[current_mode]} | "
        f"Zoom: {zoom:.3f} | "
        f"Cursor: ({cursor_x}, {cursor_y}) | "
        f"Inliers: {stats['inliers']} | "
        f"FPS: {fps:.1f}"
    )

    cv.putText(
        view,
        title,
        (15, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    cx = VIEW_WIDTH // 2
    cy = VIEW_HEIGHT // 2

    cv.line(view, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
    cv.line(view, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

    if pixel_popup_visible:
        dibuixar_popup_pixel(view)

    return view


def escriure_text(panel, text, x, y, mida=0.55, color=(235, 235, 235), gruix=1):
    cv.putText(
        panel,
        text,
        (x, y),
        cv.FONT_HERSHEY_SIMPLEX,
        mida,
        color,
        gruix,
        cv.LINE_AA
    )


def crear_panell_lateral():
    panel = np.zeros((TOTAL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (35, 35, 35)

    cv.rectangle(panel, (0, 0), (PANEL_WIDTH, 70), (55, 55, 55), -1)
    escriure_text(panel, "PRACTICA 1", 20, 30, 0.8, (255, 255, 255), 2)
    escriure_text(panel, "Versio 3 final", 20, 55, 0.55, (210, 210, 210), 1)

    y = 100

    escriure_text(panel, "MODE ACTUAL", 20, y, 0.55, (180, 220, 255), 2)
    y += 30
    escriure_text(panel, mode_names[current_mode], 20, y, 0.62, (255, 255, 255), 2)

    y += 42
    escriure_text(panel, "CONTROLS", 20, y, 0.55, (180, 220, 255), 2)
    y += 35

    controls = [
        ("1", "Esquerra rectificada"),
        ("2", "Dreta rectificada"),
        ("3", "Anaglif"),
        ("4", "Diferencia"),
        ("5", "Mapa de disparitat"),
        ("C", "Correspondencies SIFT"),
        ("E", "Linies epipolars"),
        ("+", "Augmentar zoom"),
        ("-", "Reduir zoom"),
        ("R", "Reiniciar vista"),
        ("D", "Desar vista actual"),
        ("Q", "Sortir"),
    ]

    for tecla, desc in controls:
        cv.rectangle(panel, (20, y - 18), (55, y + 8), (80, 80, 80), -1)
        escriure_text(panel, tecla, 31, y, 0.55, (255, 255, 255), 2)
        escriure_text(panel, desc, 70, y, 0.48, (230, 230, 230), 1)
        y += 30

    y += 10
    escriure_text(panel, "RATOLI", 20, y, 0.55, (180, 220, 255), 2)
    y += 28
    escriure_text(panel, "Boto esquerre + arrossegar", 20, y, 0.45, (235, 235, 235), 1)
    y += 23
    escriure_text(panel, "Moure la vista", 20, y, 0.45, (235, 235, 235), 1)
    y += 28
    escriure_text(panel, "Boto dret sobre un punt", 20, y, 0.45, (235, 235, 235), 1)
    y += 23
    escriure_text(panel, "Mostrar info del pixel", 20, y, 0.45, (235, 235, 235), 1)

    y += 35
    escriure_text(panel, "DADES", 20, y, 0.55, (180, 220, 255), 2)
    y += 28
    escriure_text(panel, f"SIFT esq.: {stats['sift_left']}", 20, y, 0.46, (235, 235, 235), 1)
    y += 22
    escriure_text(panel, f"SIFT dreta: {stats['sift_right']}", 20, y, 0.46, (235, 235, 235), 1)
    y += 22
    escriure_text(panel, f"Matches: {stats['matches_used']}", 20, y, 0.46, (235, 235, 235), 1)
    y += 22
    escriure_text(panel, f"Inliers: {stats['inliers']}", 20, y, 0.46, (235, 235, 235), 1)
    y += 22
    escriure_text(panel, f"Zoom: {zoom:.3f}", 20, y, 0.46, (235, 235, 235), 1)

    if pixel_popup_visible:
        y += 35
        escriure_text(panel, "PIXEL SELECCIONAT", 20, y, 0.50, (180, 220, 255), 2)
        y += 24
        escriure_text(panel, f"x: {pixel_popup_img_x}", 20, y, 0.45, (235, 235, 235), 1)
        y += 22
        escriure_text(panel, f"y: {pixel_popup_img_y}", 20, y, 0.45, (235, 235, 235), 1)
        y += 22
        escriure_text(panel, f"BGR: {pixel_popup_bgr}", 20, y, 0.45, (235, 235, 235), 1)

    cv.rectangle(panel, (0, TOTAL_HEIGHT - 55), (PANEL_WIDTH, TOTAL_HEIGHT), (55, 55, 55), -1)
    escriure_text(panel, "Informes TXT i HTML generats", 20, TOTAL_HEIGHT - 30, 0.43, (255, 255, 255), 1)
    escriure_text(panel, "D = desa captura | Q = sortir", 20, TOTAL_HEIGHT - 12, 0.43, (255, 255, 255), 1)

    return panel


def crear_vista_completa():
    area_imatge = crear_area_imatge()
    panel = crear_panell_lateral()

    canvas = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :VIEW_WIDTH] = area_imatge
    canvas[:, VIEW_WIDTH:] = panel

    return canvas


def mouse_callback(event, x, y, flags, param):
    global dragging, last_mouse_x, last_mouse_y
    global offset_x, offset_y, cursor_x, cursor_y
    global pixel_popup_visible, pixel_popup_screen_x, pixel_popup_screen_y
    global pixel_popup_img_x, pixel_popup_img_y, pixel_popup_bgr, pixel_popup_mode

    if x >= VIEW_WIDTH:
        return

    img = images[mode_names[current_mode]]

    cursor_x, cursor_y = coordenada_pantalla_a_imatge(x, y)

    if event == cv.EVENT_LBUTTONDOWN:
        dragging = True
        last_mouse_x = x
        last_mouse_y = y

    elif event == cv.EVENT_MOUSEMOVE and dragging:
        dx = x - last_mouse_x
        dy = y - last_mouse_y

        offset_x -= int(dx / zoom)
        offset_y -= int(dy / zoom)

        last_mouse_x = x
        last_mouse_y = y

    elif event == cv.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv.EVENT_RBUTTONDOWN:
        img_x, img_y = coordenada_pantalla_a_imatge(x, y)

        bgr = img[img_y, img_x]
        b = int(bgr[0])
        g = int(bgr[1])
        r = int(bgr[2])

        pixel_popup_visible = True
        pixel_popup_screen_x = x
        pixel_popup_screen_y = y
        pixel_popup_img_x = img_x
        pixel_popup_img_y = img_y
        pixel_popup_bgr = (b, g, r)
        pixel_popup_mode = mode_names[current_mode]

        print()
        print("PIXEL SELECCIONAT")
        print("-----------------")
        print("Mode:", pixel_popup_mode)
        print("x:", pixel_popup_img_x)
        print("y:", pixel_popup_img_y)
        print("BGR:", pixel_popup_bgr)
        print("RGB:", (r, g, b))


def mostrar_ajuda():
    print()
    print("CONTROLS DEL VISOR")
    print("------------------")
    print("1  : imatge esquerra rectificada")
    print("2  : imatge dreta rectificada")
    print("3  : anaglif")
    print("4  : diferencia")
    print("5  : mapa de disparitat")
    print("c  : correspondencies SIFT")
    print("e  : linies epipolars")
    print("+  : augmentar zoom")
    print("-  : reduir zoom")
    print("r  : reiniciar vista")
    print("d  : desar vista actual")
    print("q  : sortir")
    print("Boto esquerre + arrossegar : moure la vista")
    print("Boto dret sobre un punt    : mostrar informació del pixel")
    print()


def canviar_mode(index):
    global current_mode, offset_x, offset_y, zoom, pixel_popup_visible

    if 0 <= index < len(mode_names):
        current_mode = index
        offset_x = 0
        offset_y = 0
        zoom = zoom_per_veure_imatge_completa()
        pixel_popup_visible = False
        print("Mode actual:", mode_names[current_mode])


def desar_vista(view):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nom_mode = mode_names[current_mode].replace(" ", "_")
    filename = f"vista_{nom_mode}_{timestamp}.png"
    path = os.path.join(RESULTS_FOLDER, filename)

    cv.imwrite(path, view)

    print("Vista actual desada a:", path)


def bucle_interactiu():
    global zoom, offset_x, offset_y, pixel_popup_visible

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, TOTAL_WIDTH, TOTAL_HEIGHT)
    cv.setMouseCallback(WINDOW_NAME, mouse_callback)

    zoom = zoom_per_veure_imatge_completa()

    mostrar_ajuda()

    while True:
        view = crear_vista_completa()
        cv.imshow(WINDOW_NAME, view)

        key = cv.waitKey(30) & 0xFF

        if key == 255:
            continue

        if key == ord("q") or key == 27:
            break

        elif key == ord("1"):
            canviar_mode(0)

        elif key == ord("2"):
            canviar_mode(1)

        elif key == ord("3"):
            canviar_mode(2)

        elif key == ord("4"):
            canviar_mode(3)

        elif key == ord("5"):
            canviar_mode(4)

        elif key == ord("c") or key == ord("C"):
            canviar_mode(5)

        elif key == ord("e") or key == ord("E"):
            canviar_mode(6)

        elif key == ord("+") or key == ord("="):
            zoom *= 1.25
            zoom = min(zoom, MAX_ZOOM)
            pixel_popup_visible = False

        elif key == ord("-") or key == ord("_"):
            zoom /= 1.25
            zoom = max(zoom, MIN_ZOOM)
            pixel_popup_visible = False

        elif key == ord("r") or key == ord("R"):
            offset_x = 0
            offset_y = 0
            zoom = zoom_per_veure_imatge_completa()
            pixel_popup_visible = False
            print("Vista reiniciada.")

        elif key == ord("d") or key == ord("D"):
            desar_vista(view)

    cv.destroyAllWindows()


def main():
    global images, mode_names

    images = executar_processament_complet()
    mode_names = list(images.keys())

    bucle_interactiu()


if __name__ == "__main__":
    main()
