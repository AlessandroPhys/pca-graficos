import io
import zipfile
import urllib.request
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="PCA | Communities and Crime",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================================================
# PARÁMETROS DEL ANÁLISIS (equivalen a las celdas 5 y 6 del notebook)
# ==========================================================================
RANDOM_STATE = 42
N_COMPONENTES = 7                  # componentes que se muestran (PC1–PC7)
TOP_N_DEFECTO = 15                 # variables por gráfico de cargas / biplot
MAX_FALTANTES = 0.50               # se eliminan las variables con más de 50 % de faltantes
COLUMNAS_IDENTIFICADORAS = ["State"]          # identificadores o metadatos
COLUMNAS_REDUNDANTES = ["kidsBornNevrMarr"]   # redundantes / no deseadas
# Variables de criminalidad (targets) que se suman a la matriz X. Como solo se hace un PCA (no hay
# variable objetivo), se incorporan como una variable más y pasan por los mismos filtros.
#   None -> todas las variables de criminalidad del dataset (18)
#   o una lista, por ejemplo ["violentPerPop", "nonViolPerPop"]
TARGETS_INCLUIDOS = None

# ==========================================================================
# GUÍA DE COMPONENTES
# Edite aquí el texto que se muestra en el panel de la derecha.
#   "titulo":  nombre corto del componente (opcional)
#   "detalle": descripción de qué representa (admite saltos de línea con \n)
# Si se deja vacío, el panel muestra "Descripción por completar."
# ==========================================================================
DESCRIPCIONES = {
    "PC1": {"titulo": "", "detalle": ""},
    "PC2": {"titulo": "", "detalle": ""},
    "PC3": {"titulo": "", "detalle": ""},
    "PC4": {"titulo": "", "detalle": ""},
    "PC5": {"titulo": "", "detalle": ""},
    "PC6": {"titulo": "", "detalle": ""},
    "PC7": {"titulo": "", "detalle": ""},
}

# Nombres reales de las 143 variables del dataset (125 explicativas + 18 de criminalidad),
# en el orden de ucimlrepo. Se usan solo si hay que descargar el ZIP de UCI directamente.
NOMBRES_UCI = [
    "State", "pop", "perHoush", "pctBlack", "pctWhite", "pctAsian", "pctHisp", "pct12-21",
    "pct12-29", "pct16-24", "pct65up", "persUrban", "pctUrban", "medIncome", "pctWwage",
    "pctWfarm", "pctWdiv", "pctWsocsec", "pctPubAsst", "pctRetire", "medFamIncome", "perCapInc",
    "whitePerCap", "blackPerCap", "NAperCap", "asianPerCap", "otherPerCap", "hispPerCap",
    "persPoverty", "pctPoverty", "pctLowEdu", "pctNotHSgrad", "pctCollGrad", "pctUnemploy",
    "pctEmploy", "pctEmployMfg", "pctEmployProfServ", "pctOccupManu", "pctOccupMgmt",
    "pctMaleDivorc", "pctMaleNevMar", "pctFemDivorc", "pctAllDivorc", "persPerFam", "pct2Par",
    "pctKids2Par", "pctKids-4w2Par", "pct12-17w2Par", "pctWorkMom-6", "pctWorkMom-18",
    "kidsBornNevrMarr", "pctKidsBornNevrMarr", "numForeignBorn", "pctFgnImmig-3", "pctFgnImmig-5",
    "pctFgnImmig-8", "pctFgnImmig-10", "pctImmig-3", "pctImmig-5", "pctImmig-8", "pctImmig-10",
    "pctSpeakOnlyEng", "pctNotSpeakEng", "pctLargHousFam", "pctLargHous", "persPerOccupHous",
    "persPerOwnOccup", "persPerRenterOccup", "pctPersOwnOccup", "pctPopDenseHous",
    "pctSmallHousUnits", "medNumBedrm", "houseVacant", "pctHousOccup", "pctHousOwnerOccup",
    "pctVacantBoarded", "pctVacant6up", "medYrHousBuilt", "pctHousWOphone", "pctHousWOplumb",
    "ownHousLowQ", "ownHousMed", "ownHousUperQ", "ownHousQrange", "rentLowQ", "rentMed",
    "rentUpperQ", "rentQrange", "medGrossRent", "medRentpctHousInc", "medOwnCostpct",
    "medOwnCostPctWO", "persEmergShelt", "persHomeless", "pctForeignBorn", "pctBornStateResid",
    "pctSameHouse-5", "pctSameCounty-5", "pctSameState-5", "numPolice", "policePerPop",
    "policeField", "policeFieldPerPop", "policeCalls", "policCallPerPop", "policCallPerOffic",
    "policePerPop2", "racialMatch", "pctPolicWhite", "pctPolicBlack", "pctPolicHisp",
    "pctPolicAsian", "pctPolicMinority", "officDrugUnits", "numDiffDrugsSeiz", "policAveOT",
    "landArea", "popDensity", "pctUsePubTrans", "policCarsAvail", "policOperBudget",
    "pctPolicPatrol", "gangUnit", "pctOfficDrugUnit", "policBudgetPerPop", "murders", "murdPerPop",
    "rapes", "rapesPerPop", "robberies", "robbbPerPop", "assaults", "assaultPerPop", "burglaries",
    "burglPerPop", "larcenies", "larcPerPop", "autoTheft", "autoTheftPerPop", "arsons",
    "arsonsPerPop", "violentPerPop", "nonViolPerPop",
]
N_EXPLICATIVAS = 125   # State + 124 variables numéricas
N_TARGETS = 18

# --------------------------------------------------------------------------
# Paleta de alto contraste (pensada para proyectar sobre una pared)
# --------------------------------------------------------------------------
INK = "#0B1220"        # texto, ejes y líneas principales (casi negro)
BLUE = "#1D4ED8"       # color principal / cargas positivas
ORANGE = "#D9480F"     # cargas negativas y resaltado
GRID = "#CBD5E1"       # cuadrícula (visible pero secundaria)
PAPER = "#F1F5F9"      # fondo de la página
WHITE = "#FFFFFF"      # fondo de tarjetas y gráficos
BODY = "#1E293B"       # texto de párrafos
FONT = "Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background: %PAPER%;
    color: %INK%;
    font-family: 'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 18px;
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }

/* Contenedor central adaptable al monitor */
.block-container {
    max-width: min(1500px, 96vw);
    margin: 0 auto;
    padding: 1.4rem 1rem 3rem 1rem;
}
@media (min-width: 1700px) {
    html, body, [data-testid="stApp"] { font-size: 20px; }
    .block-container { max-width: min(1700px, 92vw); }
}
@media (min-width: 2300px) {
    html, body, [data-testid="stApp"] { font-size: 24px; }
    .block-container { max-width: min(2100px, 88vw); }
}
@media (max-width: 700px) { .block-container { max-width: 98vw; } }

/* Título y encabezados (serif, estilo artículo científico) */
h1 {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    color: %INK%; text-align: center; font-weight: 700;
    font-size: 2.2rem; padding: 0 0 0.2rem 0; letter-spacing: -0.01em;
}
.subtitle {
    text-align: center; color: %BODY%; font-size: 1.05rem; margin: 0 0 1rem 0;
    font-family: 'Source Serif 4', Georgia, serif; font-style: italic;
}
.section-h {
    font-family: 'Source Serif 4', Georgia, serif;
    text-align: center; font-weight: 700; font-size: 1.55rem; color: %INK%;
    margin: 1.2rem 0 0.4rem 0; padding-bottom: 0.4rem; border-bottom: 2px solid %INK%;
}

/* Fichas de datos */
.stats { display: flex; flex-wrap: wrap; justify-content: center; gap: 0.8rem; margin-bottom: 0.4rem; }
.stat { background: %WHITE%; border: 2px solid %INK%; border-radius: 6px; padding: 0.5rem 1.2rem; text-align: center; min-width: 11rem; }
.stat-v { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.7rem; font-weight: 700; color: %INK%; line-height: 1.2; }
.stat-l { font-size: 0.92rem; color: %BODY%; font-weight: 600; }

/* Botones de selección (radio con aspecto de botones) */
[data-testid="stRadio"] [role="radiogroup"] { justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: %WHITE%; border: 2px solid %INK%; border-radius: 8px;
    padding: 0.3rem 1.1rem; margin: 0; cursor: pointer;
}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stRadio"] label[data-baseweb="radio"] p {
    font-weight: 600; font-size: 1.05rem; color: %INK%; margin: 0;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: %BLUE%; border-color: %BLUE%;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color: #FFFFFF; }

/* Interruptor de vectores alineado con los selectores */
[data-testid="stToggle"], [data-testid="stCheckbox"] { margin-top: 1.7rem; }
[data-testid="stToggle"] p, [data-testid="stCheckbox"] p { font-weight: 700; color: %INK%; }

/* Etiquetas y listas desplegables con buen contraste */
[data-testid="stWidgetLabel"] { justify-content: center; }
[data-testid="stWidgetLabel"] p { text-align: center; font-weight: 700; color: %INK%; font-size: 1rem; }
[data-baseweb="select"] > div { border: 2px solid %INK% !important; background: %WHITE%; font-size: 1.05rem; }
[data-testid="stSlider"] { padding: 0 0.5rem; }
[data-testid="stAlert"] { text-align: center; }

/* Gráfico: sin relleno extra para evitar barras de desplazamiento internas */
[data-testid="stPlotlyChart"] { padding: 0 !important; margin: 0.2rem 0 0 0; overflow: hidden !important; }
[data-testid="stPlotlyChart"] > div { overflow: hidden !important; }

/* Pie de figura */
.figcap {
    font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem; line-height: 1.5;
    color: %BODY%; margin: 0.4rem 0.2rem 0 0.2rem; text-align: justify;
}

/* Tablas */
.tabla-titulo { font-weight: 700; font-size: 1.05rem; color: %INK%; margin: 0.8rem 0 0.3rem 0; text-align: center; }
.tabla { width: 100%; border-collapse: collapse; background: %WHITE%; border: 2px solid %INK%; font-size: 1rem; }
.tabla th { background: %INK%; color: #FFFFFF; padding: 0.4rem 0.7rem; text-align: right; font-weight: 700; }
.tabla td { padding: 0.28rem 0.7rem; border-top: 1px solid %GRID%; text-align: right; color: %INK%; }
.tabla th:first-child, .tabla td:first-child { text-align: left; font-weight: 600; }

/* Panel estático: guía de componentes */
.guide { background: %WHITE%; border: 2px solid %INK%; border-radius: 6px; padding: 0.9rem 1rem 0.4rem 1rem; margin-top: 0.2rem; }
.guide-title {
    font-family: 'Source Serif 4', Georgia, serif; font-weight: 700; font-size: 1.25rem;
    color: %INK%; border-bottom: 2px solid %INK%; padding-bottom: 0.4rem; margin-bottom: 0.7rem;
}
.guide-item { border-left: 6px solid #94A3B8; padding: 0.15rem 0 0.15rem 0.7rem; margin-bottom: 0.8rem; }
.guide-item.active { border-left-color: %BLUE%; background: #EEF3FE; }
.guide-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; }
.guide-pc { font-weight: 700; font-size: 1.1rem; color: %INK%; }
.guide-item.active .guide-pc { color: %BLUE%; }
.guide-var { font-size: 0.88rem; color: %BODY%; font-weight: 600; }
.guide-name { font-weight: 700; font-size: 1rem; color: %INK%; }
.guide-text { font-size: 0.97rem; line-height: 1.4; color: %BODY%; }
.guide-text.empty { font-style: italic; color: #475569; }
</style>
"""
for _name, _value in {
    "%PAPER%": PAPER, "%INK%": INK, "%BLUE%": BLUE, "%WHITE%": WHITE, "%BODY%": BODY, "%GRID%": GRID,
}.items():
    CSS = CSS.replace(_name, _value)
st.markdown(CSS, unsafe_allow_html=True)

st.title("Communities and Crime — Análisis de componentes principales")
st.markdown(
    '<p class="subtitle">Análisis de componentes principales (PCA) del conjunto de datos '
    "Communities and Crime Unnormalized, UCI Machine Learning Repository (id. 211).</p>",
    unsafe_allow_html=True,
)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"


def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")


def html_table(df):
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f'<table class="tabla"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def style_axes(fig):
    """Ejes cartesianos bien definidos: caja, líneas oscuras, marcas y texto grandes."""
    axis = dict(
        showline=True, linewidth=2, linecolor=INK, mirror=True,
        ticks="outside", ticklen=8, tickwidth=2, tickcolor=INK,
        tickfont=dict(size=17, color=INK),
        title_font=dict(size=21, color=INK, family=FONT),
        showgrid=True, gridcolor=GRID, gridwidth=1,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def base_layout(fig, height, title, title_size=25, legend_size=17, top=105):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=FONT, size=17, color=INK),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=title_size, color=INK)),
        margin=dict(l=20, r=25, t=top, b=20),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
            font=dict(size=legend_size, color=INK), bgcolor="rgba(255,255,255,0)", itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(size=17, color="#FFFFFF", family=FONT)),
    )
    style_axes(fig)
    return fig


def show_chart(fig, filename):
    """Muestra el gráfico a todo el ancho (compatible con versiones nuevas y antiguas de Streamlit).
    El botón de cámara exporta un PNG de alta resolución (x3) apto para publicación."""
    config = {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 3},
    }
    try:
        st.plotly_chart(fig, width="stretch", config=config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config)


@st.cache_data(show_spinner="Descargando dataset desde UCI...")
def load_data():
    """Carga el dataset de UCI.

    Devuelve (variables explicativas, variables de criminalidad (targets), nombres de comunidades).
    """
    api_error = None
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=211)
        features = dataset.data.features
        targets = dataset.data.targets
        if features is not None and not features.empty:
            features = features.copy().reset_index(drop=True)
            targets = (
                targets.copy().reset_index(drop=True)
                if targets is not None else pd.DataFrame(index=features.index)
            )
            ids = dataset.data.ids
            names = None
            if ids is not None and not ids.empty:
                name_col = next((c for c in ids.columns if "name" in c.lower()), ids.columns[0])
                names = ids[name_col].astype(str).str.strip()
            elif "communityname" in features.columns:
                names = features["communityname"].astype(str).str.strip()
            if names is not None:
                # Añade el estado cuando está disponible para distinguir comunidades homónimas.
                state = None
                if ids is not None and "State" in ids.columns:
                    state = ids["State"]
                elif "State" in features.columns:
                    state = features["State"]
                if state is not None:
                    names = names + " (" + state.astype(str).str.strip() + ")"
                names = names.reset_index(drop=True)
            else:
                names = pd.Series([f"Comunidad {i + 1}" for i in range(len(features))])
            return features, targets, names
    except Exception as exc:
        api_error = str(exc)

    # Respaldo: descarga directa del ZIP original de UCI.
    request = urllib.request.Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 PCA-visualization-app"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        data_file = next(
            (n for n in archive.namelist() if n.lower().endswith("commviolpredunnormalizeddata.txt")),
            None,
        )
        if data_file is None:
            raise RuntimeError("El archivo de datos no se encontró dentro del ZIP de UCI.")
        with archive.open(data_file) as file:
            raw = pd.read_csv(
                file,
                header=None,
                na_values=["?", " ?", ""],
                skipinitialspace=True,
            )

    # Estructura del archivo (147 columnas): 0 nombre de la comunidad, 1 estado, 2 código de condado,
    # 3 código de comunidad, 4 partición "fold", 5–128 variables numéricas y 129–146 criminalidad.
    if raw.shape[1] != 147:
        raise RuntimeError(f"El archivo descargado tiene un formato inesperado ({raw.shape[1]} columnas).")
    numeric = raw.iloc[:, 5:5 + (N_EXPLICATIVAS - 1)].copy()
    numeric.columns = NOMBRES_UCI[1:N_EXPLICATIVAS]
    features = pd.concat([raw.iloc[:, 1].astype(str).str.strip().rename("State"), numeric], axis=1)
    targets = raw.iloc[:, -N_TARGETS:].copy()
    targets.columns = NOMBRES_UCI[N_EXPLICATIVAS:]
    names = (raw.iloc[:, 0].astype(str).str.strip() + " (" + raw.iloc[:, 1].astype(str).str.strip() + ")")
    if api_error:
        st.info("Se utilizó la descarga directa de UCI porque la API no respondió correctamente.")
    return features.reset_index(drop=True), targets.reset_index(drop=True), names.reset_index(drop=True)


try:
    features, targets_all, community_names = load_data()

    # ==================== CELDA 5 DEL NOTEBOOK: preparar la matriz ====================
    # Variables explicativas + variables de criminalidad (targets): una sola matriz X para el PCA.
    if TARGETS_INCLUIDOS is None:
        targets_sel = targets_all
    else:
        targets_sel = targets_all[[c for c in TARGETS_INCLUIDOS if c in targets_all.columns]]
    target_set = set(targets_sel.columns)
    X = pd.concat([features, targets_sel], axis=1)
    n_inicial = X.shape[1]

    # 1. Identificadores o metadatos.
    elim_identificadores = [c for c in COLUMNAS_IDENTIFICADORAS if c in X.columns]
    X = X.drop(columns=elim_identificadores)
    # Las columnas que queden sin tipo numérico se convierten (lo no convertible pasa a faltante).
    for _c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[_c]):
            X[_c] = pd.to_numeric(X[_c], errors="coerce")

    # 2. Variables con más de 50 % de faltantes.
    porcentaje_na = X.isna().mean()
    elim_por_na = porcentaje_na[porcentaje_na > MAX_FALTANTES]
    X = X.drop(columns=elim_por_na.index)

    # 3. Variables redundantes o no deseadas.
    elim_redundantes = [c for c in COLUMNAS_REDUNDANTES if c in X.columns]
    X = X.drop(columns=elim_redundantes)
    if X.empty:
        raise RuntimeError("No se encontraron variables numéricas utilizables.")

    # 4. Imputación con la mediana y estandarización.
    total_faltantes_imputados = int(X.isna().sum().sum())
    n_targets_kept = sum(1 for c in X.columns if c in target_set)
    X_imputado = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index,
    )
    Z = StandardScaler().fit_transform(X_imputado)

    descartadas = (
        [(v, "Identificador", "—") for v in elim_identificadores]
        + [(v, "Redundante", "—") for v in elim_redundantes]
        + [
            (v, "Criminalidad (target)" if v in target_set else "Explicativa", f"{p * 100:.1f} %")
            for v, p in elim_por_na.sort_values(ascending=False).items()
        ]
    )
    descartadas_df = pd.DataFrame(descartadas, columns=["Variable", "Tipo", "% faltante"])
    descartadas_df.insert(2, "Motivo", [
        "Identificador o metadato" if t == "Identificador"
        else "Redundante o no deseada" if t == "Redundante"
        else f"Más de {MAX_FALTANTES * 100:.0f} % de faltantes"
        for t in descartadas_df["Tipo"]
    ])

    # ==================== CELDA 6 DEL NOTEBOOK: PCA ====================
    pca = PCA(n_components=None, random_state=RANDOM_STATE)
    scores_pca = pca.fit_transform(Z)
    varianza = pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)
    n_total = len(varianza)
    k = min(N_COMPONENTES, n_total)
    components = [f"PC{i + 1}" for i in range(k)]
    var_of = {f"PC{i + 1}": 100 * v for i, v in enumerate(varianza)}

    scores_df = pd.DataFrame(scores_pca[:, :k], columns=components, index=X.index)
    scores_df["Comunidad"] = community_names.reindex(X.index).fillna("Sin nombre").values
    # Cargas = autovectores x raíz cuadrada de los autovalores (correlación variable-componente)
    cargas = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(cargas[:, :k], index=X.columns, columns=components)

    def render_guide(active):
        """Panel estático con la descripción de cada componente (se edita en DESCRIPCIONES)."""
        rows = []
        for pc in components:
            info = DESCRIPCIONES.get(pc, {})
            titulo = escape(str(info.get("titulo", "")).strip())
            detalle = escape(str(info.get("detalle", "")).strip()).replace("\n", "<br>")
            cls = "guide-item active" if pc in active else "guide-item"
            name_html = f'<div class="guide-name">{titulo}</div>' if titulo else ""
            text_html = (
                f'<div class="guide-text">{detalle}</div>' if detalle
                else '<div class="guide-text empty">Descripción por completar.</div>'
            )
            rows.append(
                f'<div class="{cls}"><div class="guide-head"><span class="guide-pc">{pc}</span>'
                f'<span class="guide-var">{var_of[pc]:.1f} % de la varianza</span></div>'
                f"{name_html}{text_html}</div>"
            )
        st.markdown(
            '<div class="guide"><div class="guide-title">Guía de componentes</div>'
            + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    def etiqueta(v):
        return ("★ " + v) if v in target_set else v

    # ------------------------------ Resumen del análisis ------------------------------
    stats = [
        (fmt_int(X.shape[0]), "Observaciones (n)"),
        (fmt_int(X.shape[1]), "Variables analizadas (p)"),
        (f"{100 * acumulada[k - 1]:.1f} %", f"Varianza acumulada PC1–PC{k}"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>' for v, l in stats)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------ 1. Varianza explicada ------------------------------
    _, col_head, col_cfg = st.columns([1, 6, 1])
    col_head.markdown(
        '<div class="section-h">1. Varianza explicada por los componentes</div>',
        unsafe_allow_html=True,
    )
    with col_cfg.popover("⚙ Tamaño"):
        chart_height = st.slider(
            "Alto de los gráficos (px)", min_value=420, max_value=1000, value=640, step=20,
            help="Ajuste según la pantalla o el proyector.",
        )

    m = min(50, n_total)
    xs = np.arange(1, m + 1)
    half_h = max(400, int(chart_height * 0.85))
    sc1, sc2 = st.columns(2, gap="medium")

    with sc1:
        kc = min(N_COMPONENTES, m)
        yk = 100 * varianza[kc - 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=100 * varianza[:m], mode="lines+markers", name="Varianza explicada",
            line=dict(color=INK, width=3), marker=dict(size=6, color=INK),
            hovertemplate="PC%{x}: %{y:.2f} %<extra></extra>",
        ))
        fig.add_vline(x=kc, line=dict(color=ORANGE, width=2.5, dash="dot"))
        fig.add_hline(y=yk, line=dict(color=ORANGE, width=2.5, dash="dash"))
        fig.add_trace(go.Scatter(
            x=[kc], y=[yk], mode="markers", name=f"Corte en PC{kc} ({yk:.2f} %)",
            marker=dict(size=13, color=ORANGE, line=dict(width=2, color=INK)),
            hovertemplate=f"PC{kc}: {yk:.2f} %<extra></extra>",
        ))
        base_layout(fig, half_h, "Gráfico de sedimentación", title_size=21, legend_size=15, top=110)
        fig.update_xaxes(title_text="<b>Componente principal</b>", range=[1, m])
        fig.update_yaxes(title_text="<b>Varianza explicada (%)</b>", range=[0, float(100 * varianza[0]) * 1.08])
        show_chart(fig, "sedimentacion")

    with sc2:
        fig = go.Figure()
        for umbral, color in ((70, "#B91C1C"), (80, "#B45309"), (90, "#15803D"), (95, BLUE)):
            fig.add_trace(go.Scatter(
                x=[1, m], y=[umbral, umbral], mode="lines", name=f"{umbral} %",
                line=dict(color=color, width=2.5, dash="dash"), hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=xs, y=100 * acumulada[:m], mode="lines+markers", name="Varianza acumulada",
            line=dict(color=INK, width=3), marker=dict(size=6, color=INK),
            hovertemplate="Hasta PC%{x}: %{y:.2f} %<extra></extra>",
        ))
        base_layout(fig, half_h, "Varianza explicada acumulada", title_size=21, legend_size=15, top=110)
        fig.update_xaxes(title_text="<b>Número de componentes</b>", range=[1, m])
        fig.update_yaxes(title_text="<b>Varianza acumulada (%)</b>", range=[0, 102])
        show_chart(fig, "varianza_acumulada")

    st.markdown(
        f'<p class="figcap"><b>Figuras 1 y 2.</b> Gráfico de sedimentación (izquierda) y varianza explicada acumulada '
        f"(derecha) de los primeros {m} componentes, obtenidos con las {fmt_int(X.shape[1])} variables estandarizadas. "
        f"La marca naranja señala el corte en PC{min(N_COMPONENTES, m)}; las líneas discontinuas de la derecha "
        f"indican los umbrales de 70, 80, 90 y 95 % de la varianza total.</p>",
        unsafe_allow_html=True,
    )

    tabla_pca = pd.DataFrame({
        "Componente": [f"PC{i + 1}" for i in range(n_total)],
        "Varianza explicada (%)": [f"{100 * v:.2f}" for v in varianza],
        "Varianza acumulada (%)": [f"{100 * v:.2f}" for v in acumulada],
    }).head(15)
    umbrales_df = pd.DataFrame({
        "Umbral de varianza": [f"{u:.0%}" for u in (0.70, 0.80, 0.90, 0.95)],
        "Componentes necesarias": [int(np.searchsorted(acumulada, u) + 1) for u in (0.70, 0.80, 0.90, 0.95)],
    })
    t1, t2 = st.columns([1.5, 1], gap="medium")
    t1.markdown('<div class="tabla-titulo">Varianza por componente (primeras 15)</div>' + html_table(tabla_pca),
                unsafe_allow_html=True)
    t2.markdown('<div class="tabla-titulo">Componentes necesarias para alcanzar cada umbral</div>'
                + html_table(umbrales_df), unsafe_allow_html=True)

    # ------------------------------ 2. Proyección PCA (con vectores opcionales) ------------------------------
    st.markdown(
        '<div class="section-h" style="margin-top:2.4rem;">2. Proyección de las comunidades y vectores de las variables</div>',
        unsafe_allow_html=True,
    )
    MODO_EQUIL = "Equilibrado (mitad por eje)"
    MODO_X = "Según el eje X"
    MODO_Y = "Según el eje Y"
    MODO_PLANO = "Mejor representadas en el plano"
    max_vec = min(30, len(X.columns))

    col_x, col_y, col_find = st.columns([1, 1, 1.6])
    x_component = col_x.selectbox("Eje X", components, index=0)
    y_component = col_y.selectbox("Eje Y", components, index=1 if k > 1 else 0)
    highlighted = col_find.selectbox(
        "Resaltar una comunidad",
        sorted(scores_df["Comunidad"].unique()),
        index=None,
        placeholder="Escriba para buscar…",
    )

    vc1, vc2, vc3, vc4 = st.columns([1.1, 2, 1.8, 1.6])
    show_vec = vc1.toggle("Mostrar vectores", value=False, key="vec_on")
    if show_vec:
        modo = vc2.selectbox(
            "Variables según el componente",
            [MODO_EQUIL, MODO_X, MODO_Y, MODO_PLANO], index=0, key="vec_modo",
            help="Elige de qué componente se toman las variables que se dibujan como flechas.",
        )
        n_vec = vc3.slider(
            "Variables (flechas)", min_value=min(5, max_vec), max_value=max_vec,
            value=min(TOP_N_DEFECTO, max_vec), key="vec_n",
        )
        escala_flechas = vc4.slider(
            "Tamaño de las flechas (%)", min_value=50, max_value=200, value=100, step=10, key="vec_escala",
        ) / 100

    col_chart, col_guide = st.columns([3.2, 1.25], gap="medium")
    with col_chart:
        if x_component == y_component:
            st.warning("Seleccione componentes distintos para los ejes X e Y.")
        else:
            sx, sy = scores_df[x_component].to_numpy(), scores_df[y_component].to_numpy()
            fig = go.Figure()

            # Comunidades: más tenues cuando hay vectores, para que las flechas se vean bien.
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="markers", name="Comunidades",
                text=scores_df["Comunidad"],
                marker=dict(
                    size=6 if show_vec else 9, color=BLUE, opacity=0.22 if show_vec else 0.72,
                    line=dict(width=0 if show_vec else 0.8, color=WHITE),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{x_component}: %{{x:.2f}}<br>"
                    f"{y_component}: %{{y:.2f}}<extra></extra>"
                ),
            ))

            # Elipse de confianza del 95 % (aprox. normal bivariada): chi2(0.95, 2 g.l.) = -2 ln(0.05)
            ix, iy = components.index(x_component), components.index(y_component)
            radius = np.sqrt(-2 * np.log(0.05))
            t = np.linspace(0, 2 * np.pi, 240)
            fig.add_trace(go.Scatter(
                x=radius * np.sqrt(pca.explained_variance_[ix]) * np.cos(t),
                y=radius * np.sqrt(pca.explained_variance_[iy]) * np.sin(t),
                mode="lines", name="Elipse de confianza del 95 %",
                line=dict(color="#64748B" if show_vec else INK, width=2 if show_vec else 2.5, dash="dash"),
                hoverinfo="skip",
            ))

            if highlighted:
                sel = scores_df[scores_df["Comunidad"] == highlighted]
                fig.add_trace(go.Scatter(
                    x=sel[x_component], y=sel[y_component], mode="markers+text",
                    text=sel["Comunidad"], textposition="top center",
                    textfont=dict(size=19, color=INK, family=FONT),
                    marker=dict(size=19, color=ORANGE, symbol="diamond", line=dict(width=2.5, color=INK)),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"{x_component}: %{{x:.2f}}<br>"
                        f"{y_component}: %{{y:.2f}}<extra></extra>"
                    ),
                    showlegend=False,
                ))

            desc_modo = ""
            top_vars = []
            if show_vec:
                lx, ly = loadings_df[x_component], loadings_df[y_component]
                reach = np.hypot(lx, ly)
                if modo == MODO_X:
                    top_vars = list(lx.abs().sort_values(ascending=False).head(n_vec).index)
                    desc_modo = f"con mayor carga absoluta en {x_component}"
                elif modo == MODO_Y:
                    top_vars = list(ly.abs().sort_values(ascending=False).head(n_vec).index)
                    desc_modo = f"con mayor carga absoluta en {y_component}"
                elif modo == MODO_PLANO:
                    top_vars = list(reach.sort_values(ascending=False).head(n_vec).index)
                    desc_modo = "mejor representadas en este plano"
                else:
                    # Mitad de las flechas según cada componente (sin repetir variables).
                    top_vars = list(lx.abs().sort_values(ascending=False).head(n_vec - n_vec // 2).index)
                    for v in ly.abs().sort_values(ascending=False).index:
                        if len(top_vars) >= n_vec:
                            break
                        if v not in top_vars:
                            top_vars.append(v)
                    desc_modo = f"con mayor carga absoluta en {x_component} y en {y_component} (mitad según cada eje)"

                                # Un único factor escala las flechas para que quepan en la nube de puntos.
                ref = min(np.percentile(np.abs(sx), 98), np.percentile(np.abs(sy), 98))
                scale = 0.9 * ref * escala_flechas / float(reach[top_vars].max())
                for is_target, color, label in (
                    (False, INK, "Variable explicativa"),
                    (True, ORANGE, "Variable de criminalidad (target)"),
                ):
                    names = [v for v in top_vars if (v in target_set) == is_target]
                    if not names:
                        continue
                    tx, ty = lx[names].to_numpy() * scale, ly[names].to_numpy() * scale
                    # Flecha simple desde el origen
                    for a, b in zip(tx, ty):
                        fig.add_annotation(
                            x=a, y=b, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                            arrowcolor=color, text="",
                        )
                    # Nombre de la variable como texto simple junto a la punta
                    positions = [
                        ("top" if b >= 0 else "bottom") + (" right" if a >= 0 else " left")
                        for a, b in zip(tx, ty)
                    ]
                    fig.add_trace(go.Scatter(
                        x=tx, y=ty, mode="markers+text", name=label,
                        text=[etiqueta(n) for n in names], textposition=positions,
                        textfont=dict(size=14, color=color, family=FONT),
                        marker=dict(size=6, color=color),
                        customdata=np.column_stack([lx[names].to_numpy(), ly[names].to_numpy()]),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"Carga en {x_component}: %{{customdata[0]:+.3f}}<br>"
                            f"Carga en {y_component}: %{{customdata[1]:+.3f}}<extra></extra>"
                        ),
                    ))
            base_layout(fig, chart_height, titulo)
            fig.update_xaxes(
                title_text=f"<b>{x_component}</b> ({var_of[x_component]:.2f} % de varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
            )
            fig.update_yaxes(
                title_text=f"<b>{y_component}</b> ({var_of[y_component]:.2f} % de varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
                **(dict(scaleanchor="x", scaleratio=1) if show_vec else {}),
            )
            show_chart(fig, f"proyeccion_{x_component}_{y_component}")

            caption = (
                f'<p class="figcap"><b>Figura 3.</b> Proyección de las n = {fmt_int(X.shape[0])} comunidades '
                f"sobre {x_component} ({var_of[x_component]:.2f} % de la varianza) y {y_component} "
                f"({var_of[y_component]:.2f} % de la varianza). Cada punto es una comunidad; al pasar el "
                f"puntero se muestra su nombre. La elipse discontinua delimita la región esperada del 95 % "
                f"bajo normalidad bivariada, y las líneas negras marcan el origen (puntuación 0) de cada eje."
            )
            if show_vec:
                caption += (
                    f" Las flechas son las {len(top_vars)} variables {desc_modo} (las de criminalidad llevan ★ y "
                    f"van en naranja); los puntos se atenúan para destacarlas. Con los ejes a la misma escala, el "
                    f"ángulo entre dos flechas aproxima la correlación entre sus variables (pequeño: positiva; "
                    f"cercano a 90°: sin relación en este plano; cercano a 180°: negativa). La longitud está "
                    f"escalada para verla junto a las comunidades, y una comunidad se ubica hacia donde apuntan "
                    f"las variables en las que es alta."
                )
            st.markdown(caption + "</p>", unsafe_allow_html=True)
    with col_guide:
        render_guide([x_component, y_component])

    # ------------------------------ 3. Cargas (1 por vez) ------------------------------
    st.markdown(
        '<div class="section-h" style="margin-top:2.4rem;">3. Cargas de las variables en cada componente</div>',
        unsafe_allow_html=True,
    )
    component = st.radio(
        "Componente", components, horizontal=True, label_visibility="collapsed",
        format_func=lambda pc: f"{pc} · {var_of[pc]:.1f} %", key="pc_choice",
    )
    max_top = min(30, len(X.columns))
    _, col_n, _ = st.columns([1.5, 2, 1.5])
    top_n = col_n.slider(
        "Variables a mostrar", min_value=min(5, max_top), max_value=max_top,
        value=min(TOP_N_DEFECTO, max_top),
    )

    col_chart2, col_guide2 = st.columns([3.2, 1.25], gap="medium")
    with col_chart2:
        # Como en el notebook: las top_n variables con mayor carga absoluta, ordenadas de menor a mayor carga.
        carga_comp = loadings_df[component]
        values = carga_comp.reindex(carga_comp.abs().sort_values(ascending=False).head(top_n).index).sort_values()
        order = [etiqueta(v) for v in values.index]
        pos, neg = values[values >= 0], values[values < 0]
        fig = go.Figure()
        for part, color, label in ((pos, BLUE, "Carga positiva"), (neg, ORANGE, "Carga negativa")):
            fig.add_trace(go.Bar(
                x=part.values, y=[etiqueta(v) for v in part.index], orientation="h", name=label,
                marker=dict(color=color, line=dict(width=1.5, color=INK)),
                text=[f"{v:+.2f}" for v in part.values], textposition="outside", cliponaxis=False,
                textfont=dict(size=16, color=INK, family=FONT),
                hovertemplate="<b>%{y}</b><br>Carga: %{x:+.3f}<extra></extra>",
            ))
        limit = float(values.abs().max()) * 1.2
        base_layout(fig, chart_height, f"Principales cargas de {component} ({var_of[component]:.2f} % de la varianza)")
        fig.update_layout(barmode="overlay", bargap=0.25)
        fig.update_xaxes(
            title_text="<b>Carga</b> (correlación entre la variable y el componente)",
            range=[-limit, limit],
            zeroline=True, zerolinecolor=INK, zerolinewidth=3,
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=order,
            showgrid=False, tickfont=dict(size=17 if top_n <= 18 else 13, color=INK),
        )
        show_chart(fig, f"cargas_{component}")
        st.markdown(
            f'<p class="figcap"><b>Figura 4.</b> Principales cargas de {component} '
            f"({var_of[component]:.2f} % de la varianza): las {top_n} variables con mayor valor absoluto, "
            f"ordenadas de menor a mayor carga (★ = variable de criminalidad). La carga es el autovector multiplicado "
            f"por la raíz del autovalor, es decir, la correlación entre la variable estandarizada y el componente; "
            f"el signo indica el sentido de la asociación, no una diferencia de naturaleza.</p>",
            unsafe_allow_html=True,
        )
    with col_guide2:
        render_guide([component])

    # ------------------------------ Datos y metodología ------------------------------
    with st.expander("Datos y metodología", expanded=False):
        targets_txt = ", ".join(targets_sel.columns)
        targets_perdidos = [c for c in targets_sel.columns if c not in X.columns]
        st.markdown(
            f"""
**Fuente.** UCI Machine Learning Repository, *Communities and Crime Unnormalized* (id. 211).

**Matriz de partida.** {fmt_int(features.shape[1])} variables explicativas más las variables de criminalidad
(targets) incluidas en el análisis: {targets_txt}. En total, {fmt_int(n_inicial)} variables.

**Preparación (celda 5 del notebook).**
1. Se excluyen identificadores o metadatos: {", ".join(COLUMNAS_IDENTIFICADORAS) or "ninguno"}.
2. Se eliminan las variables con más de {MAX_FALTANTES * 100:.0f} % de datos faltantes, tanto explicativas como
de criminalidad: {fmt_int(len(elim_por_na))} variables.
3. Se eliminan las variables redundantes o no deseadas: {", ".join(COLUMNAS_REDUNDANTES) or "ninguna"}.
4. Los datos faltantes restantes ({fmt_int(total_faltantes_imputados)} celdas) se imputan con la mediana de cada
variable y luego cada variable se estandariza (media 0, desviación 1).

Quedan {fmt_int(X.shape[1])} variables, de las cuales {fmt_int(n_targets_kept)} son de criminalidad
(descartadas por los filtros: {", ".join(targets_perdidos) or "ninguna"}). No se elimina ninguna comunidad
(n = {fmt_int(X.shape[0])}).

**PCA (celda 6 del notebook).** Se calculan todos los componentes (`n_components=None`, `random_state={RANDOM_STATE}`).
La varianza explicada y acumulada se muestra en la sección 1. Las cargas se calculan como autovectores ×
√autovalor, equivalentes a la correlación entre cada variable y cada componente. En los gráficos de cargas y
en el biplot se muestran las {TOP_N_DEFECTO} variables de mayor carga absoluta (ajustable).

**Vectores de variables (biplot).** Se activan en la sección 2 y se dibujan sobre la proyección de las comunidades.
Las flechas son las variables de mayor carga absoluta según el componente elegido (eje X, eje Y, mitad de cada uno, o las
mejor representadas en el plano), con coordenadas iguales a sus cargas multiplicadas por un factor común para que quepan
en la nube de puntos. El modo «Según el eje X» reproduce el criterio del notebook.

**Elipse de confianza.** Región del 95 % bajo normalidad bivariada de las puntuaciones; los puntos fuera de ella
son candidatos a observaciones atípicas en ese plano.

**Nota metodológica.** La imputación y la estandarización se realizan sobre el conjunto completo; es razonable para
una exploración descriptiva, pero para modelos predictivos debería ajustarse solo con datos de entrenamiento.
            """
        )
        if len(descartadas_df):
            st.markdown("**Variables descartadas**")
            st.markdown(html_table(descartadas_df), unsafe_allow_html=True)
        else:
            st.markdown("No se descartó ninguna variable.")

except Exception as error:
    st.error(f"No se pudo cargar o procesar el dataset: {error}")
    st.info(
        "Compruebe la conexión a internet. Si el error persiste, UCI puede estar temporalmente "
        "inaccesible; vuelva a intentar más tarde."
    )
