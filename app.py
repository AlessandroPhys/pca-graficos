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

# Se eliminan las variables (predictoras y de criminalidad/target) con más de este
# porcentaje de datos faltantes.
MAX_FALTANTES = 0.50

# Columnas que identifican (no miden) y por tanto no deben entrar al PCA.
COLUMNAS_IDENTIFICADORAS = {"fold", "countycode", "communitycode", "communityname", "state"}

# Nombres de las 18 variables de criminalidad (targets) del dataset sin normalizar.
NOMBRES_CRIMEN = [
    "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop",
    "assaults", "assaultPerPop", "burglaries", "burglPerPop", "larcenies", "larcPerPop",
    "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop",
    "ViolentCrimesPerPop", "nonViolPerPop",
]

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
    "%PAPER%": PAPER, "%INK%": INK, "%BLUE%": BLUE, "%WHITE%": WHITE, "%BODY%": BODY,
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


def base_layout(fig, height, title):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=FONT, size=17, color=INK),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=25, color=INK)),
        margin=dict(l=20, r=25, t=105, b=20),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
            font=dict(size=17, color=INK), bgcolor="rgba(255,255,255,0)", itemsizing="constant",
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

    Devuelve (tabla de variables, nombres de comunidades, lista de columnas target).
    La tabla incluye las variables predictoras Y las de criminalidad (targets).
    """
    api_error = None
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=211)
        features = dataset.data.features
        if features is not None and not features.empty:
            features = features.copy().reset_index(drop=True)
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

            # Las variables target (criminalidad) se incorporan siempre.
            target_cols = []
            targets = dataset.data.targets
            if targets is not None and not targets.empty:
                targets = targets.copy().reset_index(drop=True)
                targets = targets.loc[:, ~targets.columns.isin(features.columns)]
                target_cols = list(targets.columns)
                features = pd.concat([features, targets], axis=1)
            return features, names, target_cols
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

    # Estructura: 4 columnas de identificación (nombre, estado, condado, comunidad),
    # opcionalmente la columna "fold" (partición de validación, valores 1 a 10),
    # las variables predictoras y, al final, 18 variables de criminalidad (targets).
    if raw.shape[1] < 30:
        raise RuntimeError(f"El archivo descargado tiene un formato inesperado ({raw.shape[1]} columnas).")
    start = 4
    col4 = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
    if col4.notna().all() and col4.between(1, 10).all() and col4.nunique() <= 10:
        start = 5  # columna "fold": identificador de partición, no una medición
    body = raw.iloc[:, start:].copy()
    n_pred = body.shape[1] - len(NOMBRES_CRIMEN)
    body.columns = [f"Variable {i + 1:03d}" for i in range(n_pred)] + NOMBRES_CRIMEN
    names = raw.iloc[:, 0].astype(str).str.strip()
    states = raw.iloc[:, 1].astype(str).str.strip()
    names = (names + " (" + states + ")").reset_index(drop=True)
    body = body.reset_index(drop=True)
    if api_error:
        st.info("Se utilizó la descarga directa de UCI porque la API no respondió correctamente.")
    return body, names, list(NOMBRES_CRIMEN)


try:
    raw, community_names, target_cols = load_data()
    target_set = set(target_cols)

    # ------------------------------ Preparación de datos ------------------------------
    raw = raw.drop(columns=[c for c in raw.columns if str(c).lower() in COLUMNAS_IDENTIFICADORAS])
    X = raw.select_dtypes(include=[np.number]).copy()
    # Convierte columnas que parecen numéricas.
    for column in raw.columns:
        if column not in X.columns:
            converted = pd.to_numeric(raw[column], errors="coerce")
            if converted.notna().any():
                X[column] = converted
    X = X.replace([np.inf, -np.inf], np.nan)
    n_candidatas = X.shape[1]

    # Filtro: se eliminan las variables (predictoras y targets) con más de 50 % de faltantes.
    missing_frac = X.isna().mean()
    too_missing = missing_frac[missing_frac > MAX_FALTANTES]
    X = X.drop(columns=too_missing.index)
    removed_df = pd.DataFrame({
        "Variable": list(too_missing.index),
        "Tipo": ["Criminalidad (target)" if v in target_set else "Predictora" for v in too_missing.index],
        "% faltante": (too_missing.values * 100).round(1),
    }).sort_values("% faltante", ascending=False)

    # Se descartan además las variables constantes (varianza nula).
    const_cols = list(X.columns[X.nunique(dropna=True) <= 1])
    X = X.drop(columns=const_cols)
    if X.empty:
        raise RuntimeError("No se encontraron variables numéricas utilizables.")

    n_targets_kept = sum(1 for c in X.columns if c in target_set)
    n_removed_targets = int((removed_df["Tipo"] == "Criminalidad (target)").sum()) if len(removed_df) else 0
    missing_after = float(X.isna().to_numpy().mean() * 100)

    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    Z = StandardScaler().fit_transform(X)
    n_components = min(7, Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Z)
    components = [f"PC{i + 1}" for i in range(n_components)]
    variance = pca.explained_variance_ratio_ * 100
    var_of = dict(zip(components, variance))
    scores_df = pd.DataFrame(scores, columns=components, index=X.index)
    scores_df["Comunidad"] = community_names.reindex(X.index).fillna("Sin nombre").values
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=components)

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

    # ------------------------------ Resumen del análisis ------------------------------
    stats = [
        (fmt_int(X.shape[0]), "Observaciones (n)"),
        (fmt_int(X.shape[1]), "Variables analizadas (p)"),
        (str(n_components), "Componentes (k)"),
        (f"{variance.sum():.1f} %", f"Varianza acumulada PC1–PC{n_components}"),
        (fmt_int(len(too_missing)), f"Variables excluidas (> {MAX_FALTANTES * 100:.0f} % faltantes)"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>' for v, l in stats)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------ 1. Proyección PCA ------------------------------
    _, col_head, col_cfg = st.columns([1, 6, 1])
    col_head.markdown(
        '<div class="section-h">1. Proyección de las comunidades en el espacio de componentes</div>',
        unsafe_allow_html=True,
    )
    with col_cfg.popover("⚙ Tamaño"):
        chart_height = st.slider(
            "Alto de los gráficos (px)", min_value=420, max_value=1000, value=640, step=20,
            help="Ajuste según la pantalla o el proyector.",
        )

    col_x, col_y, col_find = st.columns([1, 1, 1.6])
    x_component = col_x.selectbox("Eje X", components, index=0)
    y_component = col_y.selectbox("Eje Y", components, index=1 if n_components > 1 else 0)
    highlighted = col_find.selectbox(
        "Resaltar una comunidad",
        sorted(scores_df["Comunidad"].unique()),
        index=None,
        placeholder="Escriba para buscar…",
    )

    col_chart, col_guide = st.columns([3.2, 1.25], gap="medium")
    with col_chart:
        if x_component == y_component:
            st.warning("Seleccione componentes distintos para los ejes X e Y.")
        else:
            fig = px.scatter(
                scores_df, x=x_component, y=y_component,
                hover_name="Comunidad", render_mode="svg",
            )
            fig.update_traces(
                marker=dict(size=9, color=BLUE, opacity=0.72, line=dict(width=0.8, color=WHITE)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    f"{x_component}: %{{x:.2f}}<br>"
                    f"{y_component}: %{{y:.2f}}<extra></extra>"
                ),
                name="Comunidades", showlegend=True,
            )
            # Elipse de confianza del 95 % (aprox. normal bivariada): chi2(0.95, 2 g.l.) = -2 ln(0.05)
            ix, iy = components.index(x_component), components.index(y_component)
            radius = np.sqrt(-2 * np.log(0.05))
            t = np.linspace(0, 2 * np.pi, 240)
            fig.add_trace(go.Scatter(
                x=radius * np.sqrt(pca.explained_variance_[ix]) * np.cos(t),
                y=radius * np.sqrt(pca.explained_variance_[iy]) * np.sin(t),
                mode="lines", name="Elipse de confianza del 95 %",
                line=dict(color=INK, width=2.5, dash="dash"), hoverinfo="skip",
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
            base_layout(fig, chart_height, f"{x_component} vs. {y_component}")
            fig.update_xaxes(
                title_text=f"<b>{x_component}</b> ({var_of[x_component]:.1f} % de la varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
            )
            fig.update_yaxes(
                title_text=f"<b>{y_component}</b> ({var_of[y_component]:.1f} % de la varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
            )
            show_chart(fig, f"proyeccion_{x_component}_{y_component}")
            st.markdown(
                f'<p class="figcap"><b>Figura 1.</b> Proyección de las n = {fmt_int(X.shape[0])} comunidades '
                f"sobre {x_component} ({var_of[x_component]:.1f} % de la varianza) y {y_component} "
                f"({var_of[y_component]:.1f} % de la varianza). Cada punto es una comunidad; al pasar el "
                f"puntero se muestra su nombre. La elipse discontinua delimita la región esperada del 95 % "
                f"bajo normalidad bivariada, y las líneas negras marcan el origen (puntuación 0) de cada eje.</p>",
                unsafe_allow_html=True,
            )
    with col_guide:
        render_guide([x_component, y_component])

    # ------------------------------ 2. Cargas (1 por vez) ------------------------------
    st.markdown(
        '<div class="section-h" style="margin-top:2.4rem;">2. Cargas de las variables en cada componente</div>',
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
        value=min(15, max_top),
    )

    col_chart2, col_guide2 = st.columns([3.2, 1.25], gap="medium")
    with col_chart2:
        values = loadings_df[component].reindex(
            loadings_df[component].abs().sort_values(ascending=False).head(top_n).index
        )
        order = list(values.index)
        pos, neg = values[values >= 0], values[values < 0]
        fig = go.Figure()
        for part, color, label in ((pos, BLUE, "Carga positiva"), (neg, ORANGE, "Carga negativa")):
            fig.add_trace(go.Bar(
                x=part.values, y=part.index, orientation="h", name=label,
                marker=dict(color=color, line=dict(width=1.5, color=INK)),
                text=[f"{v:+.2f}" for v in part.values], textposition="outside", cliponaxis=False,
                textfont=dict(size=16, color=INK, family=FONT),
                hovertemplate="<b>%{y}</b><br>Carga: %{x:+.3f}<extra></extra>",
            ))
        limit = float(values.abs().max()) * 1.2
        base_layout(fig, chart_height, f"Cargas de {component} ({var_of[component]:.1f} % de la varianza)")
        fig.update_layout(barmode="overlay", bargap=0.25)
        fig.update_xaxes(
            title_text="<b>Carga</b> (correlación entre la variable y el componente)",
            range=[-limit, limit],
            zeroline=True, zerolinecolor=INK, zerolinewidth=3,
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=order, autorange="reversed",
            showgrid=False, tickfont=dict(size=17 if top_n <= 18 else 13, color=INK),
        )
        show_chart(fig, f"cargas_{component}")
        st.markdown(
            f'<p class="figcap"><b>Figura 2.</b> Cargas de las {top_n} variables con mayor valor absoluto en '
            f"{component} ({var_of[component]:.1f} % de la varianza). La carga es la correlación entre la "
            f"variable estandarizada y el componente: valores cercanos a ±1 indican que la variable "
            f"contribuye con fuerza a {component}; el signo indica el sentido de la asociación.</p>",
            unsafe_allow_html=True,
        )
    with col_guide2:
        render_guide([component])

    # ------------------------------ Datos y metodología ------------------------------
    with st.expander("Datos y metodología", expanded=False):
        st.markdown(
            f"""
**Fuente.** UCI Machine Learning Repository, *Communities and Crime Unnormalized* (id. 211).

**Variables consideradas.** Se parte de {fmt_int(n_candidatas)} variables numéricas: predictoras y de
criminalidad (targets). Se excluyen los identificadores (nombre de la comunidad, estado, códigos de condado y
comunidad, partición *fold*).

**Filtro de datos faltantes.** Se eliminan las variables con más de {MAX_FALTANTES * 100:.0f} % de datos
faltantes, tanto predictoras como de criminalidad: {fmt_int(len(too_missing))} variables eliminadas
({fmt_int(n_removed_targets)} de criminalidad). Se eliminan también {fmt_int(len(const_cols))} variables constantes.
Quedan {fmt_int(X.shape[1])} variables, de las cuales {fmt_int(n_targets_kept)} son de criminalidad.

**Imputación.** Los datos faltantes restantes ({missing_after:.2f} % de las celdas de las variables conservadas)
se reemplazan por la mediana de cada variable. No se elimina ninguna comunidad.

**Estandarización y PCA.** Las variables se estandarizan (media 0, desviación 1) y se aplica PCA sobre la matriz
de correlaciones. Las cargas se calculan como vectores propios × √(autovalor), equivalentes a la correlación
entre cada variable y cada componente.

**Elipse de confianza.** Región del 95 % bajo normalidad bivariada de las puntuaciones; los puntos fuera de ella
son candidatos a observaciones atípicas en ese plano.
            """
        )
        if len(removed_df):
            st.markdown("**Variables eliminadas por exceso de datos faltantes**")
            st.dataframe(removed_df, hide_index=True)
        else:
            st.markdown("Ninguna variable superó el umbral de datos faltantes.")

except Exception as error:
    st.error(f"No se pudo cargar o procesar el dataset: {error}")
    st.info(
        "Compruebe la conexión a internet. Si el error persiste, UCI puede estar temporalmente "
        "inaccesible; vuelva a intentar más tarde."
    )
