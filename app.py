import io
import zipfile
import urllib.request

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

# --------------------------------------------------------------------------
# Paleta de alto contraste (pensada para proyectar sobre una pared)
# --------------------------------------------------------------------------
INK = "#0B1220"        # texto, ejes y líneas principales (casi negro)
BLUE = "#1D4ED8"       # color principal / cargas positivas
ORANGE = "#D9480F"     # cargas negativas y resaltado
GRID = "#CBD5E1"       # cuadrícula (visible pero secundaria)
PAPER = "#F1F5F9"      # fondo de la página
WHITE = "#FFFFFF"      # fondo de los gráficos
FONT = "Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background: %PAPER%;
    color: %INK%;
    font-family: 'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 18px;
}
/* Barra superior de Streamlit fuera: gana espacio vertical */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }

/* Contenedor central adaptable al monitor */
.block-container {
    max-width: min(1280px, 94vw);
    margin: 0 auto;
    padding: 1.2rem 1rem 1rem 1rem;
}
@media (min-width: 1700px) {
    html, body, [data-testid="stApp"] { font-size: 20px; }
    .block-container { max-width: min(1500px, 90vw); }
}
@media (min-width: 2300px) {
    html, body, [data-testid="stApp"] { font-size: 24px; }
    .block-container { max-width: min(1900px, 86vw); }
}
@media (max-width: 700px) { .block-container { max-width: 98vw; } }

/* Título */
h1 {
    color: %INK%; text-align: center; font-weight: 700;
    font-size: 2.1rem; padding: 0 0 0.1rem 0; letter-spacing: -0.01em;
}
.subtitle {
    text-align: center; color: #334155; font-size: 1.05rem; margin: 0 0 0.8rem 0;
}

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

/* Gráfico: sin relleno extra para evitar la barra de desplazamiento interna */
[data-testid="stPlotlyChart"] { padding: 0 !important; margin: 0.4rem 0 0 0; overflow: hidden !important; }
[data-testid="stPlotlyChart"] > div { overflow: hidden !important; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
</style>
"""
for _name, _value in {"%PAPER%": PAPER, "%INK%": INK, "%BLUE%": BLUE, "%WHITE%": WHITE}.items():
    CSS = CSS.replace(_name, _value)
st.markdown(CSS, unsafe_allow_html=True)

st.title("Communities and Crime — Componentes principales")
st.markdown(
    '<p class="subtitle">Análisis de componentes principales (PCA) sobre las comunidades '
    "del dataset de UCI.</p>",
    unsafe_allow_html=True,
)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"


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
        margin=dict(l=20, r=25, t=75, b=20),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(size=17, color="#FFFFFF", family=FONT)),
    )
    style_axes(fig)
    return fig



def show_chart(fig):
    """Muestra el gráfico a todo el ancho (compatible con versiones nuevas y antiguas de Streamlit)."""
    config = {"displaylogo": False}
    try:
        st.plotly_chart(fig, width="stretch", config=config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config)


@st.cache_data(show_spinner="Descargando dataset desde UCI...")
def load_data():
    """Carga el dataset de UCI. Devuelve (predictores, nombres de comunidades)."""
    api_error = None
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=211)
        features = dataset.data.features
        if features is not None and not features.empty:
            features = features.copy()
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
            features = features.reset_index(drop=True)
            return features, names
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
    # 125 predictores y 18 variables de criminalidad.
    if raw.shape[1] < 30:
        raise RuntimeError(f"El archivo descargado tiene un formato inesperado ({raw.shape[1]} columnas).")
    predictors = raw.iloc[:, 4:-18].copy()
    predictors.columns = [f"Variable {i + 1:03d}" for i in range(predictors.shape[1])]
    names = raw.iloc[:, 0].astype(str).str.strip()
    states = raw.iloc[:, 1].astype(str).str.strip()
    names = (names + " (" + states + ")").reset_index(drop=True)
    predictors = predictors.reset_index(drop=True)
    if api_error:
        st.info("Se utilizó la descarga directa de UCI porque la API no respondió correctamente.")
    return predictors, names


try:
    raw, community_names = load_data()
    X = raw.select_dtypes(include=[np.number]).copy()
    # Convierte columnas que parecen numéricas y elimina las que no tienen valores útiles.
    for column in raw.columns:
        if column not in X.columns:
            converted = pd.to_numeric(raw[column], errors="coerce")
            if converted.notna().any():
                X[column] = converted
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.empty:
        raise RuntimeError("No se encontraron variables numéricas utilizables.")

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

    # ------------------- Selector de vista + ajuste de tamaño -------------------
    _, col_view, col_cfg = st.columns([1, 6, 1])
    view = col_view.radio(
        "Gráfico", ["Proyección PCA", "Cargas por componente"],
        horizontal=True, label_visibility="collapsed", key="view",
    )
    with col_cfg.popover("⚙ Tamaño"):
        chart_height = st.slider(
            "Alto del gráfico (px)", min_value=420, max_value=1000, value=620, step=20,
            help="Ajuste según la pantalla o el proyector para evitar el desplazamiento vertical.",
        )

    # ------------------------------ Proyección PCA ------------------------------
    if view == "Proyección PCA":
        col_x, col_y, col_find = st.columns([1, 1, 1.6])
        x_component = col_x.selectbox("Eje X", components, index=0)
        y_component = col_y.selectbox("Eje Y", components, index=1 if n_components > 1 else 0)
        highlighted = col_find.selectbox(
            "Resaltar una comunidad",
            sorted(scores_df["Comunidad"].unique()),
            index=None,
            placeholder="Escriba para buscar…",
        )

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
                name="Comunidades", showlegend=False,
            )
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
            base_layout(fig, chart_height, f"Proyección PCA: {x_component} vs. {y_component}")
            fig.update_xaxes(
                title_text=f"<b>{x_component}</b> ({var_of[x_component]:.1f}% de la varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
            )
            fig.update_yaxes(
                title_text=f"<b>{y_component}</b> ({var_of[y_component]:.1f}% de la varianza)",
                zeroline=True, zerolinecolor=INK, zerolinewidth=2.5,
            )
            show_chart(fig)

    # ------------------------------ Cargas (1 por vez) ------------------------------
    else:
        col_pc, col_n = st.columns([3.2, 1.3])
        component = col_pc.radio(
            "Componente", components, horizontal=True, label_visibility="collapsed",
            format_func=lambda pc: f"{pc} · {var_of[pc]:.1f}%", key="pc_choice",
        )
        max_top = min(30, len(X.columns))
        top_n = col_n.slider(
            "Variables a mostrar", min_value=min(5, max_top), max_value=max_top,
            value=min(15, max_top),
        )

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
        base_layout(
            fig, chart_height,
            f"Cargas de {component} — {var_of[component]:.1f}% de la varianza",
        )
        fig.update_layout(
            barmode="overlay", bargap=0.25,
            legend=dict(
                orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom",
                font=dict(size=17, color=INK), bgcolor="rgba(255,255,255,0)",
            ),
            margin=dict(l=20, r=25, t=105, b=20),
        )
        fig.update_xaxes(
            title_text="<b>Carga</b> (correlación de la variable con el componente)",
            range=[-limit, limit],
            zeroline=True, zerolinecolor=INK, zerolinewidth=3,
        )
        fig.update_yaxes(
            categoryorder="array", categoryarray=order, autorange="reversed",
            showgrid=False, tickfont=dict(size=17 if top_n <= 18 else 13, color=INK),
        )
        show_chart(fig)

except Exception as error:
    st.error(f"No se pudo cargar o procesar el dataset: {error}")
    st.info(
        "Compruebe la conexión a internet. Si el error persiste, UCI puede estar temporalmente "
        "inaccesible; vuelva a intentar más tarde."
    )
