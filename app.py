import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
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
# Paleta y estilo
# --------------------------------------------------------------------------
INK = "#1B2A3A"        # texto principal
MUTED = "#5B6B7B"      # texto secundario
NAVY = "#1F4E79"       # color principal
BRICK = "#B5543F"      # carga negativa
GRID = "#E3E8EE"       # líneas de cuadrícula
PAPER = "#F6F8FA"      # fondo de la página
WHITE = "#FFFFFF"      # fondo de gráficos

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: {PAPER};
        color: {INK};
        font-family: 'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 17px;
    }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* Contenedor central: se adapta al ancho del monitor */
    .block-container {{
        max-width: min(1180px, 92vw);
        margin: 0 auto;
        padding: 3rem 1rem 4rem 1rem;
    }}
    @media (min-width: 1700px) {{
        html, body, [data-testid="stApp"] {{ font-size: 19px; }}
        .block-container {{ max-width: min(1400px, 88vw); }}
    }}
    @media (min-width: 2300px) {{
        html, body, [data-testid="stApp"] {{ font-size: 22px; }}
        .block-container {{ max-width: min(1750px, 82vw); }}
    }}
    @media (max-width: 700px) {{
        .block-container {{ max-width: 96vw; padding-top: 1.5rem; }}
    }}

    /* Tipografía */
    h1, h2, h3 {{
        color: {INK};
        text-align: center;
        font-family: 'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif;
        letter-spacing: -0.01em;
    }}
    h1 {{ font-size: 2.4rem; font-weight: 700; padding-bottom: 0.2rem; }}
    h2, h3 {{ font-size: 1.5rem; font-weight: 600; margin-top: 2.2rem; }}
    .subtitle {{
        text-align: center; color: {MUTED};
        font-size: 1.1rem; margin: 0 auto 1.5rem auto; max-width: 46rem;
    }}
    .section-rule {{
        height: 1px; background: {GRID}; border: 0; margin: 2.2rem auto 0 auto; width: 100%;
    }}

    /* Controles centrados */
    [data-testid="stWidgetLabel"] {{ justify-content: center; }}
    [data-testid="stWidgetLabel"] p {{
        text-align: center; font-weight: 600; color: {INK}; font-size: 1rem;
    }}
    [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {{ text-align: left; }}
    [data-testid="stSlider"] {{ padding-left: 0.5rem; padding-right: 0.5rem; }}
    [data-testid="stAlert"] {{ text-align: center; }}

    /* Gráficos sobre tarjeta blanca */
    [data-testid="stPlotlyChart"] {{
        background: {WHITE};
        border: 1px solid {GRID};
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.8rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Communities and Crime — Componentes principales")
st.markdown(
    '<p class="subtitle">Proyecciones PCA de las comunidades y cargas de las variables '
    "en los componentes PC1 a PC7.</p>",
    unsafe_allow_html=True,
)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"


def base_layout(fig, height):
    """Estilo común y legible para presentar."""
    fig.update_layout(
        height=height,
        template="simple_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif", size=15, color=INK),
        title=dict(x=0.5, xanchor="center", font=dict(size=20, color=INK)),
        margin=dict(l=20, r=20, t=70, b=40),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=NAVY, font=dict(size=15, color=INK)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, title_font=dict(size=16))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title_font=dict(size=16))
    return fig


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
    scores_df = pd.DataFrame(scores, columns=components, index=X.index)
    scores_df["Comunidad"] = community_names.reindex(X.index).fillna("Sin nombre").values
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=components)

    # ---------------------------- Proyecciones ----------------------------
    st.subheader("Proyecciones de componentes principales")
    _, col_x, col_y, _ = st.columns([0.5, 2, 2, 0.5])
    x_component = col_x.selectbox("Componente del eje X", components, index=0)
    y_component = col_y.selectbox(
        "Componente del eje Y", components,
        index=1 if n_components > 1 else 0,
    )
    if x_component == y_component:
        st.warning("Seleccione componentes distintos para los ejes X e Y.")
    else:
        fig = px.scatter(
            scores_df, x=x_component, y=y_component,
            hover_name="Comunidad",
            opacity=0.7,
            labels={
                x_component: f"{x_component} ({variance[components.index(x_component)]:.2f}% de varianza)",
                y_component: f"{y_component} ({variance[components.index(y_component)]:.2f}% de varianza)",
            },
            title=f"Proyección: {x_component} vs. {y_component}",
        )
        fig.update_traces(
            marker=dict(size=8, color=NAVY, line=dict(width=0.6, color=WHITE)),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                f"{x_component}: %{{x:.2f}}<br>"
                f"{y_component}: %{{y:.2f}}<extra></extra>"
            ),
        )
        fig.add_hline(y=0, line_color="#9AA7B4", line_width=1, line_dash="dot")
        fig.add_vline(x=0, line_color="#9AA7B4", line_width=1, line_dash="dot")
        base_layout(fig, height=680)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------- Cargas -------------------------------
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Cargas de los componentes principales (PC1–PC7)")

    _, col_sel, _ = st.columns([0.5, 4, 0.5])
    selected_components = col_sel.multiselect(
        "Componentes para mostrar", components, default=components
    )
    max_top = min(30, len(X.columns))
    _, col_sld, _ = st.columns([1, 3, 1])
    top_n = col_sld.slider(
        "Cantidad de variables por gráfico",
        min_value=min(5, max_top), max_value=max_top, value=min(15, max_top),
    )

    if selected_components:
        bar_height = max(460, 28 * top_n + 150)
        chart_cols = st.columns(2, gap="medium")
        for i, component in enumerate(selected_components):
            values = loadings_df[component].sort_values(
                key=lambda series: series.abs(), ascending=False
            ).head(top_n).sort_values()
            fig = px.bar(
                x=values.values, y=values.index, orientation="h",
                labels={"x": "Carga", "y": ""},
                title=(
                    f"Cargas de {component} "
                    f"({variance[components.index(component)]:.1f}% de varianza)"
                ),
                color=values.values,
                color_continuous_scale=[BRICK, "#E6EAEF", NAVY],
                color_continuous_midpoint=0,
            )
            fig.update_traces(hovertemplate="<b>%{y}</b><br>Carga: %{x:.3f}<extra></extra>")
            fig.add_vline(x=0, line_color=INK, line_width=1)
            base_layout(fig, height=bar_height)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=20, t=70, b=40))
            fig.update_yaxes(showgrid=False, tickfont=dict(size=13))
            chart_cols[i % 2].plotly_chart(fig, use_container_width=True)

except Exception as error:
    st.error(f"No se pudo cargar o procesar el dataset: {error}")
    st.info(
        "Compruebe la conexión a internet. Si el error persiste, UCI puede estar temporalmente "
        "inaccesible; vuelva a intentar más tarde."
    )
