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

st.set_page_config(page_title="PCA | Communities and Crime", layout="wide")
st.title("Communities and Crime — Componentes principales")
st.caption("Proyecciones PCA y gráficos de cargas de PC1 a PC7.")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"

@st.cache_data(show_spinner="Descargando dataset desde UCI...")
def load_data():
    """Load UCI dataset, falling back to its direct archive if the API fails."""
    api_error = None
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=211)
        features = dataset.data.features
        if features is not None and not features.empty:
            return features.copy()
    except Exception as exc:
        api_error = str(exc)

    # Fallback: download the original UCI archive directly.
    request = urllib.request.Request(
        UCI_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 PCA-visualization-app"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        data_file = next(
            (name for name in archive.namelist() if name.lower().endswith("commviolpredunnormalizeddata.txt")),
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

    # Dataset layout: 4 identifying columns, 125 predictors, 18 crime outcomes.
    if raw.shape[1] < 30:
        raise RuntimeError(f"El archivo descargado tiene un formato inesperado ({raw.shape[1]} columnas).")
    predictors = raw.iloc[:, 4:-18].copy()
    predictors.columns = [f"Variable {i+1:03d}" for i in range(predictors.shape[1])]
    if api_error:
        st.info("Se utilizó la descarga directa de UCI porque la API no respondió correctamente.")
    return predictors

try:
    raw = load_data()
    X = raw.select_dtypes(include=[np.number]).copy()
    # Convert numeric-looking columns and remove columns with no usable values.
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
    components = [f"PC{i+1}" for i in range(n_components)]
    variance = pca.explained_variance_ratio_ * 100
    scores_df = pd.DataFrame(scores, columns=components, index=X.index)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=components)

    st.subheader("Proyecciones de componentes principales")
    col_x, col_y = st.columns(2)
    x_component = col_x.selectbox("Componente del eje X", components, index=0)
    y_component = col_y.selectbox(
        "Componente del eje Y", components,
        index=1 if n_components > 1 else 0,
    )
    if x_component == y_component:
        st.warning("Seleccione componentes distintos para los ejes X e Y.")
    else:
        fig = px.scatter(
            scores_df, x=x_component, y=y_component, opacity=0.65,
            labels={
                x_component: f"{x_component} ({variance[components.index(x_component)]:.2f}% varianza)",
                y_component: f"{y_component} ({variance[components.index(y_component)]:.2f}% varianza)",
            },
            title=f"Proyección: {x_component} vs. {y_component}",
        )
        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.add_vline(x=0, line_color="gray", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cargas de los componentes principales (PC1–PC7)")
    selected_components = st.multiselect(
        "Componentes para mostrar", components, default=components
    )
    max_top = min(30, len(X.columns))
    top_n = st.slider(
        "Cantidad de variables por gráfico", min_value=min(5, max_top),
        max_value=max_top, value=min(15, max_top)
    )
    if selected_components:
        chart_cols = st.columns(2)
        for i, component in enumerate(selected_components):
            values = loadings_df[component].sort_values(
                key=lambda series: series.abs(), ascending=False
            ).head(top_n).sort_values()
            fig = px.bar(
                x=values.values, y=values.index, orientation="h",
                labels={"x": "Carga", "y": "Variable"},
                title=f"Cargas de {component}",
                color=values.values,
                color_continuous_scale=["indianred", "lightgray", "steelblue"],
            )
            fig.add_vline(x=0, line_color="black", line_width=1)
            fig.update_layout(coloraxis_showscale=False)
            chart_cols[i % 2].plotly_chart(fig, use_container_width=True)

except Exception as error:
    st.error(f"No se pudo cargar o procesar el dataset: {error}")
    st.info(
        "Compruebe la conexión a internet. Si el error persiste, UCI puede estar temporalmente "
        "inaccesible; vuelva a intentar más tarde."
    )
