# Communities and Crime — gráficos PCA

Aplicación Streamlit que muestra exclusivamente:
- Proyecciones de observaciones en pares de componentes principales.
- Gráficos de cargas para PC1–PC7.

La aplicación intenta obtener el dataset mediante `ucimlrepo`. Si la API falla, descarga el archivo original directamente desde el archivo ZIP de UCI.

## Ejecutar localmente

Requiere Python 3.10 o posterior y conexión a internet.

```bash
python -m venv .venv
```

Activar el entorno:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Instalar dependencias y ejecutar:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abrir `http://localhost:8501` si el navegador no se abre automáticamente.

## Publicar con GitHub + Streamlit Community Cloud

1. Subir `app.py`, `requirements.txt` y `README.md` a la raíz de un repositorio GitHub.
2. Entrar en https://share.streamlit.io/ y conectar GitHub.
3. Seleccionar el repositorio, la rama y `app.py` como archivo principal.
4. Pulsar **Deploy**.

Fuente del dataset: UCI Machine Learning Repository, Communities and Crime Unnormalized (ID 211).
