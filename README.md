
# Fall Detector Engine

## 🧠 Motor de Detección de Caídas

Este repositorio contiene el **motor de inferencia** para la detección de caídas en videos, utilizando el modelo previamente entrenado en el repositorio [`fall-detector-training`](https://github.com/tu-usuario/fall-detector-training). Aquí se encuentra la lógica encargada de:

- Cargar y ejecutar el modelo `.keras`.
- Preprocesar los videos de entrada (redimensionado, segmentación en buffers, etc.).
- Realizar la predicción sobre secuencias de frames.
- Generar métricas de rendimiento sobre los datos de test.
- Exportar resultados para su análisis.

---

## 📦 Estructura del Proyecto

fall-detector-engine/
├── keras/ # Contiene el modelo entrenado (.keras)
├── src/
│ ├── config/ # Configuraciones y rutas (.env)
│ ├── preprocessing/ # Utilidades para procesamiento de video
│ └── detection/ # Lógica de inferencia y evaluación
├── resultados/ # Reportes generados tras las predicciones
├── run_demo.py # Script principal de ejecución
├── requirements.txt # Dependencias necesarias
└── .env # Variables de entorno (rutas locales)

---

## ⚙️ Requisitos

- Python 3.11+
- Entorno virtual (recomendado)
