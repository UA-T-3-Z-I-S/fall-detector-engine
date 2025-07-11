import os
from keras.models import load_model, Model
from keras.layers import LSTM
from keras import Input
from config.paths import MODEL_PATH, MODEL_PATH_CNN, MODEL_PATH_LSTM

print(f"🔍 Cargando modelo desde: {MODEL_PATH}")
modelo_completo = load_model(MODEL_PATH)

# =============================
# 📋 Inspección rápida de capas
# =============================
print("📋 Capas detectadas:")
for i, layer in enumerate(modelo_completo.layers):
    print(f"[{i}] {layer.name} | {type(layer).__name__}")

# -----------------------------
# Parte 1: Submodelo CNN (hasta TimeDistributed_3)
# -----------------------------
print("\n🔍 Extrayendo submodelo CNN...")

cnn_input = modelo_completo.input
cnn_output = modelo_completo.get_layer("time_distributed_3").output

modelo_cnn = Model(inputs=cnn_input, outputs=cnn_output)
modelo_cnn.save(MODEL_PATH_CNN)
print(f"✅ Submodelo CNN guardado en: {MODEL_PATH_CNN}")

# -----------------------------
# Parte 2: Submodelo LSTM + Dense
# -----------------------------
print("\n🔍 Extrayendo submodelo LSTM+Dense...")

embedding_size = cnn_output.shape[-1]
sequence_input = Input(shape=(None, embedding_size), name="input_lstm")

x = sequence_input
copiando = False
for layer in modelo_completo.layers:
    if layer.name == "lstm":
        copiando = True
    if copiando:
        x = layer(x)

modelo_lstm = Model(inputs=sequence_input, outputs=x)
modelo_lstm.save(MODEL_PATH_LSTM)
print(f"✅ Submodelo LSTM+Dense guardado en: {MODEL_PATH_LSTM}")
