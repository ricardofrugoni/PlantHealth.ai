"""
PlantHealth AI - Interface Gradio Standalone
Versão standalone da interface Gradio para uso local ou Colab
"""

import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os
import warnings

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from utils import preprocess_image, predict_disease, make_gradcam_heatmap, overlay_gradcam, find_last_conv_layer, get_treatment_recommendation
    UTILS_AVAILABLE = True
except ImportError:
    warnings.warn("Aviso: Módulo utils não disponível. Usando funções locais.")
    UTILS_AVAILABLE = False
    
    def preprocess_image(image):
        """Fallback preprocess function quando utils não está disponível"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)


# Caminhos dos arquivos - usar variáveis de ambiente com fallback
MODEL_PATH = os.getenv('PLANTHEALTH_MODEL_PATH', 'models/plant_disease_model_final.keras')
CLASS_MAPPING_PATH = os.getenv('PLANTHEALTH_CLASS_MAPPING_PATH', 'models/class_mapping.json')
SERVER_PORT = int(os.getenv('PLANTHEALTH_PORT', '7860'))
SERVER_HOST = os.getenv('PLANTHEALTH_HOST', '0.0.0.0')
SHARE_GRADIO = os.getenv('PLANTHEALTH_SHARE', 'false').lower() == 'true'


def load_model_and_classes():
    """
    Carrega modelo e classes
    
    Returns:
        tuple: (model, class_names) ou (None, None) em caso de erro
    """
    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        
        if not Path(CLASS_MAPPING_PATH).exists():
            raise FileNotFoundError(f"Mapeamento de classes não encontrado em: {CLASS_MAPPING_PATH}")
        
        model = keras.models.load_model(MODEL_PATH)
        
        with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
            class_names = {int(k): v for k, v in class_names.items()}
        
        print(f"✅ Modelo carregado com sucesso: {MODEL_PATH}")
        print(f"✅ Classes carregadas: {len(class_names)} categorias")
        
        return model, class_names
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo/classes: {e}")
        return None, None


# Carregar modelo global
model, class_names = load_model_and_classes()


def diagnose(image, show_gradcam=True):
    """Realiza diagnóstico da planta"""
    if model is None or class_names is None:
        return "ERRO: Modelo não carregado!", None, None
    
    try:
        # Preprocessar
        img_array = preprocess_image(image)
        
        # Predição
        predictions = model.predict(img_array, verbose=0)
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        # Resultado principal
        pred_idx = top_3_idx[0]
        pred_name = class_names[pred_idx]
        confidence = predictions[0][pred_idx]
        
        # Texto de resultado
        result = f"""
# DIAGNÓSTICO

## Doença Detectada
**{pred_name}**

**Confiança:** {confidence:.2%}

## Top-3 Predições
"""
        for i, idx in enumerate(top_3_idx, 1):
            result += f"{i}. {class_names[idx]}: {predictions[0][idx]:.2%}\n"
        
        # Grad-CAM
        gradcam_img = None
        if show_gradcam and UTILS_AVAILABLE:
            try:
                last_conv = find_last_conv_layer(model)
                if last_conv:
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv, pred_idx)
                    gradcam_img = overlay_gradcam(image.resize((224, 224)), heatmap)
            except:
                pass
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(8, 4))
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        names = [class_names[i].split('___')[-1][:20] for i in top_5_idx]
        probs = [predictions[0][i] for i in top_5_idx]
        
        ax.barh(names, probs, color=['green', 'orange', 'red', 'gray', 'gray'], alpha=0.7)
        ax.set_xlabel('Confiança')
        ax.set_title('Top-5 Predições')
        ax.set_xlim([0, 1])
        plt.tight_layout()
        
        return result, gradcam_img, fig
        
    except Exception as e:
        return f"Erro: {str(e)}", None, None


# Interface Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as app:
    gr.Markdown("""
    # 🌱 PlantHealth AI
    ## Sistema de Diagnóstico de Doenças em Plantas
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload da Folha")
            gradcam_check = gr.Checkbox(label="Mostrar Grad-CAM", value=True)
            btn = gr.Button("Diagnosticar", variant="primary")
        
        with gr.Column():
            text_output = gr.Markdown()
            gradcam_output = gr.Image(label="Grad-CAM")
            plot_output = gr.Plot(label="Confiança")
    
    btn.click(diagnose, [image_input, gradcam_check], [text_output, gradcam_output, plot_output])


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🌱 PlantHealth AI - Iniciando Aplicação")
    print(f"{'='*60}")
    print(f"📦 Modelo: {MODEL_PATH}")
    print(f"📋 Classes: {CLASS_MAPPING_PATH}")
    print(f"🌐 Servidor: {SERVER_HOST}:{SERVER_PORT}")
    print(f"🔗 Compartilhar: {SHARE_GRADIO}")
    print(f"{'='*60}\n")
    
    app.launch(
        share=SHARE_GRADIO,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        show_error=True
    )

