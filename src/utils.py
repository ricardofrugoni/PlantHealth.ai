"""
PlantHealth AI - Funções Utilitárias
Conjunto de funções auxiliares para processamento, visualização e predição
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import warnings

# Import condicional para pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    warnings.warn("⚠️  Pandas não instalado. Função generate_report pode não funcionar corretamente.")


# ============================================================================
# FUNÇÕES DE CARREGAMENTO
# ============================================================================

def load_model_and_classes(model_path: str, class_mapping_path: str) -> Tuple[keras.Model, Dict[int, str]]:
    """
    Carrega o modelo treinado e o mapeamento de classes
    
    Args:
        model_path: Caminho para o arquivo .keras
        class_mapping_path: Caminho para o arquivo class_mapping.json
        
    Returns:
        tuple: (model, class_names_dict)
    """
    # Carregar modelo
    model = keras.models.load_model(model_path)
    
    # Carregar classes
    with open(class_mapping_path, 'r') as f:
        class_names = json.load(f)
        class_names = {int(k): v for k, v in class_names.items()}
    
    return model, class_names


# ============================================================================
# FUNÇÕES DE PREPROCESSAMENTO
# ============================================================================

def preprocess_image(image_path: Union[str, Path, np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocessa imagem para predição
    
    Args:
        image_path: Caminho da imagem ou array numpy
        target_size: Tupla (height, width)
        
    Returns:
        numpy.ndarray: Imagem preprocessada
    """
    if isinstance(image_path, (str, Path)):
        img = Image.open(image_path)
    elif isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)
    else:
        img = image_path
    
    # Converter para RGB se necessário
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img = img.resize(target_size)
    
    # Converter para array e normalizar
    img_array = np.array(img) / 255.0
    
    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ============================================================================
# FUNÇÕES DE PREDIÇÃO
# ============================================================================

def predict_disease(model: keras.Model, image_path: Union[str, Path], 
                   class_names: Dict[int, str], top_k: int = 3) -> Dict:
    """
    Faz predição de doença em uma imagem
    
    Args:
        model: Modelo Keras treinado
        image_path: Caminho da imagem
        class_names: Dicionário de nomes das classes
        top_k: Número de predições top a retornar
        
    Returns:
        dict: Resultados da predição
    """
    # Preprocessar imagem
    img_array = preprocess_image(image_path)
    
    # Fazer predição
    predictions = model.predict(img_array, verbose=0)
    
    # Obter top-k predições
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    results = {
        'primary_class': class_names[int(top_indices[0])],
        'confidence': float(predictions[0][top_indices[0]]),
        'top_predictions': [
            {
                'class': class_names[int(idx)],
                'confidence': float(predictions[0][idx]),
                'index': int(idx)
            }
            for idx in top_indices
        ],
        'all_probabilities': predictions[0].tolist()
    }
    
    return results


def batch_predict(model: keras.Model, image_paths: List[Union[str, Path]], 
                 class_names: Dict[int, str], batch_size: int = 32) -> List[Dict]:
    """
    Faz predições em lote para múltiplas imagens
    
    Args:
        model: Modelo Keras treinado
        image_paths: Lista de caminhos das imagens
        class_names: Dicionário de nomes das classes
        batch_size: Tamanho do lote
        
    Returns:
        list: Lista de dicionários com resultados
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = np.vstack([preprocess_image(path) for path in batch_paths])
        
        predictions = model.predict(batch_images, verbose=0)
        
        for j, pred in enumerate(predictions):
            pred_class = np.argmax(pred)
            results.append({
                'image_path': str(batch_paths[j]),
                'predicted_class': class_names[pred_class],
                'confidence': float(pred[pred_class]),
                'all_probabilities': pred.tolist()
            })
    
    return results


# ============================================================================
# FUNÇÕES DE GRAD-CAM
# ============================================================================

def make_gradcam_heatmap(img_array: np.ndarray, model: keras.Model, 
                        last_conv_layer_name: str, pred_index: int = None) -> np.ndarray:
    """
    Gera heatmap Grad-CAM para uma imagem
    
    Args:
        img_array: Array da imagem (1, H, W, C)
        model: Modelo Keras
        last_conv_layer_name: Nome da última camada convolucional
        pred_index: Índice da classe (None para usar a predita)
        
    Returns:
        numpy.ndarray: Heatmap normalizado
    """
    # Criar modelo que mapeia input → (última conv layer output, predições)
    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Computar gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradientes da classe predita em relação à última conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Média dos gradientes (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal pelos seus gradientes
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalizar heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def overlay_gradcam(img_path: Union[str, Path], heatmap: np.ndarray, 
                   alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobrepõe heatmap Grad-CAM na imagem original
    
    Args:
        img_path: Caminho da imagem
        heatmap: Heatmap Grad-CAM
        alpha: Intensidade da sobreposição
        colormap: Colormap OpenCV
        
    Returns:
        tuple: (original, heatmap_colored, superimposed)
    """
    # Carregar imagem original
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    
    # Redimensionar heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Converter heatmap para RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Sobrepor heatmap
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return img.astype(np.uint8), heatmap, superimposed_img


def find_last_conv_layer(model: keras.Model) -> str:
    """
    Encontra automaticamente a última camada convolucional do modelo
    
    Args:
        model: Modelo Keras
        
    Returns:
        str: Nome da última camada convolucional
    """
    # Procurar no modelo principal
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    
    # Procurar no base model
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # É um modelo dentro de outro
            for sublayer in reversed(layer.layers):
                if 'conv' in sublayer.name.lower():
                    return sublayer.name
    
    raise ValueError("Nenhuma camada convolucional encontrada no modelo")


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_prediction_with_gradcam(img_path: Union[str, Path], model: keras.Model, 
                                class_names: Dict[int, str], 
                                last_conv_layer_name: str = None,
                                figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plota imagem original, predição e Grad-CAM lado a lado
    
    Args:
        img_path: Caminho da imagem
        model: Modelo Keras
        class_names: Dicionário de nomes das classes
        last_conv_layer_name: Nome da última camada conv (None para auto-detectar)
        figsize: Tamanho da figura
    """
    # Fazer predição
    result = predict_disease(model, img_path, class_names, top_k=3)
    
    # Encontrar última camada conv se não fornecida
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
    
    # Gerar Grad-CAM
    img_array = preprocess_image(img_path)
    pred_index = result['top_predictions'][0]['index']
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    # Criar visualização
    original, heatmap_img, superimposed = overlay_gradcam(img_path, heatmap)
    
    # Plotar
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Imagem original
    axes[0].imshow(original)
    axes[0].set_title('Imagem Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_img)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Sobreposição
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Predição: {result["primary_class"]}\n'
                     f'Confiança: {result["confidence"]:.2%}',
                     fontsize=12, fontweight='bold', color='green' if result["confidence"] > 0.9 else 'orange')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir top-3 predições
    print("\n📊 Top-3 Predições:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"   {i}. {pred['class']}: {pred['confidence']:.2%}")


def plot_training_history(history: Dict, figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plota curvas de treinamento (loss e acurácia)
    
    Args:
        history: Dicionário com histórico do treinamento
        figsize: Tamanho da figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Evolução do Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Evolução da Acurácia', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None,
                         normalize: bool = True,
                         figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plota matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Lista de nomes das classes
        normalize: Se True, normaliza a matriz
        figsize: Tamanho da figura
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', 
                cmap='Blues', cbar_kws={'label': 'Proporção' if normalize else 'Contagem'})
    
    if class_names and len(class_names) <= 20:  # Só mostrar labels se houver poucas classes
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90, ha='right')
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.xlabel('Classe Predita', fontsize=12, fontweight='bold')
    plt.ylabel('Classe Real', fontsize=12, fontweight='bold')
    plt.title('Matriz de Confusão' + (' (Normalizada)' if normalize else ''), 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNÇÕES DE TRATAMENTO E RECOMENDAÇÕES
# ============================================================================

DISEASE_TREATMENTS = {
    'early_blight': {
        'description': 'Pinta Preta / Early Blight',
        'fungicides': ['Clorotalonil', 'Mancozeb', 'Azoxistrobina'],
        'cultural_practices': [
            'Rotação de culturas',
            'Remoção de folhas infectadas',
            'Evitar irrigação por aspersão',
            'Espaçamento adequado entre plantas'
        ],
        'prevention': 'Aplicar fungicidas preventivamente em condições úmidas'
    },
    'late_blight': {
        'description': 'Requeima / Late Blight',
        'fungicides': ['Metalaxil', 'Dimetomorf', 'Fluopicolida'],
        'cultural_practices': [
            'Destruir restos culturais',
            'Plantar tubérculos sadios',
            'Evitar excesso de umidade',
            'Monitoramento constante'
        ],
        'prevention': 'Doença de rápida disseminação, requer ação imediata'
    },
    'healthy': {
        'description': 'Planta Saudável',
        'fungicides': [],
        'cultural_practices': [
            'Manter práticas de manejo atuais',
            'Monitoramento regular',
            'Nutrição adequada',
            'Controle preventivo de pragas'
        ],
        'prevention': 'Continuar com boas práticas agrícolas'
    }
}


def get_treatment_recommendation(disease_name: str) -> Dict:
    """
    Retorna recomendações de tratamento para uma doença
    
    Args:
        disease_name: Nome da doença
        
    Returns:
        dict: Recomendações de tratamento
    """
    # Tentar encontrar a doença no dicionário
    for key in DISEASE_TREATMENTS:
        if key in disease_name.lower():
            return DISEASE_TREATMENTS[key]
    
    # Retorno padrão se não encontrar
    return {
        'description': disease_name,
        'fungicides': ['Consultar agrônomo para recomendação específica'],
        'cultural_practices': [
            'Monitorar regularmente a plantação',
            'Remover plantas infectadas',
            'Evitar excesso de umidade'
        ],
        'prevention': 'Consultar profissional especializado'
    }


def generate_report(img_path: Union[str, Path], result: Dict) -> str:
    """
    Gera relatório textual da análise
    
    Args:
        img_path: Caminho da imagem
        result: Resultado da predição
        
    Returns:
        str: Relatório formatado
    """
    from datetime import datetime
    
    disease = result['primary_class']
    confidence = result['confidence']
    treatment = get_treatment_recommendation(disease)
    
    # Usar pandas se disponível, senão datetime
    if PANDAS_AVAILABLE and pd is not None:
        current_date = pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')
    else:
        current_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    
    report = f"""
{'='*70}
RELATÓRIO DE DIAGNÓSTICO FITOSSANITÁRIO - PlantHealth AI
{'='*70}

📁 Imagem: {Path(img_path).name}
📅 Data: {current_date}

{'='*70}
DIAGNÓSTICO
{'='*70}

🦠 Doença Identificada: {disease}
📊 Confiança: {confidence:.2%}

Top-3 Possibilidades:
"""
    
    for i, pred in enumerate(result['top_predictions'], 1):
        report += f"   {i}. {pred['class']}: {pred['confidence']:.2%}\n"
    
    report += f"""
{'='*70}
RECOMENDAÇÕES DE TRATAMENTO
{'='*70}

📋 Descrição: {treatment['description']}

💊 Fungicidas/Pesticidas Recomendados:
"""
    
    for fungicide in treatment['fungicides']:
        report += f"   • {fungicide}\n"
    
    report += f"""
🌾 Práticas Culturais:
"""
    
    for practice in treatment['cultural_practices']:
        report += f"   • {practice}\n"
    
    report += f"""
⚠️  Prevenção: {treatment['prevention']}

{'='*70}
AVISO: Este é um diagnóstico automatizado. Consulte um agrônomo
para confirmação e orientações específicas para sua região.
{'='*70}
"""
    
    return report


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando se necessário
    
    Args:
        directory: Caminho do diretório
        
    Returns:
        Path: Caminho do diretório
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_results_to_json(results: Dict, output_path: Union[str, Path]) -> None:
    """
    Salva resultados em arquivo JSON
    
    Args:
        results: Dicionário com resultados
        output_path: Caminho do arquivo de saída
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados salvos em: {output_path}")



