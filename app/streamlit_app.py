"""
PlantHealth AI - Interface Web Interativa
Aplicação Streamlit para diagnóstico de doenças em plantas
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from utils import (
        preprocess_image, predict_disease, make_gradcam_heatmap,
        overlay_gradcam, find_last_conv_layer, get_treatment_recommendation
    )
except ImportError:
    st.error("⚠️ Não foi possível importar módulos auxiliares. Certifique-se de que o diretório src/ está no path.")


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES DE CACHE
# ============================================================================

@st.cache_resource
def load_model_cached(model_path):
    """Carrega modelo com cache"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None


@st.cache_data
def load_class_mapping_cached(mapping_path):
    """Carrega mapeamento de classes com cache"""
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
            return {int(k): v for k, v in class_names.items()}
    except Exception as e:
        st.error(f"❌ Erro ao carregar mapeamento de classes: {str(e)}")
        return None


# ============================================================================
# SIDEBAR - CONFIGURAÇÕES
# ============================================================================

def render_sidebar():
    """Renderiza sidebar com configurações"""
    st.sidebar.markdown("## ⚙️ Configurações")
    
    # Upload de modelo customizado
    st.sidebar.markdown("### 📦 Modelo")
    use_custom_model = st.sidebar.checkbox("Usar modelo customizado", value=False)
    
    if use_custom_model:
        model_file = st.sidebar.file_uploader("Upload do modelo (.keras)", type=['keras', 'h5'])
        mapping_file = st.sidebar.file_uploader("Upload do mapeamento (.json)", type=['json'])
        
        if model_file and mapping_file:
            # Salvar temporariamente
            temp_model_path = Path("temp_model.keras")
            temp_mapping_path = Path("temp_mapping.json")
            
            with open(temp_model_path, 'wb') as f:
                f.write(model_file.read())
            with open(temp_mapping_path, 'w') as f:
                f.write(mapping_file.getvalue().decode())
            
            return str(temp_model_path), str(temp_mapping_path)
    
    # Usar modelo padrão
    model_path = st.sidebar.text_input(
        "Caminho do modelo",
        value="models/plant_disease_model_final.keras"
    )
    
    mapping_path = st.sidebar.text_input(
        "Caminho do mapeamento",
        value="models/class_mapping.json"
    )
    
    # Configurações de predição
    st.sidebar.markdown("### 🎯 Predição")
    confidence_threshold = st.sidebar.slider(
        "Limiar de confiança",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    show_gradcam = st.sidebar.checkbox("Mostrar Grad-CAM", value=True)
    show_top_k = st.sidebar.slider("Top-K predições", min_value=1, max_value=5, value=3)
    
    # Informações
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Sobre")
    st.sidebar.info("""
    **PlantHealth AI** v1.0
    
    Sistema de diagnóstico de doenças em plantas usando Deep Learning.
    
    - 🌿 38 classes de doenças
    - 🎯 96%+ de acurácia
    - 🔥 Visualização Grad-CAM
    - 📊 Recomendações de tratamento
    """)
    
    return model_path, mapping_path, confidence_threshold, show_gradcam, show_top_k


# ============================================================================
# HEADER
# ============================================================================

def render_header():
    """Renderiza header da aplicação"""
    st.markdown('<h1 class="main-header">🌱 PlantHealth AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema Inteligente de Diagnóstico Fitossanitário</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌿 Classes", "38")
    with col2:
        st.metric("🎯 Acurácia", "96%+")
    with col3:
        st.metric("📸 Imagens Treinadas", "54K+")
    with col4:
        st.metric("⚡ Inferência", "~100ms")
    
    st.markdown("---")


# ============================================================================
# UPLOAD E PREDIÇÃO
# ============================================================================

def render_upload_section():
    """Renderiza seção de upload de imagem"""
    st.markdown("## 📤 Upload de Imagem")
    
    # Opções de entrada
    input_method = st.radio(
        "Escolha o método de entrada:",
        ["Upload de arquivo", "Câmera", "URL"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "Upload de arquivo":
        uploaded_file = st.file_uploader(
            "Escolha uma imagem de folha",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos suportados: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
    
    elif input_method == "Câmera":
        camera_image = st.camera_input("Tire uma foto da folha")
        if camera_image:
            image = Image.open(camera_image)
    
    elif input_method == "URL":
        url = st.text_input("Digite a URL da imagem:")
        if url:
            try:
                import requests
                from io import BytesIO
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"❌ Erro ao carregar imagem da URL: {str(e)}")
    
    return image


def render_prediction_results(image, model, class_names, confidence_threshold, show_gradcam, show_top_k):
    """Renderiza resultados da predição"""
    st.markdown("## 🔍 Análise e Diagnóstico")
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Imagem Original")
        st.image(image, use_column_width=True)
        
        # Informações da imagem
        st.markdown("#### ℹ️ Informações da Imagem")
        st.write(f"- **Dimensões:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"- **Formato:** {image.format}")
        st.write(f"- **Modo:** {image.mode}")
    
    with col2:
        with st.spinner("🔄 Analisando imagem..."):
            # Fazer predição
            result = predict_disease(model, image, class_names, top_k=show_top_k)
            
            primary_class = result['primary_class']
            confidence = result['confidence']
            
            # Mostrar resultado principal
            st.markdown("### 🦠 Diagnóstico")
            
            if confidence >= confidence_threshold:
                st.markdown(f'<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"#### ✅ {primary_class}")
                st.markdown(f"**Confiança:** {confidence:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(f"#### ⚠️ {primary_class}")
                st.markdown(f"**Confiança:** {confidence:.2%}")
                st.markdown("**Aviso:** Confiança baixa. Recomenda-se verificação manual.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Top-K predições
            st.markdown(f"### 📊 Top-{show_top_k} Predições")
            
            for i, pred in enumerate(result['top_predictions'], 1):
                progress = pred['confidence']
                st.write(f"**{i}. {pred['class']}**")
                st.progress(progress)
                st.caption(f"Confiança: {progress:.2%}")
    
    # Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.markdown("## 🔥 Grad-CAM: Visualização de Atenção")
        
        with st.spinner("⚡ Gerando Grad-CAM..."):
            try:
                # Encontrar última camada convolucional
                last_conv_layer = find_last_conv_layer(model)
                
                # Preparar imagem
                img_array = preprocess_image(image)
                pred_index = result['top_predictions'][0]['index']
                
                # Gerar heatmap
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)
                
                # Salvar imagem temporariamente
                temp_img_path = Path("temp_image.jpg")
                image.save(temp_img_path)
                
                # Overlay
                original, heatmap_img, superimposed = overlay_gradcam(temp_img_path, heatmap)
                
                # Mostrar
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Original")
                    st.image(original, use_column_width=True)
                
                with col2:
                    st.markdown("#### Heatmap")
                    st.image(heatmap_img, use_column_width=True)
                
                with col3:
                    st.markdown("#### Sobreposição")
                    st.image(superimposed, use_column_width=True)
                
                st.info("💡 As regiões em vermelho/amarelo indicam onde o modelo está focando para fazer o diagnóstico.")
                
                # Limpar arquivo temporário
                if temp_img_path.exists():
                    temp_img_path.unlink()
                
            except Exception as e:
                st.error(f"❌ Erro ao gerar Grad-CAM: {str(e)}")
    
    # Recomendações de tratamento
    st.markdown("---")
    st.markdown("## 💊 Recomendações de Tratamento")
    
    treatment = get_treatment_recommendation(primary_class)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Fungicidas/Pesticidas")
        if treatment['fungicides']:
            for fungicide in treatment['fungicides']:
                st.markdown(f"- {fungicide}")
        else:
            st.success("✅ Planta saudável - nenhum tratamento necessário")
    
    with col2:
        st.markdown("### 🌾 Práticas Culturais")
        for practice in treatment['cultural_practices']:
            st.markdown(f"- {practice}")
    
    st.markdown("### ⚠️ Prevenção")
    st.info(treatment['prevention'])
    
    # Aviso legal
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **⚠️ AVISO IMPORTANTE:**
    
    Este é um diagnóstico automatizado baseado em Deep Learning. 
    As recomendações são genéricas e devem ser validadas por um profissional qualificado.
    
    **Sempre consulte um agrônomo ou engenheiro agrícola para:**
    - Confirmação do diagnóstico
    - Dosagens específicas de produtos
    - Orientações adaptadas à sua região e cultura
    - Manejo integrado de pragas e doenças
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botão de download do relatório
    st.markdown("---")
    if st.button("📄 Gerar Relatório PDF"):
        st.info("🚧 Funcionalidade de geração de PDF em desenvolvimento...")


# ============================================================================
# PÁGINA DE ESTATÍSTICAS
# ============================================================================

def render_stats_page(model, class_names):
    """Renderiza página de estatísticas do modelo"""
    st.markdown("## 📈 Estatísticas do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Arquitetura")
        st.write(f"**Nome:** {model.name}")
        st.write(f"**Total de parâmetros:** {model.count_params():,}")
        st.write(f"**Parâmetros treináveis:** {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        st.write(f"**Número de camadas:** {len(model.layers)}")
    
    with col2:
        st.markdown("### 📊 Dataset")
        st.write(f"**Número de classes:** {len(class_names)}")
        st.write(f"**Backbone:** EfficientNetB4")
        st.write(f"**Input size:** 224x224x3")
        st.write(f"**Acurácia:** 96%+")
    
    # Classes
    st.markdown("### 🌿 Classes Disponíveis")
    
    # Organizar em colunas
    num_cols = 3
    classes_per_col = len(class_names) // num_cols + 1
    
    cols = st.columns(num_cols)
    
    for i, (idx, class_name) in enumerate(sorted(class_names.items())):
        col_idx = i // classes_per_col
        with cols[col_idx]:
            st.write(f"{idx}. {class_name}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Função principal da aplicação"""
    
    # Sidebar
    model_path, mapping_path, confidence_threshold, show_gradcam, show_top_k = render_sidebar()
    
    # Header
    render_header()
    
    # Verificar se os arquivos existem
    if not Path(model_path).exists():
        st.error(f"❌ Modelo não encontrado: {model_path}")
        st.info("💡 Certifique-se de treinar o modelo primeiro executando os notebooks de treinamento.")
        return
    
    if not Path(mapping_path).exists():
        st.error(f"❌ Mapeamento de classes não encontrado: {mapping_path}")
        return
    
    # Carregar modelo e classes
    with st.spinner("⏳ Carregando modelo..."):
        model = load_model_cached(model_path)
        class_names = load_class_mapping_cached(mapping_path)
    
    if model is None or class_names is None:
        st.error("❌ Erro ao carregar modelo ou mapeamento de classes.")
        return
    
    st.success(f"✅ Modelo carregado com sucesso! ({model.name})")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Diagnóstico", "📈 Estatísticas", "❓ Ajuda"])
    
    with tab1:
        # Upload
        image = render_upload_section()
        
        if image:
            # Predição
            render_prediction_results(
                image, model, class_names,
                confidence_threshold, show_gradcam, show_top_k
            )
        else:
            st.info("👆 Faça upload de uma imagem para começar o diagnóstico")
            
            # Mostrar imagens de exemplo
            st.markdown("### 📸 Exemplos de Imagens Válidas")
            st.markdown("""
            - Foto clara da folha
            - Boa iluminação
            - Folha ocupando a maior parte da imagem
            - Foco nítido
            - Fundo simples (preferencialmente)
            """)
    
    with tab2:
        render_stats_page(model, class_names)
    
    with tab3:
        st.markdown("## ❓ Ajuda e Documentação")
        
        st.markdown("""
        ### 🎯 Como Usar
        
        1. **Upload da Imagem**
           - Escolha uma foto clara de uma folha
           - Formatos aceitos: JPG, JPEG, PNG
           - Tamanho recomendado: mínimo 224x224 pixels
        
        2. **Análise**
           - O sistema automaticamente analisa a imagem
           - Resultado em segundos
           - Confiança da predição é mostrada
        
        3. **Grad-CAM**
           - Visualização de onde o modelo está "olhando"
           - Áreas vermelhas = maior atenção
           - Ajuda a validar o diagnóstico
        
        4. **Recomendações**
           - Sugestões de tratamento
           - Práticas culturais
           - Métodos de prevenção
        
        ### 🔬 Tecnologias
        
        - **Framework:** TensorFlow + Keras
        - **Arquitetura:** EfficientNetB4
        - **Técnica:** Transfer Learning
        - **Explicabilidade:** Grad-CAM
        - **Interface:** Streamlit
        
        ### 📚 Documentação Completa
        
        Visite o [GitHub do projeto](https://github.com/ricardofrugoni/planthealth-ai) 
        para documentação completa, notebooks de treinamento e mais informações.
        
        ### 🐛 Problemas ou Sugestões?
        
        Abra uma issue no GitHub ou entre em contato pelo email.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>Desenvolvido com ❤️ e 🌱 | PlantHealth AI v1.0</p>
        <p>© 2025 - Todos os direitos reservados</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

