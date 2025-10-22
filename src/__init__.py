"""
PlantHealth AI - Source Package
Sistema Inteligente de Detecção de Doenças em Plantas usando Deep Learning

Este pacote contém os módulos principais do projeto:
- data_preprocessing: Pré-processamento e augmentation de dados
- model_architecture: Arquiteturas de modelos (EfficientNet, ResNet, etc.)
- utils: Funções utilitárias para predição, visualização e Grad-CAM
"""

__version__ = "1.0.0"
__author__ = "Ricardo Frugoni"
__email__ = "ricardo@codex.ai"
__license__ = "MIT"

# Imports principais para facilitar o uso
try:
    from .utils import (
        load_model_and_classes,
        preprocess_image,
        predict_disease,
        make_gradcam_heatmap,
        overlay_gradcam,
        find_last_conv_layer,
        get_treatment_recommendation
    )
    
    from .model_architecture import (
        create_efficientnet_model,
        create_resnet_model,
        compile_classification_model,
        get_callbacks
    )
    
    from .data_preprocessing import (
        create_data_generators,
        create_tf_dataset,
        analyze_dataset
    )
    
    __all__ = [
        # Utils
        'load_model_and_classes',
        'preprocess_image',
        'predict_disease',
        'make_gradcam_heatmap',
        'overlay_gradcam',
        'find_last_conv_layer',
        'get_treatment_recommendation',
        # Models
        'create_efficientnet_model',
        'create_resnet_model',
        'compile_classification_model',
        'get_callbacks',
        # Data
        'create_data_generators',
        'create_tf_dataset',
        'analyze_dataset',
    ]

except ImportError as e:
    import warnings
    warnings.warn(f"Erro ao importar módulos: {e}")
