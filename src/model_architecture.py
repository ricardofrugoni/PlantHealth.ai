"""
PlantHealth AI - Arquitetura dos Modelos
Definição de arquiteturas de modelos para classificação de doenças em plantas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB4, EfficientNetB5, EfficientNetB6,
    ResNet50, ResNet101, InceptionV3, MobileNetV2
)
from typing import Tuple, Optional


# ============================================================================
# MODELO BASE: EFFICIENTNET
# ============================================================================

def create_efficientnet_model(
    num_classes: int,
    img_size: Tuple[int, int, int] = (224, 224, 3),
    efficientnet_version: str = 'B4',
    dropout_rate: float = 0.4,
    dense_units: int = 512,
    freeze_backbone: bool = True
) -> Tuple[keras.Model, keras.Model]:
    """
    Cria modelo baseado em EfficientNet com transfer learning
    
    Args:
        num_classes: Número de classes de saída
        img_size: Tamanho da imagem de entrada (height, width, channels)
        efficientnet_version: Versão do EfficientNet ('B4', 'B5', 'B6')
        dropout_rate: Taxa de dropout
        dense_units: Número de unidades na camada densa
        freeze_backbone: Se True, congela o backbone inicialmente
        
    Returns:
        tuple: (model, base_model)
    """
    # Selecionar versão do EfficientNet
    efficientnet_models = {
        'B4': EfficientNetB4,
        'B5': EfficientNetB5,
        'B6': EfficientNetB6
    }
    
    if efficientnet_version not in efficientnet_models:
        raise ValueError(f"Versão inválida: {efficientnet_version}. Use: B4, B5 ou B6")
    
    # Base model - EfficientNet pré-treinado no ImageNet
    EfficientNetClass = efficientnet_models[efficientnet_version]
    base_model = EfficientNetClass(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='avg'
    )
    
    # Congelar camadas iniciais
    base_model.trainable = not freeze_backbone
    
    # Construir modelo completo
    inputs = layers.Input(shape=img_size)
    
    # Normalização adicional para EfficientNet
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Dropout para regularização
    x = layers.Dropout(dropout_rate * 0.75)(x)
    
    # Camada densa intermediária
    x = layers.Dense(dense_units, activation='relu', name='dense_features')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Camada de saída - classificação
    outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'PlantHealth_EfficientNet{efficientnet_version}')
    
    return model, base_model


# ============================================================================
# MODELO ALTERNATIVO: RESNET
# ============================================================================

def create_resnet_model(
    num_classes: int,
    img_size: Tuple[int, int, int] = (224, 224, 3),
    resnet_version: str = 'ResNet50',
    dropout_rate: float = 0.4,
    dense_units: int = 512
) -> keras.Model:
    """
    Cria modelo baseado em ResNet
    
    Args:
        num_classes: Número de classes de saída
        img_size: Tamanho da imagem de entrada
        resnet_version: Versão do ResNet ('ResNet50' ou 'ResNet101')
        dropout_rate: Taxa de dropout
        dense_units: Número de unidades na camada densa
        
    Returns:
        keras.Model: Modelo compilado
    """
    resnet_models = {
        'ResNet50': ResNet50,
        'ResNet101': ResNet101
    }
    
    ResNetClass = resnet_models.get(resnet_version, ResNet50)
    
    # Base model
    base_model = ResNetClass(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='avg'
    )
    
    base_model.trainable = False
    
    # Modelo completo
    inputs = layers.Input(shape=img_size)
    x = keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'PlantHealth_{resnet_version}')
    
    return model


# ============================================================================
# MODELO MULTI-TASK: CLASSIFICAÇÃO + SEVERIDADE
# ============================================================================

def create_multitask_model(
    num_classes: int,
    img_size: Tuple[int, int, int] = (224, 224, 3),
    backbone: str = 'EfficientNetB4'
) -> keras.Model:
    """
    Cria modelo multi-task: classificação + estimativa de severidade
    
    Args:
        num_classes: Número de classes de saída
        img_size: Tamanho da imagem de entrada
        backbone: Nome do backbone a usar
        
    Returns:
        keras.Model: Modelo multi-task
    """
    # Backbone
    base_model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='avg'
    )
    
    base_model.trainable = False
    
    # Input
    inputs = layers.Input(shape=img_size)
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    # Features do backbone
    features = base_model(x, training=False)
    features = layers.Dropout(0.3)(features)
    
    # Camada compartilhada
    shared = layers.Dense(512, activation='relu', name='shared_dense')(features)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)
    
    # Head 1: Classificação (38 classes)
    classification = layers.Dense(256, activation='relu', name='class_dense')(shared)
    classification = layers.Dropout(0.3)(classification)
    classification_output = layers.Dense(
        num_classes, 
        activation='softmax', 
        name='disease_class'
    )(classification)
    
    # Head 2: Severidade (regressão 0-1)
    severity = layers.Dense(128, activation='relu', name='severity_dense')(shared)
    severity = layers.Dropout(0.3)(severity)
    severity_output = layers.Dense(
        1, 
        activation='sigmoid', 
        name='severity'
    )(severity)
    
    # Modelo
    model = models.Model(
        inputs=inputs,
        outputs=[classification_output, severity_output],
        name='PlantHealth_MultiTask'
    )
    
    return model


# ============================================================================
# MODELO COM ATTENTION
# ============================================================================

def create_attention_model(
    num_classes: int,
    img_size: Tuple[int, int, int] = (224, 224, 3)
) -> keras.Model:
    """
    Cria modelo com mecanismo de atenção
    
    Args:
        num_classes: Número de classes de saída
        img_size: Tamanho da imagem de entrada
        
    Returns:
        keras.Model: Modelo com attention
    """
    # Backbone
    base_model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=img_size
    )
    
    base_model.trainable = False
    
    # Input
    inputs = layers.Input(shape=img_size)
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    # Features do backbone (sem pooling)
    features = base_model(x, training=False)
    
    # Attention mechanism
    attention = layers.Conv2D(1, 1, activation='sigmoid', name='attention_weights')(features)
    attended_features = layers.Multiply()([features, attention])
    
    # Global pooling
    pooled = layers.GlobalAveragePooling2D()(attended_features)
    
    # Classificação
    x = layers.Dense(512, activation='relu')(pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='PlantHealth_Attention')
    
    return model


# ============================================================================
# MODELO MOBILE (PARA DEPLOY)
# ============================================================================

def create_mobile_model(
    num_classes: int,
    img_size: Tuple[int, int, int] = (224, 224, 3),
    alpha: float = 1.0
) -> keras.Model:
    """
    Cria modelo leve para deploy mobile
    
    Args:
        num_classes: Número de classes de saída
        img_size: Tamanho da imagem de entrada
        alpha: Width multiplier do MobileNet
        
    Returns:
        keras.Model: Modelo mobile
    """
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        alpha=alpha,
        pooling='avg'
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=img_size)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='PlantHealth_Mobile')
    
    return model


# ============================================================================
# FUNÇÕES DE COMPILAÇÃO
# ============================================================================

def compile_classification_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    metrics: Optional[list] = None
) -> keras.Model:
    """
    Compila modelo para classificação
    
    Args:
        model: Modelo Keras
        learning_rate: Taxa de aprendizado
        metrics: Lista de métricas customizadas
        
    Returns:
        keras.Model: Modelo compilado
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def compile_multitask_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    class_weight: float = 1.0,
    severity_weight: float = 0.5
) -> keras.Model:
    """
    Compila modelo multi-task
    
    Args:
        model: Modelo Keras multi-task
        learning_rate: Taxa de aprendizado
        class_weight: Peso da loss de classificação
        severity_weight: Peso da loss de severidade
        
    Returns:
        keras.Model: Modelo compilado
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'disease_class': 'categorical_crossentropy',
            'severity': 'mse'
        },
        loss_weights={
            'disease_class': class_weight,
            'severity': severity_weight
        },
        metrics={
            'disease_class': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)],
            'severity': ['mae', 'mse']
        }
    )
    
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(
    model_save_path: str,
    log_dir: str = './logs',
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3
) -> list:
    """
    Retorna lista de callbacks para treinamento
    
    Args:
        model_save_path: Caminho para salvar melhor modelo
        log_dir: Diretório para logs do TensorBoard
        early_stopping_patience: Paciência para early stopping
        reduce_lr_patience: Paciência para redução de LR
        
    Returns:
        list: Lista de callbacks
    """
    callbacks = [
        # ModelCheckpoint - salvar melhor modelo
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # EarlyStopping - parar se não melhorar
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # ReduceLROnPlateau - reduzir learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        
        # CSV Logger
        keras.callbacks.CSVLogger(
            f'{log_dir}/training_log.csv',
            separator=',',
            append=False
        )
    ]
    
    return callbacks


# ============================================================================
# FINE-TUNING
# ============================================================================

def unfreeze_model(model: keras.Model, base_model: keras.Model, 
                  num_layers_to_unfreeze: int = 50) -> keras.Model:
    """
    Descongela últimas camadas do base model para fine-tuning
    
    Args:
        model: Modelo completo
        base_model: Base model (backbone)
        num_layers_to_unfreeze: Número de camadas a descongelar
        
    Returns:
        keras.Model: Modelo com camadas descongeladas
    """
    base_model.trainable = True
    
    # Congelar apenas as primeiras camadas
    fine_tune_at = len(base_model.layers) - num_layers_to_unfreeze
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_layers = len([l for l in base_model.layers if l.trainable])
    print(f"✅ Fine-tuning ativado: {trainable_layers} camadas treináveis no backbone")
    
    return model


# ============================================================================
# UTILIDADES
# ============================================================================

def print_model_summary(model: keras.Model) -> None:
    """
    Imprime resumo detalhado do modelo
    
    Args:
        model: Modelo Keras
    """
    print("\n" + "="*70)
    print(f"RESUMO DO MODELO: {model.name}")
    print("="*70)
    
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "="*70)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Parâmetros não-treináveis: {non_trainable_params:,}")
    print("="*70 + "\n")


def save_model_architecture(model: keras.Model, output_path: str) -> None:
    """
    Salva arquitetura do modelo em JSON
    
    Args:
        model: Modelo Keras
        output_path: Caminho do arquivo de saída
    """
    import json
    
    architecture = model.to_json()
    
    with open(output_path, 'w') as f:
        f.write(architecture)
    
    print(f"✅ Arquitetura salva em: {output_path}")


def convert_to_tflite(model: keras.Model, output_path: str, 
                     quantize: bool = True) -> None:
    """
    Converte modelo para TFLite (mobile)
    
    Args:
        model: Modelo Keras
        output_path: Caminho do arquivo de saída
        quantize: Se True, aplica quantização
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calcular tamanho
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ Modelo TFLite salvo em: {output_path}")
    print(f"   Tamanho: {size_mb:.2f} MB")


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de arquitetura de modelos do PlantHealth AI")
    
    # Criar modelo exemplo
    model, base_model = create_efficientnet_model(
        num_classes=38,
        img_size=(224, 224, 3)
    )
    
    print_model_summary(model)

