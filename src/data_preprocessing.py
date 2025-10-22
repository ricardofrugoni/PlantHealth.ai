"""
PlantHealth AI - Módulo de Pré-processamento de Dados
Pipeline completo de preparação de dados para treinamento
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
import os
import warnings

# Import condicional para sklearn
try:
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("⚠️  scikit-learn não instalado. Algumas funções podem não funcionar.")


# ============================================================================
# CONFIGURAÇÕES DE DATA AUGMENTATION
# ============================================================================

def get_train_augmentation(rescale: float = 1./255, 
                          validation_split: float = 0.2) -> ImageDataGenerator:
    """
    Retorna data generator com augmentation para treino
    
    Args:
        rescale: Fator de normalização
        validation_split: Proporção dos dados para validação
        
    Returns:
        ImageDataGenerator: Gerador configurado
    """
    return ImageDataGenerator(
        rescale=rescale,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )


def get_validation_augmentation(rescale: float = 1./255,
                               validation_split: float = 0.2) -> ImageDataGenerator:
    """
    Retorna data generator para validação (apenas rescaling)
    
    Args:
        rescale: Fator de normalização
        validation_split: Proporção dos dados para validação
        
    Returns:
        ImageDataGenerator: Gerador configurado
    """
    return ImageDataGenerator(
        rescale=rescale,
        validation_split=validation_split
    )


def get_test_augmentation(rescale: float = 1./255) -> ImageDataGenerator:
    """
    Retorna data generator para teste (apenas rescaling)
    
    Args:
        rescale: Fator de normalização
        
    Returns:
        ImageDataGenerator: Gerador configurado
    """
    return ImageDataGenerator(rescale=rescale)


# ============================================================================
# CRIAÇÃO DE GERADORES
# ============================================================================

def create_data_generators(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42
) -> Tuple[keras.preprocessing.image.DirectoryIterator, 
           keras.preprocessing.image.DirectoryIterator]:
    """
    Cria geradores de dados para treino e validação
    
    Args:
        data_dir: Diretório com as imagens organizadas por classe
        img_size: Tamanho das imagens (height, width)
        batch_size: Tamanho do batch
        validation_split: Proporção dos dados para validação
        seed: Seed para reprodutibilidade
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    train_datagen = get_train_augmentation(validation_split=validation_split)
    val_datagen = get_validation_augmentation(validation_split=validation_split)
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )
    
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    
    return train_generator, validation_generator


def create_test_generator(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32
) -> keras.preprocessing.image.DirectoryIterator:
    """
    Cria gerador de dados para teste
    
    Args:
        data_dir: Diretório com as imagens organizadas por classe
        img_size: Tamanho das imagens (height, width)
        batch_size: Tamanho do batch
        
    Returns:
        DirectoryIterator: Gerador de teste
    """
    test_datagen = get_test_augmentation()
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator


# ============================================================================
# DATASET TF.DATA (ALTERNATIVA MAIS PERFORMÁTICA)
# ============================================================================

def create_tf_dataset(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
    augment: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
    """
    Cria tf.data.Dataset para treino e validação (mais eficiente)
    
    Args:
        data_dir: Diretório com as imagens organizadas por classe
        img_size: Tamanho das imagens (height, width)
        batch_size: Tamanho do batch
        validation_split: Proporção dos dados para validação
        seed: Seed para reprodutibilidade
        augment: Se True, aplica data augmentation no treino
        
    Returns:
        tuple: (train_dataset, val_dataset, class_names_dict)
    """
    # Carregar dataset do diretório
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Extrair nomes das classes
    class_names = {i: name for i, name in enumerate(train_ds.class_names)}
    
    # Normalização
    normalization_layer = keras.layers.Rescaling(1./255)
    
    # Aplicar normalização
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Data augmentation para treino
    if augment:
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomTranslation(0.2, 0.2),
        ])
        
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Otimizações de performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names


# ============================================================================
# ANÁLISE DE DADOS
# ============================================================================

def analyze_dataset(data_dir: str) -> Dict:
    """
    Analisa a estrutura e distribuição do dataset
    
    Args:
        data_dir: Diretório com as imagens organizadas por classe
        
    Returns:
        dict: Estatísticas do dataset
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Diretório não encontrado: {data_dir}")
    
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    class_counts = {}
    total_images = 0
    
    for class_name in classes:
        class_path = data_path / class_name
        # Contar arquivos de imagem
        image_files = list(class_path.glob('*.jpg')) + \
                     list(class_path.glob('*.JPG')) + \
                     list(class_path.glob('*.png')) + \
                     list(class_path.glob('*.PNG')) + \
                     list(class_path.glob('*.jpeg'))
        
        count = len(image_files)
        class_counts[class_name] = count
        total_images += count
    
    # Calcular estatísticas
    counts_array = np.array(list(class_counts.values()))
    
    stats = {
        'num_classes': len(classes),
        'total_images': total_images,
        'classes': classes,
        'class_counts': class_counts,
        'avg_images_per_class': float(np.mean(counts_array)),
        'std_images_per_class': float(np.std(counts_array)),
        'min_images_per_class': int(np.min(counts_array)),
        'max_images_per_class': int(np.max(counts_array)),
        'imbalance_ratio': float(np.max(counts_array) / np.min(counts_array))
    }
    
    return stats


def check_class_imbalance(class_counts: Dict[str, int], threshold: float = 2.0) -> bool:
    """
    Verifica se há desbalanceamento significativo entre as classes
    
    Args:
        class_counts: Dicionário com contagem de imagens por classe
        threshold: Razão máxima entre maior e menor classe
        
    Returns:
        bool: True se houver desbalanceamento significativo
    """
    counts = list(class_counts.values())
    ratio = max(counts) / min(counts)
    return ratio > threshold


def calculate_class_weights(class_counts: Dict[str, int]) -> Dict[int, float]:
    """
    Calcula pesos para balancear classes no treinamento
    
    Args:
        class_counts: Dicionário com contagem de imagens por classe
        
    Returns:
        dict: Pesos para cada classe (índice -> peso)
    
    Raises:
        ImportError: Se scikit-learn não estiver instalado
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn é necessário para esta função. "
            "Instale com: pip install scikit-learn"
        )
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    # Calcular pesos
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=np.repeat(np.arange(len(classes)), counts)
    )
    
    return {i: weight for i, weight in enumerate(class_weights)}


# ============================================================================
# SALVAMENTO E CARREGAMENTO DE METADADOS
# ============================================================================

def save_class_mapping(class_indices: Dict[str, int], output_path: str) -> None:
    """
    Salva mapeamento de classes em arquivo JSON
    
    Args:
        class_indices: Dicionário {nome_classe: índice}
        output_path: Caminho do arquivo de saída
    """
    # Inverter dicionário para {índice: nome_classe}
    class_names = {v: k for k, v in class_indices.items()}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Mapeamento de classes salvo em: {output_path}")


def load_class_mapping(mapping_path: str) -> Dict[int, str]:
    """
    Carrega mapeamento de classes de arquivo JSON
    
    Args:
        mapping_path: Caminho do arquivo de mapeamento
        
    Returns:
        dict: Dicionário {índice: nome_classe}
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        # Garantir que as chaves sejam inteiros
        class_names = {int(k): v for k, v in class_names.items()}
    
    return class_names


def save_dataset_metadata(stats: Dict, output_path: str) -> None:
    """
    Salva metadados do dataset em arquivo JSON
    
    Args:
        stats: Dicionário com estatísticas do dataset
        output_path: Caminho do arquivo de saída
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metadados do dataset salvos em: {output_path}")


# ============================================================================
# SPLIT PERSONALIZADO DE DADOS
# ============================================================================

def create_train_val_test_split(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """
    Cria split personalizado de dados em train/val/test
    
    Args:
        data_dir: Diretório original com as imagens por classe
        output_dir: Diretório de saída
        train_ratio: Proporção de treino
        val_ratio: Proporção de validação
        test_ratio: Proporção de teste
        seed: Seed para reprodutibilidade
    
    Raises:
        ImportError: Se scikit-learn não estiver instalado
    """
    import shutil
    
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn é necessário para esta função. "
            "Instale com: pip install scikit-learn"
        )
    
    np.random.seed(seed)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Criar diretórios
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    for class_name in classes:
        class_path = data_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.JPG'))
        
        # Criar splits
        train_imgs, temp_imgs = train_test_split(
            images, train_size=train_ratio, random_state=seed
        )
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            train_size=val_ratio/(val_ratio + test_ratio),
            random_state=seed
        )
        
        # Copiar imagens
        for split, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_class_dir = output_path / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in img_list:
                shutil.copy2(img_path, split_class_dir / img_path.name)
        
        print(f"✅ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    print(f"\n✅ Split concluído! Dados salvos em: {output_dir}")


# ============================================================================
# UTILIDADES
# ============================================================================

def visualize_augmentation(
    data_dir: str,
    class_name: str,
    num_augmentations: int = 9,
    img_size: Tuple[int, int] = (224, 224)
) -> None:
    """
    Visualiza exemplos de data augmentation para uma classe
    
    Args:
        data_dir: Diretório com as imagens
        class_name: Nome da classe
        num_augmentations: Número de augmentations a mostrar
        img_size: Tamanho das imagens
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    class_path = Path(data_dir) / class_name
    images = list(class_path.glob('*.jpg'))
    
    if not images:
        print(f"❌ Nenhuma imagem encontrada para a classe: {class_name}")
        return
    
    # Pegar uma imagem aleatória
    img_path = np.random.choice(images)
    img = Image.open(img_path).resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Criar gerador com augmentation
    datagen = get_train_augmentation(rescale=1.0)
    
    # Gerar augmentations
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i == 0:
            # Primeira imagem é a original
            ax.imshow(img)
            ax.set_title('Original', fontsize=12, fontweight='bold')
        else:
            # Outras são augmentations
            augmented = next(datagen.flow(img_array, batch_size=1))[0]
            ax.imshow(np.clip(augmented, 0, 1))
            ax.set_title(f'Augmentation {i}', fontsize=10)
        
        ax.axis('off')
    
    plt.suptitle(f'Data Augmentation - {class_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de pré-processamento de dados do PlantHealth AI")
    print("Use as funções deste módulo para preparar seus dados de treinamento")

