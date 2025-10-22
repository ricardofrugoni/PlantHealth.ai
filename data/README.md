# Dataset

Este diretorio contem os dados para treinar e testar o modelo PlantHealth.ai.

## PlantVillage Dataset

O projeto utiliza o PlantVillage Dataset.

## Download

Opcao 1: Kaggle (Recomendado)

```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/
```

Opcao 2: Download Manual

1. Acessar: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Clicar em "Download"
3. Extrair em data/raw/

## Estrutura de Diretorios

```
data/
├── raw/
│   └── plantvillage dataset/
│       └── color/
│           ├── train/
│           │   ├── Apple___Apple_scab/
│           │   ├── Apple___Black_rot/
│           │   ├── Apple___Cedar_apple_rust/
│           │   ├── Apple___healthy/
│           │   ├── Corn___Common_rust/
│           │   ├── Corn___Gray_leaf_spot/
│           │   ├── Corn___healthy/
│           │   ├── Corn___Northern_Leaf_Blight/
│           │   ├── Grape___Black_rot/
│           │   ├── Grape___Esca_(Black_Measles)/
│           │   ├── Grape___healthy/
│           │   ├── Grape___Leaf_blight/
│           │   ├── Potato___Early_blight/
│           │   ├── Potato___healthy/
│           │   ├── Potato___Late_blight/
│           │   └── Tomato___healthy/
│           └── test/
└── processed/
```

## Estatisticas

| Metrica | Valor |
|---------|-------|
| Total de Imagens | aproximadamente 54,000 |
| Imagens de Treino | aproximadamente 43,000 (80%) |
| Imagens de Teste | aproximadamente 11,000 (20%) |
| Numero de Classes | 16 |
| Formato | JPG/JPEG |
| Resolucao Processada | 96x96 |
| Espaco em Disco | aproximadamente 3 GB |

## Classes

16 classes total:

Maca (Apple) - 4 classes:
1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy

Milho (Corn) - 4 classes:
5. Corn___Common_rust
6. Corn___Gray_leaf_spot
7. Corn___healthy
8. Corn___Northern_Leaf_Blight

Uva (Grape) - 4 classes:
9. Grape___Black_rot
10. Grape___Esca_(Black_Measles)
11. Grape___healthy
12. Grape___Leaf_blight

Batata (Potato) - 3 classes:
13. Potato___Early_blight
14. Potato___healthy
15. Potato___Late_blight

Tomate (Tomato) - 1 classe:
16. Tomato___healthy

## Preprocessamento

Automaticamente aplicado durante treinamento:

1. Redimensionamento:
```python
IMG_SIZE = (96, 96)
image = image.resize(IMG_SIZE)
```

2. Normalizacao:
```python
image = image / 255.0
```

3. Data Augmentation (Apenas Treino):
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)
```

Transformacoes:
- Rotacao: mais ou menos 15 graus
- Flip horizontal: 50% de chance
- Zoom: mais ou menos 10%
- Shift horizontal/vertical: mais ou menos 10%

## Uso no Treinamento

Carregar Dados:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/raw/plantvillage dataset/color/train',
    target_size=(96, 96),
    batch_size=256,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/raw/plantvillage dataset/color/train',
    target_size=(96, 96),
    batch_size=256,
    class_mode='categorical',
    subset='validation'
)
```

## Citacao

Se usar este dataset, cite:

```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

## Licenca

O PlantVillage Dataset esta disponivel sob licenca Creative Commons.

Dataset original: https://plantvillage.psu.edu

## Troubleshooting

Erro: Diretorio nao encontrado

Solucao: Baixe e extraia o dataset conforme instrucoes acima.

Erro: Sem imagens encontradas

Solucao: Verifique se:
1. Dataset foi extraido corretamente
2. Estrutura de diretorios esta correta
3. Imagens estao em formato JPG/JPEG/PNG

---

Dataset: PlantVillage (Hughes & Salathe, 2015)
