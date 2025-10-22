# PlantHealth.ai

Sistema Inteligente de Deteccao de Doencas em Plantas usando Deep Learning e Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Sumario

- [Sobre o Projeto](#sobre-o-projeto)
- [Caracteristicas](#caracteristicas)
- [Demonstracao](#demonstracao)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Instalacao](#instalacao)
- [Como Usar](#como-usar)
- [Dataset](#dataset)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Treinamento](#treinamento)
- [Deploy](#deploy)
- [Contribuindo](#contribuindo)
- [Roadmap](#roadmap)
- [Licenca](#licenca)
- [Autor](#autor)
- [Agradecimentos](#agradecimentos)

---

## Sobre o Projeto

PlantHealth.ai e um sistema de classificacao automatica de doencas em plantas utilizando tecnicas avancadas de Deep Learning. O projeto implementa Transfer Learning com a arquitetura MobileNetV2 pre-treinada no ImageNet, permitindo identificar 16 classes diferentes de plantas saudaveis e doentes atraves da analise de imagens de folhas.

### Motivacao

A deteccao precoce de doencas em plantas e crucial para:
- Prevenir perdas economicas na agricultura
- Reduzir o uso de pesticidas
- Aumentar a produtividade agricola
- Facilitar o monitoramento em larga escala

---

## Caracteristicas

### Funcionalidades Principais

- Identificacao automatica de 16 classes de plantas e doencas
- Interface web interativa com Gradio
- Inferencia rapida (< 100ms por imagem)
- Modelo leve e otimizado (~14 MB)
- Funcionamento em CPU (nao requer GPU)

### Tecnologias Utilizadas

- Framework: TensorFlow 2.13+ / Keras
- Arquitetura: MobileNetV2
- Interface: Gradio 4.0+
- Linguagem: Python 3.10+
- Ambiente: Google Colab

### Especificacoes Tecnicas

- Input: Imagens RGB 96x96 pixels
- Output: Probabilidades para 16 classes
- Parametros: ~2.3M total (~65K treinaveis)
- Tamanho: ~14 MB
- Precisao: ~76% no conjunto de teste
- Tempo de Inferencia: < 100ms

---

## Demonstracao

### Interface Gradio

A aplicacao oferece uma interface web simples e intuitiva:

1. Upload de imagem da folha
2. Analise automatica
3. Resultados com top 5 predicoes e niveis de confianca

---

## Arquitetura do Modelo

### Estrutura da Rede Neural

```
Input Layer (96x96x3)
        |
        v
MobileNetV2 Base (Congelada)
  - Pre-treinada no ImageNet
  - 1.4M imagens, 1000 classes
        |
        v
GlobalAveragePooling2D
        |
        v
Dense Layer (64 neurons, ReLU)
        |
        v
Output Layer (16 neurons, Softmax)
```

### Transfer Learning

O modelo utiliza Transfer Learning para aproveitar features ja aprendidas:

Camadas Congeladas (MobileNetV2 Base):
- Parametros pre-treinados no ImageNet
- Extraem features de baixo nivel (bordas, texturas)
- Nao sao atualizadas durante o treinamento

Camadas Treinaveis (Custom Layers):
- GlobalAveragePooling2D: Reduz dimensionalidade
- Dense(64): Features especificas do dominio
- Dense(16): Classificacao final

### Detalhes Tecnicos

| Componente | Especificacao |
|------------|---------------|
| Arquitetura Base | MobileNetV2 |
| Pre-treinamento | ImageNet |
| Input Shape | (96, 96, 3) |
| Output Classes | 16 |
| Total Parametros | 2,323,712 |
| Parametros Treinaveis | 65,808 |
| Parametros Congelados | 2,257,904 |
| Tamanho do Modelo | 14.2 MB |

---

## Instalacao

### Prerequisitos

- Python 3.10 ou superior
- pip ou conda

### Instalacao Rapida

#### Opcao 1: Local

```bash
git clone https://github.com/ricardofrugoni/planthealth-ai.git
cd planthealth.ai

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app/gradio_app.py
```

#### Opcao 2: Google Colab

```python
!git clone https://github.com/ricardofrugoni/planthealth-ai.git
%cd planthealth.ai
!pip install -r requirements.txt
```

---

## Como Usar

### 1. Interface Gradio (Recomendado)

```bash
python app/gradio_app.py
```

Acessar no navegador: http://localhost:7860

### 2. Via Python Script

```python
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.models.load_model('models/planthealth_mobilenet_fast.keras')

def preprocess_image(image_path):
    image = Image.open(image_path).resize((96, 96))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

img_array = preprocess_image('sua_imagem.jpg')
predictions = model.predict(img_array)

class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Classe predita: {class_idx}")
print(f"Confianca: {confidence:.2%}")
```

### 3. Via Jupyter Notebook

```bash
jupyter notebook notebooks/planthealth_training.ipynb
```

---

## Dataset

### PlantVillage Dataset

O projeto utiliza um subconjunto do PlantVillage Dataset.

### Estatisticas

| Metrica | Valor |
|---------|-------|
| Total de Imagens | ~54,000 |
| Imagens de Treino | ~43,000 (80%) |
| Imagens de Teste | ~11,000 (20%) |
| Numero de Classes | 16 |
| Formato | JPG/JPEG |
| Resolucao Processada | 96x96 |

### Classes

16 classes de 5 tipos de plantas:

Maca (Apple): 4 classes
- Apple scab
- Black rot
- Cedar apple rust
- Healthy

Milho (Corn): 4 classes
- Common rust
- Gray leaf spot
- Healthy
- Northern leaf blight

Uva (Grape): 4 classes
- Black rot
- Esca (Black Measles)
- Healthy
- Leaf blight

Batata (Potato): 3 classes
- Early blight
- Healthy
- Late blight

Tomate (Tomato): 1 classe
- Healthy

### Download do Dataset

Opcao 1: Kaggle
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw/
```

Opcao 2: Manual
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

---

## Resultados

### Metricas de Performance

| Metrica | Treino | Validacao | Teste |
|---------|--------|-----------|-------|
| Acuracia | 92.3% | 78.5% | 75.8% |
| Loss | 0.2156 | 0.5823 | 0.6234 |

### Tempo de Execucao

| Operacao | Tempo |
|----------|-------|
| Inferencia (1 imagem) | < 100ms |
| Batch 32 imagens | ~2s |
| Epoca completa (treino) | ~3-4 min |

---

## Estrutura do Projeto

```
planthealth.ai/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── planthealth_training.ipynb
│
├── app/
│   ├── gradio_app.py
│   └── predict.py
│
├── models/
│   ├── README.md
│   └── planthealth_mobilenet_fast.keras
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── results/
│   └── visualizations/
│
└── src/
    ├── data/
    ├── models/
    └── utils/
```

---

## Treinamento

### Configuracoes de Treinamento

```python
IMG_SIZE = (96, 96)
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
```

### Treinar o Modelo

```bash
jupyter notebook notebooks/planthealth_training.ipynb
```

Requisitos para Treinamento:
- GPU: Recomendado (Google Colab oferece GPU gratuita)
- RAM: Minimo 8GB
- Espaco em Disco: ~5GB
- Tempo: ~30-60 minutos (com GPU)

---

## Deploy

### Hugging Face Spaces

1. Criar conta em https://huggingface.co
2. Criar novo Space (tipo Gradio)
3. Upload dos arquivos:
   - app/gradio_app.py -> app.py
   - models/planthealth_mobilenet_fast.keras
   - requirements.txt

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app/gradio_app.py"]
```

---

## Contribuindo

Contribuicoes sao muito bem-vindas!

### Como Contribuir

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudancas (`git commit -m 'Add: nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## Roadmap

### Versao 1.0 (Concluido)
- [x] Implementacao basica do modelo
- [x] Interface Gradio funcional
- [x] Documentacao completa

### Versao 2.0 (Planejado)
- [ ] Adicionar mais classes de plantas (30+)
- [ ] Implementar deteccao de multiplas doencas
- [ ] Criar API REST com FastAPI
- [ ] Dashboard de analytics

### Versao 3.0 (Futuro)
- [ ] App mobile (Flutter/React Native)
- [ ] Integracao com IoT
- [ ] Sistema de recomendacao de tratamento
- [ ] Suporte multilinguagem

---

## Licenca

Este projeto esta licenciado sob a Licenca MIT - veja o arquivo LICENSE para detalhes.

---

## Autor

Ricardo Frugoni

- GitHub: [@ricardofrugoni](https://github.com/ricardofrugoni)
- LinkedIn: [Ricardo Frugoni](https://linkedin.com/in/ricardofrugoni)
- Email: ricardo@codex.ai

---

## Agradecimentos

- PlantVillage Dataset: Dataset de alta qualidade
- TensorFlow/Keras Team: Framework de Deep Learning
- Gradio Team: Interface web intuitiva
- Google Colab: Ambiente com GPU gratuita
- Comunidade Open Source: Suporte e inspiracao

---

## Citacoes

### PlantVillage Dataset

```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

### MobileNetV2

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew},
  booktitle={CVPR},
  year={2018}
}
```

---

## Disclaimer

Este e um projeto educacional e de pesquisa. Para diagnosticos reais de doencas em plantas, consulte um agronomo qualificado.

---

Desenvolvido com dedicacao usando Transfer Learning e TensorFlow.
