# Modelos Treinados

Este diretorio contem os modelos de Deep Learning treinados para o projeto PlantHealth.ai.

## Modelo Principal

planthealth_mobilenet_fast.keras

Arquivo: planthealth_mobilenet_fast.keras
Tamanho: aproximadamente 14 MB
Formato: Keras SavedModel (.keras)
Framework: TensorFlow 2.13+

## Download

Link para download: [ADICIONE-SEU-LINK-AQUI]

Apos baixar:
```bash
mv planthealth_mobilenet_fast.keras models/
```

## Estrutura do Modelo

Arquitetura:

```
Input (96x96x3)
    |
MobileNetV2 Base (Pre-treinada, Congelada)
    |
GlobalAveragePooling2D
    |
Dense(64, relu)
    |
Dense(16, softmax)
```

## Detalhes Tecnicos

| Componente | Valor |
|------------|-------|
| Arquitetura Base | MobileNetV2 |
| Pre-treinamento | ImageNet |
| Input Shape | (96, 96, 3) |
| Output Classes | 16 |
| Parametros Totais | 2,323,712 |
| Parametros Treinaveis | 65,808 |
| Tamanho | 14.2 MB |

## Uso

Carregar Modelo:

```python
from tensorflow import keras

model = keras.models.load_model('models/planthealth_mobilenet_fast.keras')
```

Fazer Predicao:

```python
import numpy as np
from PIL import Image

image = Image.open('folha.jpg').resize((96, 96))
img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Classe: {class_idx}, Confianca: {confidence:.2%}")
```

## Performance

Metricas:

| Metrica | Treino | Validacao | Teste |
|---------|--------|-----------|-------|
| Acuracia | 92.3% | 78.5% | 75.8% |
| Loss | 0.2156 | 0.5823 | 0.6234 |

Tempo de Inferencia:
- CPU: menor que 100ms por imagem
- GPU: menor que 20ms por imagem

## Classes

O modelo identifica 16 classes:

1. Apple - Apple scab
2. Apple - Black rot
3. Apple - Cedar apple rust
4. Apple - Healthy
5. Corn - Common rust
6. Corn - Gray leaf spot
7. Corn - Healthy
8. Corn - Northern leaf blight
9. Grape - Black rot
10. Grape - Esca (Black Measles)
11. Grape - Healthy
12. Grape - Leaf blight
13. Potato - Early blight
14. Potato - Healthy
15. Potato - Late blight
16. Tomato - Healthy

## Retreinar o Modelo

Para retreinar:

1. Preparar dataset em data/raw/
2. Abrir notebook: notebooks/planthealth_training.ipynb
3. Executar todas as celulas
4. Modelo sera salvo automaticamente

Requisitos:
- Python 3.10+
- TensorFlow 2.13+
- GPU recomendada
- aproximadamente 5GB espaco em disco

## Versoes

v1.0 (Atual):
- Modelo base MobileNetV2
- 16 classes
- Acuracia: aproximadamente 76%
- Data: 2025-01-22

## Troubleshooting

Erro: Modelo nao encontrado

Solucao: Baixe o modelo usando os links acima.

Erro: Versao incompativel

Solucao: Use TensorFlow 2.13 ou superior:
```bash
pip install tensorflow>=2.13.0
```

## Informacoes Adicionais

O arquivo .keras inclui:
- Arquitetura do modelo
- Pesos treinados
- Configuracao de compilacao
- Estado do optimizer

Compatibilidade:
- TensorFlow: maior ou igual 2.13.0
- Keras: maior ou igual 2.13.0
- Python: maior ou igual 3.10

## Licenca

Este modelo e disponibilizado sob a licenca MIT do projeto.

---

Desenvolvido usando Transfer Learning com MobileNetV2
