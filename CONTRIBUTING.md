# Contribuindo para PlantHealth AI

Obrigado por considerar contribuir para o PlantHealth AI! 🌱

## Como Contribuir

### Reportando Bugs

Se você encontrou um bug, por favor:

1. Verifique se o bug já não foi reportado nas [Issues](https://github.com/ricardofrugoni/planthealth-ai/issues)
2. Crie uma nova issue com:
   - Título claro e descritivo
   - Descrição detalhada do problema
   - Passos para reproduzir o bug
   - Comportamento esperado vs comportamento atual
   - Screenshots (se aplicável)
   - Informações do ambiente (OS, versão do Python, etc.)

### Sugerindo Melhorias

Sugestões são bem-vindas! Para propor uma melhoria:

1. Abra uma issue com a tag `enhancement`
2. Descreva a funcionalidade desejada
3. Explique por que seria útil
4. Forneça exemplos de uso, se possível

### Pull Requests

1. **Fork** o repositório
2. Crie uma **branch** para sua feature (`git checkout -b feature/MinhaFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add: minha nova feature'`)
4. **Push** para a branch (`git push origin feature/MinhaFeature`)
5. Abra um **Pull Request**

#### Padrões de Código

- Use **PEP 8** para estilo de código Python
- Adicione **docstrings** para funções e classes
- Escreva **comentários** claros quando necessário
- Mantenha **type hints** quando possível
- Adicione **testes** para novas funcionalidades

#### Padrão de Commits

Usamos commits semânticos:

- `Add:` para novas features
- `Fix:` para correções de bugs
- `Update:` para atualizações
- `Remove:` para remoções
- `Refactor:` para refatorações
- `Docs:` para documentação
- `Test:` para testes
- `Style:` para formatação

Exemplo: `Add: suporte para detecção de múltiplas doenças`

### Estrutura do Projeto

```
planthealth_essentials/
├── app/              # Interfaces web (Gradio, Streamlit)
├── data/             # Datasets
├── models/           # Modelos treinados
├── src/              # Código fonte principal
│   ├── data/         # Módulos de dados
│   ├── models/       # Arquiteturas de modelos
│   └── utils/        # Utilitários
├── results/          # Resultados e visualizações
└── notebooks/        # Jupyter notebooks
```

### Ambiente de Desenvolvimento

1. Clone o repositório:
```bash
git clone https://github.com/ricardofrugoni/planthealth-ai.git
cd planthealth-ai
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env conforme necessário
```

### Testes

Antes de submeter um PR:

1. Teste seu código localmente
2. Verifique o estilo com `flake8` ou `pylint`
3. Execute os testes (quando disponíveis)
4. Teste a aplicação Gradio/Streamlit

### Documentação

- Atualize o README.md se necessário
- Adicione docstrings para novas funções
- Mantenha comentários claros e em português
- Atualize arquivos README.md específicos (data/, models/)

### Código de Conduta

- Seja respeitoso e inclusivo
- Aceite críticas construtivas
- Foque no que é melhor para a comunidade
- Mostre empatia com outros membros

### Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a mesma licença MIT do projeto.

---

## Contato

- GitHub Issues: Para bugs e features
- Email: ricardo@codex.ai

Obrigado por contribuir! 🌱🚀

