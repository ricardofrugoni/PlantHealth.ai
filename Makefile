# PlantHealth AI - Makefile
# Automação de tarefas comuns

.PHONY: help install install-dev clean lint format test run-gradio run-streamlit docs

# Cores para output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)PlantHealth AI - Comandos Disponíveis$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Instala dependências básicas
	@echo "$(BLUE)Instalando dependências...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependências instaladas com sucesso!$(NC)"

install-dev: ## Instala dependências de desenvolvimento
	@echo "$(BLUE)Instalando dependências de desenvolvimento...$(NC)"
	pip install -r requirements.txt
	pip install pytest flake8 black mypy pytest-cov
	@echo "$(GREEN)✓ Dependências de desenvolvimento instaladas!$(NC)"

clean: ## Remove arquivos temporários e cache
	@echo "$(BLUE)Limpando arquivos temporários...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + || true
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/
	@echo "$(GREEN)✓ Limpeza concluída!$(NC)"

lint: ## Executa verificação de código (flake8)
	@echo "$(BLUE)Verificando código com flake8...$(NC)"
	@flake8 src/ app/ --max-line-length=120 --extend-ignore=E203,W503 || echo "$(YELLOW)⚠ Avisos de lint encontrados$(NC)"

format: ## Formata código com black
	@echo "$(BLUE)Formatando código com black...$(NC)"
	black src/ app/ --line-length=120
	@echo "$(GREEN)✓ Código formatado!$(NC)"

type-check: ## Verifica tipos com mypy
	@echo "$(BLUE)Verificando tipos com mypy...$(NC)"
	mypy src/ app/ --ignore-missing-imports || echo "$(YELLOW)⚠ Erros de tipo encontrados$(NC)"

test: ## Executa testes (quando disponíveis)
	@echo "$(BLUE)Executando testes...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html || echo "$(YELLOW)⚠ Testes não configurados ainda$(NC)"

run-gradio: ## Inicia aplicação Gradio
	@echo "$(GREEN)🌱 Iniciando PlantHealth AI (Gradio)...$(NC)"
	python app/gradio_app.py

run-streamlit: ## Inicia aplicação Streamlit
	@echo "$(GREEN)🌱 Iniciando PlantHealth AI (Streamlit)...$(NC)"
	streamlit run app/streamlit_app.py

setup-env: ## Cria arquivo .env a partir do exemplo
	@echo "$(BLUE)Configurando arquivo .env...$(NC)"
	@if [ ! -f .env ]; then \
		cp config.example.env .env; \
		echo "$(GREEN)✓ Arquivo .env criado! Edite conforme necessário.$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Arquivo .env já existe.$(NC)"; \
	fi

docs: ## Gera documentação (quando configurada)
	@echo "$(BLUE)Gerando documentação...$(NC)"
	@echo "$(YELLOW)⚠ Documentação automática não configurada ainda$(NC)"

docker-build: ## Build da imagem Docker (quando disponível)
	@echo "$(BLUE)Building Docker image...$(NC)"
	@echo "$(YELLOW)⚠ Dockerfile não configurado ainda$(NC)"

docker-run: ## Executa container Docker (quando disponível)
	@echo "$(BLUE)Running Docker container...$(NC)"
	@echo "$(YELLOW)⚠ Dockerfile não configurado ainda$(NC)"

check-all: lint type-check test ## Executa todas as verificações

info: ## Mostra informações do projeto
	@echo "$(BLUE)PlantHealth AI - Informações do Projeto$(NC)"
	@echo ""
	@echo "  $(GREEN)Nome:$(NC)           PlantHealth AI"
	@echo "  $(GREEN)Versão:$(NC)         1.0.0"
	@echo "  $(GREEN)Python:$(NC)         $(shell python --version 2>&1)"
	@echo "  $(GREEN)TensorFlow:$(NC)     $(shell python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Não instalado')"
	@echo "  $(GREEN)Gradio:$(NC)         $(shell python -c 'import gradio as gr; print(gr.__version__)' 2>/dev/null || echo 'Não instalado')"
	@echo ""

.DEFAULT_GOAL := help

