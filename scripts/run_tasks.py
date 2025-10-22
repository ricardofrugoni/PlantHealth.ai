"""
PlantHealth AI - Script de Automação de Tarefas
Alternativa multiplataforma ao Makefile
"""

import sys
import subprocess
import shutil
from pathlib import Path
import argparse


class Colors:
    """Cores ANSI para terminal"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_colored(text, color):
    """Imprime texto colorido"""
    print(f"{color}{text}{Colors.NC}")


def run_command(command, shell=True):
    """Executa um comando e retorna o resultado"""
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def task_install():
    """Instala dependências básicas"""
    print_colored("Instalando dependências...", Colors.BLUE)
    success, output = run_command("pip install -r requirements.txt")
    if success:
        print_colored("✓ Dependências instaladas com sucesso!", Colors.GREEN)
    else:
        print_colored(f"✗ Erro ao instalar dependências:\n{output}", Colors.RED)
    return success


def task_install_dev():
    """Instala dependências de desenvolvimento"""
    print_colored("Instalando dependências de desenvolvimento...", Colors.BLUE)
    success1, _ = run_command("pip install -r requirements.txt")
    success2, _ = run_command("pip install pytest flake8 black mypy pytest-cov")
    
    if success1 and success2:
        print_colored("✓ Dependências de desenvolvimento instaladas!", Colors.GREEN)
    else:
        print_colored("✗ Erro ao instalar dependências", Colors.RED)
    
    return success1 and success2


def task_clean():
    """Remove arquivos temporários e cache"""
    print_colored("Limpando arquivos temporários...", Colors.BLUE)
    
    # Patterns to clean
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for pattern in patterns:
        for path in project_root.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"  Removido: {path}")
            except Exception as e:
                print_colored(f"  Erro ao remover {path}: {e}", Colors.YELLOW)
    
    print_colored("✓ Limpeza concluída!", Colors.GREEN)
    return True


def task_lint():
    """Executa verificação de código com flake8"""
    print_colored("Verificando código com flake8...", Colors.BLUE)
    success, output = run_command("flake8 src/ app/ --max-line-length=120 --extend-ignore=E203,W503")
    
    if success:
        print_colored("✓ Nenhum problema encontrado!", Colors.GREEN)
    else:
        print_colored("⚠ Avisos de lint encontrados:", Colors.YELLOW)
        print(output)
    
    return True  # Não falhar em warnings


def task_format():
    """Formata código com black"""
    print_colored("Formatando código com black...", Colors.BLUE)
    success, output = run_command("black src/ app/ --line-length=120")
    
    if success:
        print_colored("✓ Código formatado!", Colors.GREEN)
    else:
        print_colored(f"✗ Erro ao formatar código:\n{output}", Colors.RED)
    
    return success


def task_test():
    """Executa testes"""
    print_colored("Executando testes...", Colors.BLUE)
    success, output = run_command("pytest tests/ -v --cov=src --cov-report=html")
    
    if success:
        print_colored("✓ Testes concluídos!", Colors.GREEN)
        print(output)
    else:
        print_colored("⚠ Testes não configurados ou falharam", Colors.YELLOW)
    
    return True  # Não falhar se testes não existirem


def task_run_gradio():
    """Inicia aplicação Gradio"""
    print_colored("🌱 Iniciando PlantHealth AI (Gradio)...", Colors.GREEN)
    subprocess.run([sys.executable, "app/gradio_app.py"])


def task_run_streamlit():
    """Inicia aplicação Streamlit"""
    print_colored("🌱 Iniciando PlantHealth AI (Streamlit)...", Colors.GREEN)
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


def task_setup_env():
    """Cria arquivo .env a partir do exemplo"""
    print_colored("Configurando arquivo .env...", Colors.BLUE)
    
    env_file = Path(".env")
    example_file = Path("config.example.env")
    
    if env_file.exists():
        print_colored("⚠ Arquivo .env já existe.", Colors.YELLOW)
        return False
    
    if not example_file.exists():
        print_colored("✗ Arquivo config.example.env não encontrado!", Colors.RED)
        return False
    
    shutil.copy(example_file, env_file)
    print_colored("✓ Arquivo .env criado! Edite conforme necessário.", Colors.GREEN)
    return True


def task_info():
    """Mostra informações do projeto"""
    print_colored("PlantHealth AI - Informações do Projeto", Colors.BLUE)
    print()
    
    info = {
        "Nome": "PlantHealth AI",
        "Versão": "1.0.0",
        "Python": sys.version.split()[0],
    }
    
    # Tentar obter versões de pacotes
    try:
        import tensorflow as tf
        info["TensorFlow"] = tf.__version__
    except ImportError:
        info["TensorFlow"] = "Não instalado"
    
    try:
        import gradio as gr
        info["Gradio"] = gr.__version__
    except ImportError:
        info["Gradio"] = "Não instalado"
    
    try:
        import streamlit as st
        info["Streamlit"] = st.__version__
    except ImportError:
        info["Streamlit"] = "Não instalado"
    
    for key, value in info.items():
        print(f"  {Colors.GREEN}{key:15}{Colors.NC} {value}")
    print()


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="PlantHealth AI - Script de Automação de Tarefas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tarefas disponíveis:
  install       - Instala dependências básicas
  install-dev   - Instala dependências de desenvolvimento
  clean         - Remove arquivos temporários e cache
  lint          - Executa verificação de código (flake8)
  format        - Formata código com black
  test          - Executa testes
  run-gradio    - Inicia aplicação Gradio
  run-streamlit - Inicia aplicação Streamlit
  setup-env     - Cria arquivo .env a partir do exemplo
  info          - Mostra informações do projeto

Exemplos:
  python scripts/run_tasks.py install
  python scripts/run_tasks.py run-gradio
  python scripts/run_tasks.py lint format
        """
    )
    
    parser.add_argument(
        'tasks',
        nargs='+',
        help='Tarefas a executar (uma ou mais)'
    )
    
    args = parser.parse_args()
    
    # Mapeamento de tarefas
    tasks = {
        'install': task_install,
        'install-dev': task_install_dev,
        'clean': task_clean,
        'lint': task_lint,
        'format': task_format,
        'test': task_test,
        'run-gradio': task_run_gradio,
        'run-streamlit': task_run_streamlit,
        'setup-env': task_setup_env,
        'info': task_info,
    }
    
    # Executar tarefas
    for task_name in args.tasks:
        if task_name in tasks:
            print()
            print("=" * 60)
            tasks[task_name]()
            print("=" * 60)
        else:
            print_colored(f"✗ Tarefa desconhecida: {task_name}", Colors.RED)
            print(f"Tarefas disponíveis: {', '.join(tasks.keys())}")
            sys.exit(1)


if __name__ == "__main__":
    main()

