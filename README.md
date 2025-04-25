📌 Descrição do Projeto
Este projeto é um assistente de IA multimodal que processa e analisa diferentes tipos de entradas:

Texto: Respostas contextualizadas usando o modelo Llama3

PDF: Extração e sumarização de conteúdo

Imagens: Reconhecimento de objetos (YOLOv8), classificação de cenas (CLIP) e OCR (Tesseract)

🛠️ Funcionalidades Principais
Processamento de Texto:

Análise acadêmica/profissional de textos

Respostas contextualizadas com limite de 400 palavras

Análise de PDF:

Extração de texto

Identificação de conceitos-chave

Sumarização automática

Análise de Imagens:

Detecção de objetos com YOLOv8

Classificação de cenas com CLIP

Reconhecimento óptico de caracteres (OCR) em português/inglês

Distinção automática entre imagens regulares e documentos

📦 Dependências
Backend (Python):
flask
ollama
pypdf2
torch
transformers
ultralytics
pillow
pytesseract

Frontend:
Navegador moderno (Chrome, Firefox, Edge)



🚀 Como Executar o Projeto
1. Pré-requisitos:
Python 3.8+
Ollama instalado e rodando (para modelos locais)
Tesseract OCR instalado

2. Instalação:
# Clone o repositório
git clone [seu-repositorio]

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt


3. Configuração de Modelos Locais:
Ollama (Llama3):
# Instale e inicie o Ollama
ollama pull llama3
ollama serve



Tesseract OCR:
Windows: Baixe o instalador https://github.com/UB-Mannheim/tesseract/wiki
Linux: sudo apt install tesseract-ocr tesseract-ocr-por
macOS: brew install tesseract

YOLOv8 e CLIP:
Serão baixados automaticamente na primeira execução


4. Iniciar a Aplicação:
python app.py

Acesse: http://localhost:5000

📄 Exemplo de Uso
Selecione a modalidade (Texto, PDF ou Imagem)

Insira um prompt 

Faça upload do arquivo (para PDF/Imagem)

Veja os resultados da análise





