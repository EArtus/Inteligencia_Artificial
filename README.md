# 📌 Descrição do Projeto

Este projeto é um **assistente de IA multimodal** que processa e analisa diferentes tipos de entrada:

- **Texto**: Respostas contextualizadas usando o modelo **LLaMA3**
- **PDF**: Extração e sumarização de conteúdo
- **Imagens**:  
  - Reconhecimento de objetos (**YOLOv8**)  
  - Classificação de cenas (**CLIP**)  
  - OCR com suporte a **português** e **inglês** (**Tesseract**)

---

# 🛠️ Funcionalidades Principais

### 📄 Processamento de Texto
- Análise acadêmica/profissional de textos  
- Respostas contextualizadas com limite de palavras

### 📚 Análise de PDF
- Extração de texto  
- Identificação de conceitos-chave  
- Sumarização automática

### 🖼️ Análise de Imagens
- Detecção de objetos com **YOLOv8**  
- Classificação de cenas com **CLIP**  
- Reconhecimento óptico de caracteres (**OCR**)  
- Detecção automática entre imagens regulares e documentos

---

# 📦 Dependências

### 🔧 Backend (Python)

- `flask`  
- `ollama`  
- `pypdf2`  
- `torch`  
- `transformers`  
- `ultralytics`  
- `pillow`  
- `pytesseract`

### 💻 Frontend

- Navegador (Chrome, Firefox, Edge)

---

# 🚀 Como Executar o Projeto

### 📋 Pré-requisitos

- Python 3.8+ instalado  
- [Ollama](https://ollama.com) instalado e rodando (para modelos locais)  
- Tesseract OCR instalado

---

### 🔧 Instalação

## Clone o repositório
git clone https://github.com/EArtus/Inteligencia_Artificial.git

## Acesse a pasta
cd Inteligencia_Artificial

## Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

## Instale as dependências
pip install -r requirements.txt

#⚙️ Configuração de Modelos Locais  
##🧠 Ollama (LLaMA3)  
-Instale e inicie o Ollama  
ollama pull llama3  
ollama serve

##🔠 Tesseract OCR  
-Windows:  
Baixe o instalador em https://github.com/UB-Mannheim/tesseract/wiki  
Rode tesseract --list-langs no terminal para ver se tem 'por' nas linguagens, caso não tenha:  
(Opção manual)  
Baixe o arquivo por.traineddata de: https://github.com/tesseract-ocr/tessdata  
Cole o arquivo na pasta:C:\Program Files\Tesseract-OCR\tessdata  

(OPção por linha de comando)
Execute no Powershell(como administrador)
1.cd "C:\Program Files\Tesseract-OCR"  
2.if (!(Test-Path "tessdata")) { New-Item -ItemType Directory -Path "tessdata" }  
3.Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata" -OutFile "tessdata\por.traineddata"

-Linux:  
sudo apt install tesseract-ocr tesseract-ocr-por

-macOS:  
brew install tesseract

##🔍 YOLOv8 e CLIP
-Serão baixados automaticamente na primeira execução.

#▶️ Iniciar a Aplicação  
- python app.py

##🧪 Exemplo de Uso  
1. Selecione a modalidade desejada (Texto, PDF ou Imagem).  
2.Faça upload do arquivo (para PDF/Imagem)  
3.Insira um prompt de comando  
4.Veja os resultados da análise  
