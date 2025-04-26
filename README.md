# 📌 Descrição do Projeto

Este projeto é um **Auxiliar Estudantil** que processa e analisa diferentes tipos de entrada:

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
-Linux/macOS  source venv/bin/activate      
-Windows  venv\Scripts\activate  

## Instale as dependências
pip install -r requirements.txt

# ⚙️ Configuração de Modelos Locais

## 🧠 Ollama (LLaMA3)

- Instale e inicie o Ollama:

```bash
ollama pull llama3
ollama serve
```

---

## 🔠 Tesseract OCR

### 🪟 Windows:

1. Baixe o instalador em:  
   [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

2. Verifique as linguagens instaladas:

```bash
tesseract --list-langs
```

3. Caso **não tenha `por` (Português)** nas linguagens, siga uma das opções abaixo:

#### 📥 Opção Manual:

- Baixe o arquivo `por.traineddata` de:  
  [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)

- Cole o arquivo no diretório:  
  `C:\Program Files\Tesseract-OCR\tessdata`

#### 🧑‍💻 Opção por Linha de Comando:

Execute os comandos abaixo no **PowerShell (como administrador)**:

```powershell
cd "C:\Program Files\Tesseract-OCR"
if (!(Test-Path "tessdata")) { New-Item -ItemType Directory -Path "tessdata" }
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/por.traineddata" -OutFile "tessdata\por.traineddata"
```

---

### 🐧 Linux:

```bash
sudo apt install tesseract-ocr tesseract-ocr-por
```

---

### 🍏 macOS:

```bash
brew install tesseract
```

---

## 🔍 YOLOv8 e CLIP

- Serão baixados **automaticamente** na primeira execução da aplicação.

---

# ▶️ Iniciar a Aplicação

```bash
python app.py
```

Acesse: [http://localhost:5000](http://localhost:5000)

---

# 🧪 Exemplo de Uso

1. Selecione a **modalidade desejada** (Texto, PDF ou Imagem)  
2. Faça **upload do arquivo** (para PDF/Imagem)  
3. Insira um **prompt de comando**  
4. Veja os **resultados da análise**

---
