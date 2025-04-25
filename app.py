import os
from flask import Flask, render_template, request, jsonify, send_file
from PyPDF2 import PdfReader
from io import BytesIO
import ollama
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
import torch
import pytesseract
from pytesseract import Output

app = Flask(__name__)

# Configurações
TEXT_MODEL = "llama3"  # Modelo de texto (usando Ollama local)
IMAGE_MODEL = "openai/clip-vit-base-patch32"  # Modelo para análise de imagens
YOLO_MODEL = "yolov8n.pt"  # Modelo para detecção de objetos
AUDIO_MODEL = "base"  # Modelo Whisper para áudio

# Cache para modelos pesados
model_cache = {}

# Função para carregar modelos sob demanda
def load_model(model_type):
    if model_type not in model_cache:
        if model_type == "image":
            # Carrega CLIP para classificação de imagens
            clip_pipe = pipeline("zero-shot-image-classification", 
                               model=IMAGE_MODEL,
                               device="cuda" if torch.cuda.is_available() else "cpu")
            
            # Carrega YOLO para detecção de objetos
            yolo_model = YOLO(YOLO_MODEL)
            
            model_cache["image"] = {
                'clip': clip_pipe,
                'yolo': yolo_model
            }
    return model_cache.get(model_type)

# Melhoria de prompts
def enhance_prompt(user_prompt, modality):
    enhancements = {
        'text': "[CONTEXTO ACADÊMICO] Responda de forma clara, objetiva e com referências quando possível. Limite a 400 palavras.\nPergunta: ",
        'pdf': "[ANÁLISE DE DOCUMENTO] Resuma o texto abaixo em tópicos principais, destacando conceitos-chave:\n",
        'image': "[ANÁLISE DE IMAGEM] Descreva e analise o conteúdo desta imagem, identificando elementos relevantes:\n",
        'audio': "[TRANSCRIÇÃO] Converta o áudio para texto, identificando os tópicos principais:\n"
    }
    return enhancements.get(modality, '') + user_prompt

# Função para analisar imagens
def analyze_image(image_file, prompt=None):
    try:
        image = Image.open(BytesIO(image_file.read()))
        
        # Primeiro verifica se é um documento/texto usando OCR
        try:
            text = pytesseract.image_to_string(image, lang='por+eng')
            if len(text.strip()) > 50:  # Se encontrar texto significativo
                return {
                    'is_document': True,
                    'text': text,
                    'analysis': "Documento/texto detectado na imagem"
                }
        except Exception as e:
            print(f"Erro no OCR: {str(e)}")
        
        # Se não for documento, prossegue com análise visual
        models = load_model("image")
        if not models:
            raise Exception("Modelos de imagem não carregados")
        
        clip_result = models['clip'](
            image,
            candidate_labels=["documento", "texto", "natureza", "pessoa", "objeto", "tela", "foto"]
        )
        
        yolo_results = models['yolo'](image)
        detections = []
        
        for result in yolo_results:
            for box in result.boxes:
                if float(box.conf[0]) > 0.25:  # Filtra detecções com baixa confiança
                    detections.append({
                        'object': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'position': box.xyxy[0].tolist()
                    })
        
        return {
            'is_document': False,
            'classification': clip_result,
            'detections': detections,
            'ocr_text': pytesseract.image_to_string(image, lang='por+eng')[:500]  # Texto OCR limitado
        }
        
    except Exception as e:
        raise Exception(f"Erro na análise de imagem: {str(e)}")

# Rota principal
@app.route('/')
def home():
    return render_template('index.html')

# Processamento multimodal
@app.route('/process', methods=['POST'])
def process():
    modality = request.form.get('modality')
    user_prompt = request.form.get('prompt', '')
    
    try:
        # Processamento de texto
        if modality == 'text':
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(user_prompt, 'text')
            )
            return jsonify({'result': response['response']})
        
        # Processamento de PDF
        elif modality == 'pdf' and 'file' in request.files:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(text, 'pdf'))
            return jsonify({'result': response['response']})
        
        # Análise de imagens
        elif modality == 'image' and 'file' in request.files:
            image_file = request.files['file']
            
            # Verifica se é realmente uma imagem
            try:
                Image.open(image_file).verify()
                image_file.seek(0)
            except:
                return jsonify({'error': 'Arquivo não é uma imagem válida'}), 400
            
            analysis = analyze_image(image_file, user_prompt)
            
            if analysis.get('is_document'):
                response = ollama.generate(
                    model=TEXT_MODEL,
                    prompt=f"Analise este texto extraído de uma imagem:\n{analysis['text']}\nContexto: {user_prompt}"
                )
                return jsonify({
                    'type': 'document',
                    'description': response['response'],
                    'text': analysis['text']
                })
            else:
                description_prompt = f"Descreva esta imagem com base na análise:\n" \
                                    f"Classificação: {analysis['classification']}\n" \
                                    f"Objetos detectados: {analysis['detections']}\n" \
                                    f"Texto OCR (se houver): {analysis.get('ocr_text', 'Nenhum texto significativo encontrado')}\n" \
                                    f"Contexto adicional: {user_prompt}"
                
                response = ollama.generate(
                    model=TEXT_MODEL,
                    prompt=enhance_prompt(description_prompt, 'image')
                )
                
                return jsonify({
                    'type': 'image',
                    'description': response['response'],
                    'classification': analysis['classification'],
                    'detections': analysis['detections'],
                    'ocr_text': analysis.get('ocr_text')
                })
        
        return jsonify({'error': 'Modalidade não suportada ou arquivo não enviado'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper para extrair texto de PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(BytesIO(file.read()))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        raise Exception(f"Erro ao ler PDF: {str(e)}")

if __name__ == '__main__':
    # Verifica se o modelo Ollama está disponível
    try:
        ollama.list()
    except Exception as e:
        print(f"Erro: Ollama não está rodando. Execute 'ollama serve' em outro terminal. Detalhes: {str(e)}")
    
    # Configura o Tesseract (ajuste o caminho conforme necessário)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    app.run(debug=True, port=5000)