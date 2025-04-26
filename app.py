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


TEXT_MODEL = "llama3" 
IMAGE_MODEL = "openai/clip-vit-base-patch32"  
YOLO_MODEL = "yolov8n.pt" 


model_cache = {}


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

def enhance_prompt(user_prompt, modality):
    enhancements = {
        'text': """
Atue como um professor especialista na área. Siga estas diretrizes:
 **Contextualize**: Introduza o conceito de forma clara antes de detalhá-lo.
 **Desmembre o assunto**: Divida em tópicos lógicos (ex.: definição, aplicações, exemplos).
 **Aprofunde**: Inclua:
   - Fundamentos teóricos (com citações de livros/artigos, se relevante)
   - Casos práticos ou analogias
   - Possíveis equívocos comuns
 **Formato**: Use até 600 palavras, parágrafos curtos e marcadores (🔹) para ênfase.
 **Extra**: Ao final, sugira 1 exercício prático e 2 referências para estudo adicional.

Pergunta: """,

        'pdf': """
[ANÁLISE ACADÊMICA DE DOCUMENTO] Siga esta estrutura:
 **Resumo Executivo**: 3-4 frases com o núcleo do conteúdo.
 **Análise Detalhada**:
   - Seção 1: Principais teses/argumentos (ordem de importância)
   - Seção 2: Dados estatísticos ou evidências citadas (com crítica à metodologia, se aplicável)
   - Seção 3: Relação com outras teorias (compare com 1-2 autores clássicos)
 **Aplicações Práticas**: Como esse conteúdo pode ser usado em pesquisas ou projetos reais?
 **Limitações**: Pontos fracos ou lacunas no documento analisado.
""",

        'image': """
[DESCRIÇÃO PEDAGÓGICA DE IMAGEM] Siga este roteiro:
 **Análise Visual**:
   - Descreva elementos literais (objetos, cores, disposição)
   - Simbolismo ou metáforas visuais (se houver)
 **Contexto Acadêmico**:
   - Relacione com teorias/conceitos (ex.: "Esta imagem ilustra o modelo de Bohr porque...")
   - Compare com outras representações iconográficas do tema
 **Perguntas Orientadoras**: Proponha 3 questões para estimular análise crítica (ex.: "Como a escolha cromática reforça a mensagem?")
 **Atividade**: Sugira uma tarefa baseada na imagem (ex.: "Esboce um diagrama alternativo que...")
"""
    }
    return enhancements.get(modality, "Modo não reconhecido") + user_prompt

def analyze_image(image_file, prompt=None):
    try:
        image = Image.open(BytesIO(image_file.read()))
        
        try:
            text = pytesseract.image_to_string(image, lang='por+eng')
            if len(text.strip()) > 50:  
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
                if float(box.conf[0]) > 0.25:  
                    detections.append({
                        'object': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'position': box.xyxy[0].tolist()
                    })
        
        return {
            'is_document': False,
            'classification': clip_result,
            'detections': detections,
            'ocr_text': pytesseract.image_to_string(image, lang='por+eng')[:500]  
        }
        
    except Exception as e:
        raise Exception(f"Erro na análise de imagem: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    modality = request.form.get('modality')
    user_prompt = request.form.get('prompt', '')
    
    try:
        if modality == 'text':
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(user_prompt, 'text')
            )
            return jsonify({'result': response['response']})
        
        elif modality == 'pdf' and 'file' in request.files:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(text, 'pdf'))
            return jsonify({'result': response['response']})
        
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

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(BytesIO(file.read()))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        raise Exception(f"Erro ao ler PDF: {str(e)}")

if __name__ == '__main__':
    try:
        ollama.list()
    except Exception as e:
        print(f"Erro: Ollama não está rodando. Execute 'ollama serve' em outro terminal. Detalhes: {str(e)}")
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    app.run(debug=True, port=5000)