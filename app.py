import os
from flask import Flask, render_template, request, jsonify, send_file
from PyPDF2 import PdfReader
from io import BytesIO
import ollama
import torch
from diffusers import StableDiffusionPipeline
try:
    import whisper
except ImportError:
    print("Biblioteca whisper não instalada. Recursos de áudio serão desativados.")
    whisper = None
import tempfile

app = Flask(__name__)

# Configurações
TEXT_MODEL = "llama3"  # Modelo de texto (usando Ollama local)
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # Modelo de imagens
AUDIO_MODEL = "base"  # Modelo Whisper para áudio

# Cache para modelos pesados
model_cache = {}

# Função para carregar modelos sob demanda
def load_model(model_type):
    if model_type not in model_cache:
        if model_type == "image":
            model_cache["image"] = StableDiffusionPipeline.from_pretrained(
                IMAGE_MODEL,
                torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        elif model_type == "audio" and whisper is not None:
            model_cache["audio"] = whisper.load_model(AUDIO_MODEL)
    return model_cache.get(model_type)

# Melhoria de prompts
def enhance_prompt(user_prompt, modality):
    enhancements = {
        'text': "[CONTEXTO ACADÊMICO] Responda de forma clara, objetiva e com referências quando possível. Limite a 400 palavras.\nPergunta: ",
        'pdf': "[ANÁLISE DE DOCUMENTO] Resuma o texto abaixo em tópicos principais, destacando conceitos-chave:\n",
        'image': "[DESCRIÇÃO DETALHADA] Gere uma imagem realista que represente: ",
        'audio': "[TRANSCRIÇÃO] Converta o áudio para texto, identificando os tópicos principais:\n"
    }
    return enhancements.get(modality, '') + user_prompt

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
            )  # Este parêntese fecha a chamada do generate
            return jsonify({'result': response['response']})
        
        # Processamento de PDF
        elif modality == 'pdf' and 'file' in request.files:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(text, 'pdf'))
            return jsonify({'result': response['response']})
        
        # Geração de imagens
        elif modality == 'image':
            pipe = load_model("image")
            if not pipe:
                return jsonify({'error': 'Modelo de imagem não carregado'}), 500
            
            enhanced_prompt = enhance_prompt(user_prompt, 'image')
            image = pipe(enhanced_prompt).images[0]
            img_io = BytesIO()
            image.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
        
        # Processamento de áudio
        elif modality == 'audio' and 'file' in request.files and whisper is not None:
            audio_model = load_model("audio")
            if not audio_model:
                return jsonify({'error': 'Modelo de áudio não carregado'}), 500
            
            audio_file = request.files['file']
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                audio_file.save(tmp.name)
                result = audio_model.transcribe(tmp.name)
                os.unlink(tmp.name)
                
            response = ollama.generate(
                model=TEXT_MODEL,
                prompt=enhance_prompt(result["text"], 'audio'))
            return jsonify({
                'transcription': result["text"],
                'analysis': response['response']
            })
        
        return jsonify({'error': 'Modalidade não suportada'}), 400
    
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
    
    app.run(debug=True, port=5000)