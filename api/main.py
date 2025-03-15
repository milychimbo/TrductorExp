from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from flask import  request, jsonify
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os

app = FastAPI()

ModelosIdioma = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en"
}

ModeloQuechua = "somosnlp-hackathon-2022/t5-small-finetuned-spanish-to-quechua"
ModeloQuechuaT5 = AutoModelForSeq2SeqLM.from_pretrained(ModeloQuechua)
TokenizerQuechuaT5 = AutoTokenizer.from_pretrained(ModeloQuechua)

def traducir_a_quechua(texto):
    input_ids = TokenizerQuechuaT5(texto, return_tensors="pt").input_ids
    outputs = ModeloQuechuaT5.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)
    texto_traducido = TokenizerQuechuaT5.decode(outputs[0], skip_special_tokens=True)
    return texto_traducido

# Path to the favicon.ico file
favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")

# Mount the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.post("/traducir")
def traducir_texto():
    try:
        data = request.get_json()
        texto = data.get('texto')
        idioma_origen = data.get('origen', 'es')
        idioma_destino = data.get('destino', 'en')

        if not texto:
            return jsonify({'error': 'No se proporcionó texto para traducir'}), 400

        # Si la traducción es es a qu (no hay de qu a es)
        if idioma_origen == "es" and idioma_destino == "qu":
            texto_traducido = traducir_a_quechua(texto)
            return jsonify({'texto traducido': texto_traducido})

        # Si el idioma está de es a en y viceversa
        modelo_hf = ModelosIdioma.get((idioma_origen, idioma_destino))
        if modelo_hf:
            traductor = pipeline("translation", model=modelo_hf)
            texto_traducido = traductor(texto)[0]['translation_text']
            return jsonify({'texto traducido': texto_traducido})

        return jsonify({'error': f'La traducción de {idioma_origen} a {idioma_destino} no está disponible'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500