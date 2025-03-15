from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import uvicorn
import os

app = FastAPI()

# Definir modelos de traducción disponibles
ModelosIdioma = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en"
}

# Modelo de traducción de español a quechua
ModeloQuechua = "somosnlp-hackathon-2022/t5-small-finetuned-spanish-to-quechua"
ModeloQuechuaT5 = AutoModelForSeq2SeqLM.from_pretrained(ModeloQuechua)
TokenizerQuechuaT5 = AutoTokenizer.from_pretrained(ModeloQuechua)

def traducir_a_quechua(texto: str) -> str:
    input_ids = TokenizerQuechuaT5(texto, return_tensors="pt").input_ids
    outputs = ModeloQuechuaT5.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)
    return TokenizerQuechuaT5.decode(outputs[0], skip_special_tokens=True)

# Manejo de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ruta del favicon (asegúrate de que el archivo existe en "static/")
favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de traducción"}

@app.post("/traducir")
async def traducir_texto(request: Request):
    try:
        data = await request.json()
        texto = data.get("texto")
        idioma_origen = data.get("origen", "es")
        idioma_destino = data.get("destino", "en")

        if not texto:
            return JSONResponse(content={"error": "No se proporcionó texto para traducir"}, status_code=400)

        # Traducción de español a quechua
        if idioma_origen == "es" and idioma_destino == "qu":
            texto_traducido = traducir_a_quechua(texto)
            return JSONResponse(content={"texto traducido": texto_traducido})

        # Traducción entre inglés y español
        modelo_hf = ModelosIdioma.get((idioma_origen, idioma_destino))
        if modelo_hf:
            traductor = pipeline("translation", model=modelo_hf)
            texto_traducido = traductor(texto)[0]["translation_text"]
            return JSONResponse(content={"texto traducido": texto_traducido})

        return JSONResponse(content={"error": f"La traducción de {idioma_origen} a {idioma_destino} no está disponible"}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)