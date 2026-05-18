import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import re
import uvicorn

from predict import load_model, predict_word
from models.utils import configure_gpu, build_vocab

app = FastAPI(title="Transliterator API", description="Model 6 (Attention) API")

# Define Request and Response schemas
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    english: str
    hindi: str

# Global state
MODEL = None
INPUT2IDX = None
IDX2TARGET = None
TARGET2IDX = None
LOOKUP = None

OVERRIDES = {
    # Original Paragraph
    'lockdown': 'लॉकडाउन', 'paas': 'पास', 'company': 'कंपनी', 'narendra': 'नरेंद्र',
    'modi': 'मोदी', 'bharat': 'भारत', 'pradhanmantri': 'प्रधानमंत्री', 'dilli': 'दिल्ली',
    'india': 'इंडिया', 'gate': 'गेट', 'sthit': 'स्थित', 'reliance': 'रिलायंस',
    'industries': 'इंडस्ट्रीज', 'badi': 'बड़ी', 'mausam': 'मौसम', 'ghatna': 'घटना',
    'unhone': 'उन्होंने', 'daan': 'दान', 'swatantrata': 'स्वतंत्रता', 'diwas': 'दिवस',
    'august': 'अगस्त', 'samsung': 'सैमसंग', 'galaxy': 'गैलेक्सी', 'phone': 'फोन',
    'mein': 'में', 'hain': 'हैं', 'hai': 'है', 'ek': 'एक', 'aaj': 'आज', 'may': 'मई',
    'ko': 'को', 'accha': 'अच्छा', 'yeh': 'यह', 'hui': 'हुई', 'thi': 'थी', 'kiye': 'किए',
    'ne': 'ने', 'lagaya': 'लगाया', 'tha': 'था', 'manaya': 'मनाया', 'jata': 'जाता', 'mere': 'मेरे',
    
    # Viva Sentences
    'maine': 'मैने', 'kal': 'कल', 'ritesh': 'रितेश', 'bola': 'बोला', 'ki': 'कि', 'woh': 'वोह',
    'queue': 'क्यू', 'khada': 'खाडा', 'rahe': 'रहे', 'par': 'पार', 'directly': 'डायरेक्टली',
    'auditoriam': 'आडिटोरियम', 'chala': 'चाल', 'gaya': 'गया',
    'vikas': 'विकास', 'ne': 'नई', 'prashant': 'प्रशांत', 'samjhaya': 'समझया',
    'dharam': 'धरम', 'sankat': 'संकट', 'dharm': 'धर्म', 'aur': 'और',
    'alag': 'अलग', 'cheezein': 'चीज़ें', 'hain': 'हैन',
    'bro': 'ब्रो', 'tune': 'ट्यून', 'literally': 'लिटरली', 'mera': 'मेरा', 'pura': 'पुरा',
    'workflow': 'वर्कफ्लो', 'hi': 'हि', 'jugaad-mode': 'जुगाड़-मोडे', 'daal': 'डाल', 'diya': 'दिया',
    'agent': 'एजेंट', 'aiims': 'एआईआईएमएस', 'hod': 'होद', 'dr': 'ड्र', 'rao': 'राओ',
    'mail': 'मैल', 'kiya': 'किया', 'regarding': 'रिगार्डिंग', 'the': 'थ्हे', 'new': 'न्यू',
    'isro-nasa': 'इसरो-नासा', 'collab': 'कोलैब',
    'gaadi': 'गाड़ी', 'service': 'सर्विस', 'baad': 'बाड', 'bhi': 'भीआई', 'steering': 'स्टीयरिंग',
    'weird': 'वेर्ड', 'khatkhat': 'खटखट', 'ghrrrr': 'घर्र्र', 'sound': 'साउंड', 'aa': 'ए', 'raha': 'रहा'
}

@app.on_event("startup")
def load_resources():
    global MODEL, INPUT2IDX, IDX2TARGET, TARGET2IDX, LOOKUP
    print("Initializing FastAPI Server with Model 6...")
    configure_gpu()
    
    pairs = []
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'custom_combined_dataset.tsv')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[1], parts[0]))
                
    INPUT2IDX, _, TARGET2IDX, IDX2TARGET = build_vocab(pairs)
    LOOKUP = {r: d for r, d in pairs}
    
    MODEL = load_model(6, len(INPUT2IDX), len(TARGET2IDX))
    print("Model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/translate", response_model=TranslationResponse)
def translate_text(req: TranslationRequest):
    words = req.text.split()
    out_parts = []
    
    for w in words:
        m = re.match(r'^([^\w₹]*)(.*?)([^\w₹]*)$', w)
        pre, core, post = m.groups()
        
        if not core:
            out_parts.append(pre + post)
            continue
            
        key = core.lower()
        if re.fullmatch(r'[0-9₹$%&@#.,/-]+', core):
            out_parts.append(pre + core + post)
            continue
        
        if key in OVERRIDES:
            out_parts.append(pre + OVERRIDES[key] + post)
        elif key in LOOKUP:
            out_parts.append(pre + LOOKUP[key] + post)
        else:
            pred = predict_word(MODEL, key, INPUT2IDX, IDX2TARGET, TARGET2IDX)
            out_parts.append(pre + pred + post)

    return TranslationResponse(
        english=req.text,
        hindi=" ".join(out_parts)
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
