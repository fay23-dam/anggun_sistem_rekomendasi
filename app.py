from flask import Flask, request, jsonify, render_template
from main_qa_module import PDFProcessor, QAEngine
from dotenv import load_dotenv
import os
import time
import traceback
from threading import Lock
import logging
import sys

# ====== Fix Unicode Encoding for Windows ======
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ====== Setup Logging ======
def safe_log_text(text):
    """Remove emojis and other problematic characters for Windows console"""
    return text.encode('ascii', 'replace').decode('ascii')

# Create a custom formatter that removes problematic characters
class SafeFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        return safe_log_text(original)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler with UTF-8 encoding
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler with safe formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ====== Load Environment Variables ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("ERROR: OPENAI_API_KEY tidak ditemukan di file .env")

# ====== Inisialisasi Flask App ======
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# ====== Inisialisasi Sistem QA ======
def initialize_qa_system():
    try:
        start_time = time.time()
        processor = PDFProcessor()
        pdf_path = 'dokumen/Dataaa_X1.pdf'
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"ERROR: File PDF tidak ditemukan: {pdf_path}")
        
        logger.info("Memulai inisialisasi sistem QA...")

        # Initialize vector store (load or create new)
        if not processor.initialize_vector_store(pdf_path):
            raise ValueError("ERROR: Gagal memuat atau membuat vektor store")
        
        # Inisialisasi QAEngine dengan processor
        qa_engine = QAEngine(processor.vector_store, processor)
        
        logger.info(f"Sistem QA siap digunakan dalam {time.time()-start_time:.2f} detik")
        return qa_engine
        
    except Exception as e:
        logger.error(f"Gagal inisialisasi sistem QA: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Initialize QA Engine globally
qa_engine = initialize_qa_system()

# Lock untuk thread-safe processing
processing_lock = Lock()

# ====== ROUTES ======
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ready' if qa_engine else 'error',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_engine:
        return jsonify({
            'response': 'Sistem QA belum siap. Silakan coba lagi nanti.',
            'status': 'error'
        }), 503

    # Cek jika sistem sedang sibuk
    if not processing_lock.acquire(blocking=False):
        return jsonify({
            'response': 'Sistem sedang memproses pertanyaan sebelumnya. Silakan tunggu...',
            'status': 'busy'
        }), 429

    try:
        # Parse JSON input
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'response': 'Pertanyaan kosong tidak dapat diproses'}), 400

        logger.info(f"Pertanyaan diterima: {user_message}")
        start_time = time.time()
        
        # Get response from QA engine
        response = qa_engine.generate_response(user_message)
        processing_time = time.time() - start_time

        logger.info(f"Waktu pemrosesan: {processing_time:.2f} detik")
        
        # Debug: Cetak respons mentah
        logger.info(f"Respons mentah: {safe_log_text(response[:500])}...")

        # Prepare result
        result = {
            'status': 'success',
            'processing_time': f"{processing_time:.2f} detik"
        }

        # Check if response is HTML (recommendations)
        if '<div class="product-card"' in response:
            result['response'] = response
            result['response_type'] = 'html'
            logger.info("Mengirim respons HTML")
        elif 'div class="no-recommendation"' in response:
            result['response'] = response
            result['response_type'] = 'html'
            logger.info("Mengirim respons HTML dengan rekomendasi")
        else:
            result['response'] = response
            result['response_type'] = 'text'
            logger.info(f"Jawaban: {safe_log_text(response)}")

        return jsonify(result)
            
    except Exception as e:
        error_msg = f'Terjadi kesalahan: {str(e)}'
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            'response': 'Terjadi kesalahan dalam memproses permintaan. Mohon coba lagi.',
            'status': 'error'
        }), 500
    finally:
        processing_lock.release()

# ====== MAIN ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)