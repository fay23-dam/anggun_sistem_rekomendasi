import os
import fitz
import re
import time
import logging
import difflib
from typing import Tuple, Optional, List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import tiktoken
from urllib.parse import quote

# ======================
# LOGGING CONFIGURATION
# ======================
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("skincare_assistant.log")
        ]
    )
    return logging.getLogger(__name__)

logger = configure_logging()
load_dotenv()

# ======================
# CONSTANTS
# ======================
SKIN_TYPES = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi', 'kemerahan', 'kusam']
SKIN_SYNONYMS = {
    'jerawat': 'berjerawat',
    'acne': 'berjerawat',
    'acne prone': 'berjerawat',
    'berjerawat': 'berjerawat',
    'kering': 'kering',
    'dry': 'kering',
    'dehidrasi': 'kering',
    'berminyak': 'berminyak',
    'oily': 'berminyak',
    'minyak': 'berminyak',
    'sensitif': 'sensitif',
    'sensitive': 'sensitif',
    'iritasi': 'sensitif',
    'normal': 'normal',
    'sehat': 'normal',
    'kombinasi': 'kombinasi',
    'combination': 'kombinasi',
    'T-zone': 'kombinasi',
    'kemerahan': 'kemerahan',
    'redness': 'kemerahan',
    'rosacea': 'kemerahan',
    'kusam': 'kusam',
    'dull': 'kusam'
}

# Pemisah untuk jenis kulit kombinasi
SKIN_TYPE_SEPARATORS = ['/', 'dan', '&', 'serta', '-', 'atau', '+']

# Hanya produk ini yang didukung
SUPPORTED_PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                           "sunscreen", "tabir surya", "serum"]
PRODUCT_SYNONYMS = {
    'cuci muka': 'facial wash',
    'pembersih': 'cleanser',
    'moisturizer': 'pelembap',
    'sunscreen': 'tabir surya',
    'sunblock': 'tabir surya',
    'essence': 'serum',
    'toner': 'toner'
}

GREETINGS = ['halo', 'hai', 'hello', 'hi', 'apa kabar', 'selamat pagi', 
             'selamat siang', 'selamat sore', 'selamat malam']

QUESTION_TYPES = {
    'manfaat': ['manfaat', 'kegunaan', 'guna', 'fungsi'],
    'kandungan': ['kandungan', 'ingredient', 'komposisi', 'bahan'],
    'cara_pakai': ['cara pakai', 'cara penggunaan', 'penggunaan', 'pemakaian'],
    'harga': ['harga', 'price', 'berapa harga', 'berapa price'],
    'lokasi': ['lokasi', 'tempat beli', 'beli dimana', 'tempat pembelian'],
    'rekomendasi': ['rekomendasi', 'sarankan', 'anjuran', 'saran']
}

COMBINED_TYPES = {
    'kandungan_manfaat': ['kandungan dan manfaat', 'komposisi dan kegunaan'],
    'cara_manfaat': ['cara pakai dan manfaat', 'penggunaan dan kegunaan'],
    'semua_info': ['semua informasi', 'semua info', 'semua data']
}

# ======================
# BERT SCORE UTILITIES
# ======================
class BERTScorer:
    def __init__(self, model_name='indobenchmark/indobert-base-p1'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for text"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean pooling of last hidden states
        last_hidden = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden, dim=1)
        return embeddings.cpu()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using BERT embeddings"""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        
        # Convert to numpy arrays and calculate cosine similarity
        vector1 = emb1.numpy().flatten()
        vector2 = emb2.numpy().flatten()
        
        # Handle zero vectors
        if (vector1 == 0).all() or (vector2 == 0).all():
            return 0.0
            
        return 1 - cosine(vector1, vector2)
    
    def evaluate_answer(self, system_answer: str, reference: str) -> float:
        """Evaluate system answer against reference answer"""
        return self.calculate_similarity(system_answer, reference)

# ======================
# FUZZY MATCHING UTILS
# ======================
def fuzzy_match(input_str, options, threshold=0.7, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    # Preprocess input
    input_str = input_str.lower().strip()
    
    # Pertama, coba pencocokan eksak (case-insensitive)
    exact_matches = [opt for opt in options if opt.lower() == input_str]
    if exact_matches:
        return exact_matches[0] if not get_all else exact_matches
    
    # Kedua, coba cari substring
    substring_matches = [opt for opt in options if input_str in opt.lower() or opt.lower() in input_str]
    if substring_matches:
        return substring_matches[0] if not get_all else substring_matches
    
    # Ketiga, gunakan difflib dengan partial ratio
    normalized_options = [opt.lower() for opt in options]
    matches = []
    
    for opt in normalized_options:
        # Hitung similarity dengan SequenceMatcher
        seq_ratio = difflib.SequenceMatcher(None, input_str, opt).ratio()
        
        # Hitung partial ratio
        partial_ratio = 0
        for i in range(len(input_str) - len(opt) + 1):
            substring = input_str[i:i+len(opt)]
            ratio = difflib.SequenceMatcher(None, substring, opt).ratio()
            if ratio > partial_ratio:
                partial_ratio = ratio
                
        # Ambil nilai tertinggi antara seq_ratio dan partial_ratio
        score = max(seq_ratio, partial_ratio)
        if score >= threshold:
            matches.append((opt, score))
    
    # Urutkan berdasarkan skor tertinggi
    matches.sort(key=lambda x: x[1], reverse=True)
    matched_options = [opt for opt, score in matches]
    
    if get_all:
        # Kembalikan semua match dengan casing asli
        return [next((o for o in options if o.lower() == m), m) for m in matched_options]
    
    if matched_options:
        # Kembalikan match terbaik dengan casing asli
        best_match = matched_options[0]
        return next((opt for opt in options if opt.lower() == best_match), best_match)
    
    return None

# ======================
# CONVERSATION CONTEXT (Perbaikan)
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []  # Ubah menjadi list untuk mendukung multiple skin types
        self.product_type = None
        self.last_product = None
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        self.requested_count = None  # Jumlah produk yang diminta
        self.ingredient_filter = None  # Filter kandungan
        self.price_filter = None  # Filter harga
        self.context_maintained = False  # Menandai apakah konteks dipertahankan
        self.recommended_products = set()  # Produk yang sudah direkomendasikan
        self.recommendation_context = ""   # Konteks rekomendasi terakhir
        self.need_reset_filters = False
        if product_names:
            for name in product_names:
                self.add_product(name)
                
    def set_skin_type(self, skin_type):
        """Simpan jenis kulit yang telah disebutkan"""
        self.skin_type = skin_type
        self.context_maintained = True

    def get_skin_type(self):
        """Ambil jenis kulit yang telah disimpan"""
        return self.skin_type

    def update(self, question: str, response: str):
        if 'rekomendasi lainnya' not in question.lower():
            self.need_reset_filters = True
        self.last_question = question
        self._extract_context(question, response)
    def _detect_question_type(self, question: str) -> str:
        q_lower = question.lower()
        
        for q_type, keywords in QUESTION_TYPES.items():
            if any(kw in q_lower for kw in keywords):
                return q_type
                
        for q_type, keywords in COMBINED_TYPES.items():
            if any(kw in q_lower for kw in keywords):
                return q_type
                
        return "general"
        
    def _extract_context(self, question: str, response: str):
        self.need_skin_type = False
        self.need_product_type = False
        self.context_maintained = False
        
        # Simpan nilai sebelumnya untuk perbandingan
        old_skin_type = self.skin_type.copy() if self.skin_type else []
        old_product_type = self.product_type
        
        # Ekstrak jenis kulit dari pertanyaan
        new_skin_types = self._extract_skin_type(question)
        
        # PERBAIKAN: Gunakan new_skin_types jika ada, jika tidak gunakan yang lama
        if new_skin_types:
            self.skin_type = new_skin_types
            logger.info(f"Updated skin types: {self.skin_type}")
        else:
            logger.info(f"Maintaining existing skin types: {self.skin_type}")
            self.context_maintained = True
        
        # Ekstrak tipe produk
        self._extract_product_type(question)
        
        # Periksa perubahan kriteria utama
        skin_changed = (
            (old_skin_type != self.skin_type) and  # PERBAIKAN: perbandingan langsung
            (self.skin_type is not None)
        )
        product_type_changed = old_product_type != self.product_type
        
        # Reset filter jika ada perubahan jenis kulit atau tipe produk
        if skin_changed or product_type_changed:
            self.ingredient_filter = None
            self.price_filter = None
            logger.info("Reset filters due to criteria change")
        
        # Ekstrak nama produk
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
            
        # Reset rekomendasi jika ada perubahan signifikan pada kriteria
        if skin_changed or product_type_changed:
            self.recommended_products = set()
            self.recommendation_context = ""
            logger.info("Reset recommended products due to criteria change")
        else:
            self.context_maintained = True
            logger.info("Maintaining recommendation context for filters")
    
    def _extract_skin_type(self, question: str):
        q_lower = question.lower()
        detected_skins = set()
        
        # 1. Normalisasi pertanyaan dengan mengganti sinonim
        for syn, skin in SKIN_SYNONYMS.items():
            # Gunakan regex untuk mengganti sinonim dengan bentuk standar
            q_lower = re.sub(rf'\b{re.escape(syn)}\b', skin, q_lower)
        
        # 2. Gunakan regex untuk deteksi yang lebih akurat dengan bentuk standar
        skin_patterns = {
            'berjerawat': r'\b(berjerawat)\b',
            'kering': r'\b(kering)\b',
            'berminyak': r'\b(berminyak)\b',
            'sensitif': r'\b(sensitif)\b',
            'kombinasi': r'\b(kombinasi)\b',
            'normal': r'\b(normal)\b',
            'kemerahan': r'\b(kemerahan)\b',
            'kusam': r'\b(kusam)\b'
        }
        
        for skin_type, pattern in skin_patterns.items():
            if re.search(pattern, q_lower):
                detected_skins.add(skin_type)
        
        # 3. Jika belum terdeteksi, coba fuzzy matching dengan threshold rendah
        if not detected_skins:
            words = re.findall(r'\b\w+\b', q_lower)
            for word in words:
                # Skip kata-kata pendek yang tidak relevan
                if len(word) < 5:
                    continue
                    
                # Fuzzy match dengan threshold rendah
                match = fuzzy_match(word, SKIN_TYPES, threshold=0.8)
                if match:
                    detected_skins.add(match)
                    logger.info(f"Fuzzy matched skin type: '{word}' -> '{match}'")
        
        # 4. Deteksi kombinasi kulit (misal: "kering dan sensitif")
        if not detected_skins:
            for separator in SKIN_TYPE_SEPARATORS:
                parts = [p.strip() for p in q_lower.split(separator)]
                for part in parts:
                    if len(part) < 5:
                        continue
                    match = fuzzy_match(part, SKIN_TYPES, threshold=0.8)
                    if match:
                        detected_skins.add(match)
        
        # 5. Pastikan hanya jenis kulit yang valid
        valid_skins = set(SKIN_TYPES)
        detected_skins = {s for s in detected_skins if s in valid_skins}
        
        # PERBAIKAN: Return None jika tidak ditemukan jenis kulit baru
        if not detected_skins:
            return None
        
        return list(detected_skins)

    def _extract_product_type(self, question: str):
        # Cari sinonim terlebih dahulu
        q_lower = question.lower()
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type via synonym: {syn} -> {pt}")
                return
                
        # Gunakan regex untuk deteksi yang lebih akurat
        product_patterns = {
            'sunscreen': r'\b(sunscreen|tabir surya|sunblock)\b',
            'moisturizer': r'\b(moisturizer|pelembap)\b',
            'cleanser': r'\b(cleanser|facial wash|cuci muka|pembersih)\b',
            'serum': r'\b(serum|essence)\b'
        }
        
        for product_type, pattern in product_patterns.items():
            if re.search(pattern, q_lower):
                self.product_type = product_type
                logger.info(f"Detected product type via regex: {product_type}")
                return
                
        # Cari eksak di daftar product type
        for pt in SUPPORTED_PRODUCT_TYPES:
            if pt in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(q_lower, SUPPORTED_PRODUCT_TYPES, threshold=0.7)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            
    def _extract_product_name(self, text: str) -> Optional[str]:
        # Pola regex yang mempertahankan tanda kurung dan karakter khusus
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)',  # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+() ]{5,}(?:\s+[A-Z0-9&.\-+() ]{3,})*)'  # Izinkan tanda kurung
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, text, re.IGNORECASE):
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                if self.is_valid_product(name):
                    # Cari casing asli yang cocok
                    for p in self.all_products:
                        if p.lower() == name.lower():
                            return p
                
        # Jika tidak ditemukan, coba dengan metode lain
        return None
    
    def get_context_summary(self) -> str:
        summary = []
        if self.skin_type: summary.append(f"Kulit: {', '.join(self.skin_type)}")
        if self.product_type: summary.append(f"Produk: {self.product_type}")
        if self.last_product: summary.append(f"Terakhir: {self.last_product[:20]}")
        return ", ".join(summary)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return any(p.lower() == product_name.lower() for p in self.all_products)
    
    def set_last_info_types(self, info_types: List[str]):
        self.last_info_types = info_types
        
    def get_last_info_types(self) -> List[str]:
        return self.last_info_types

# ======================
# PDF PROCESSOR
# ======================
class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024,
            chunk_size=1000,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "skincare_index"
        self.MAX_TEXT_LENGTH = 1_000_000
        self.product_names = set()
        self.product_data = {}
        self.full_text = ""
        self.recommendation_reference = {}
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        self.full_text = self.extract_text(pdf_path)
        if not self.full_text:
            logger.error("No text extracted from PDF")
            return False
        
        self._extract_all_product_data(self.full_text)
        self._extract_recommendation_references()
        
        if os.path.exists(index_path):
            try:
                logger.info("Loading existing vector store")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        
        return self._create_new_vector_store()
    
    def _create_new_vector_store(self) -> bool:
        chunks = self.chunk_text(self.full_text)
        if not chunks:
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                save_path = os.path.join(self.vector_store_dir, self.index_name)
                self.vector_store.save_local(save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
                return False
        return False

    def _extract_all_product_data(self, text: str):
        # Split sections by "Produk:" using lookahead
        product_sections = re.split(r'(?=\nProduk:\s*)', text)
        
        for section in product_sections:
            if not section.strip():
                continue
                
            # Tangkap nama produk (bisa multi-line)
            name_match = re.search(
                r'Produk:\s*((?:.|\n)*?)\s*(?=harga:|type produk:|jenis kulit:|informasi produk:|kandungan:|cara pemakaian:|manfaat:|lokasi :|\nProduk:|$)',
                section, 
                re.IGNORECASE
            )
            
            if name_match:
                name = name_match.group(1).strip()
                # Bersihkan baris baru dan spasi berlebihan
                name = re.sub(r'\s+', ' ', name)
                
                # Jika nama terlalu pendek, skip
                if len(name) < 3 or not any(c.isalpha() for c in name):
                    continue
                    
                # Ekstrak jenis kulit dengan support kombinasi
                jenis_kulit_text = self._extract_field(section, 'jenis kulit:').lower()
                
                # Pisahkan jenis kulit kombinasi
                skin_types = []
                found_separator = False
                for separator in SKIN_TYPE_SEPARATORS:
                    if separator in jenis_kulit_text:
                        parts = re.split(f'\\s*{re.escape(separator)}\\s*', jenis_kulit_text)
                        skin_types = [p.strip() for p in parts]
                        found_separator = True
                        break
                
                if not found_separator:
                    skin_types = [jenis_kulit_text.strip()]
                
                # Normalisasi setiap jenis kulit
                normalized_skin_list = []
                skin_mapping = {
                    'berjerawat': ['berjerawat', 'jerawat', 'acne prone', 'acne'],
                    'kering': ['kering', 'dry'],
                    'berminyak': ['berminyak', 'oily'],
                    'sensitif': ['sensitif', 'sensitive'],
                    'kombinasi': ['kombinasi', 'combination'],
                    'normal': ['normal'],
                    'semua': ['semua jenis', 'semua jenis kulit', 'all skin types']
                }
                
                for skin in skin_types:
                    found = False
                    for key, values in skin_mapping.items():
                        if any(value in skin for value in values):
                            normalized_skin_list.append(key)
                            found = True
                            break
                    if not found and skin:
                        normalized_skin_list.append(skin)
                
                # Ekstrak tipe produk
                type_produk = self._extract_field(section, 'type produk:').lower()
                
                # Normalisasi tipe produk
                type_mapping = {
                    'sunscreen': ['sunscreen', 'tabir surya', 'sunblock'],
                    'moisturizer': ['moisturizer', 'pelembap'],
                    'cleanser': ['cleanser', 'facial wash', 'pembersih'],
                    'serum': ['serum', 'essence']
                }
                
                normalized_type = 'lain'
                for key, values in type_mapping.items():
                    if any(value in type_produk for value in values):
                        normalized_type = key
                        break
                
                # Buat data produk
                product_info = {
                    'name': name,
                    'harga': self._extract_field(section, 'harga:'),
                    'type_produk': normalized_type,
                    'jenis_kulit': list(set(normalized_skin_list)),  # Hapus duplikat
                    'informasi_produk': self._extract_field(section, 'informasi produk:'),
                    'kandungan': self._extract_field(section, 'kandungan:'),
                    'cara_pemakaian': self._extract_field(section, 'cara pemakaian:'),
                    'manfaat': self._extract_field(section, 'manfaat:'),
                    'lokasi': self._extract_field(section, 'lokasi :')
                }
                
                # Simpan dengan casing asli
                self.product_names.add(name)
                self.product_data[name.lower()] = product_info
                logger.info(f"Extracted product: {name} | Skin: {product_info['jenis_kulit']} | Type: {normalized_type}")
        
        logger.info(f"Extracted {len(self.product_names)} product names from PDF")
    
    def _extract_recommendation_references(self):
        """Extract recommendation references from the document"""
        # Find recommendation sections
        recommendation_sections = re.findall(
            r'(Rekomendasi untuk kulit ([\w\s]+):([\w\s,]+))', 
            self.full_text, 
            re.IGNORECASE
        )
        
        for full_text, skin_type, products in recommendation_sections:
            skin_type = skin_type.strip().lower()
            product_list = [p.strip() for p in products.split(',')]
            self.recommendation_reference[skin_type] = product_list
            logger.info(f"Found recommendation for {skin_type}: {product_list}")

    def _extract_field(self, text: str, field_name: str) -> str:
        try:
            # Case-insensitive search for field
            pattern = re.compile(re.escape(field_name), re.IGNORECASE)
            match = pattern.search(text)
            if not match:
                return ""
            
            start_idx = match.end()
            remaining_text = text[start_idx:]
            
            # Find the next field or end of section
            end_patterns = [
                r'\n\s*(harga:|type produk:|jenis kulit:|informasi produk:|kandungan:|cara pemakaian:|manfaat:|lokasi :)',
                r'\n\s*[A-Z][a-z]+:',
                r'\n\s*Produk:'
            ]
            
            end_idx = len(remaining_text)
            for pat in end_patterns:
                end_match = re.search(pat, remaining_text, re.IGNORECASE)
                if end_match and end_match.start() < end_idx:
                    end_idx = end_match.start()
            
            content = remaining_text[:end_idx].strip()
            
            # Remove leading numbers and bullet points
            content = re.sub(r'^[\d\s•\-\.]+', '', content)
            return content
        except Exception as e:
            logger.error(f"Error extracting field {field_name}: {e}")
            return ""

    def get_product_names(self) -> List[str]:
        return list(self.product_names)
    
    def get_product_data(self, product_name: str) -> Optional[Dict]:
        # Ambil data dengan casing asli
        return self.product_data.get(product_name.lower())
    
    def get_reference_text(self, product_name: str, info_type: str) -> str:
        """Get reference text from document for a specific product and info type"""
        if not self.full_text:
            return ""
            
        # Find product section
        pattern = re.compile(rf'\nProduk:\s*{re.escape(product_name)}\s*\n', re.IGNORECASE)
        match = pattern.search(self.full_text)
        if not match:
            return ""
        
        start_idx = match.start()
        remaining_text = self.full_text[start_idx:]
        
        # Find end of product section
        end_match = re.search(r'\nProduk:\s*[^\n]+\n', remaining_text[1:])
        if end_match:
            section = remaining_text[:end_match.start()+1]
        else:
            section = remaining_text
            
        # Extract specific field
        field_names = {
            'manfaat': 'manfaat',
            'kandungan': 'kandungan',
            'cara_pakai': 'cara pemakaian',
            'harga': 'harga',
            'lokasi': 'lokasi'
        }
        
        field_key = field_names.get(info_type)
        if not field_key:
            return section
            
        field_text = self._extract_field(section, field_key + ':')
        return field_text or section
    
    def get_recommendation_reference(self, skin_type: str) -> List[str]:
        """Get reference recommendations for a specific skin type"""
        return self.recommendation_reference.get(skin_type.lower(), [])

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > self.MAX_TEXT_LENGTH:
                        break
            return text.strip() if text.strip() else None
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            )
            return splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error during chunk creation: {e}")
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            return

        documents = []
        for chunk in chunks:
            metadata = {}
            skin_match = re.search(r'jenis kulit:\s*([^\n]+)', chunk, re.IGNORECASE)
            if skin_match:
                metadata['skin_type'] = skin_match.group(1).strip().lower()
            
            product_match = re.search(r'type produk:\s*([^\n]+)', chunk, re.IGNORECASE)
            if product_match:
                metadata['product_type'] = product_match.group(1).strip().lower()
            
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        batch_size = 200
        total_chunks = len(documents)
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="Creating embeddings"):
            batch = documents[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    self.vector_store.add_documents(batch)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

# ======================
# QA ENGINE
# ======================
class QAEngine:
    def __init__(self, vector_store, processor: PDFProcessor):
        self.vector_store = vector_store
        self.processor = processor
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0.7,
            max_tokens=2000
        )
        self.context = ConversationContext(processor.get_product_names())
        self.bertscorer = BERTScorer()
        self.evaluation_results = []
        

    def _generate_product_cards(self, products: List[Dict]) -> str:
        cards = []
        for product in products:
            name = product['name']
            
            # Encode khusus untuk nama file gambar (ganti % dengan %25)
            image_name = name.replace('%', '%25')  # Hanya encoding khusus untuk %
            
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{image_name}.jpg" alt="{name}" class="product-image">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
            </div>
            """
            cards.append(card)
        return "\n".join(cards)

    def _extract_info_types(self, question: str) -> List[str]:
        """Ekstrak tipe informasi yang diminta secara eksplisit"""
        info_types = []
        q_lower = question.lower()
        
        for q_type, keywords in QUESTION_TYPES.items():
            if any(kw in q_lower for kw in keywords):
                info_types.append(q_type)
                
        return info_types

    def _extract_filters(self, question: str):
        """Ekstrak filter dari pertanyaan (kandungan, harga, jumlah)"""
        q_lower = question.lower()
        
        # Deteksi filter kandungan
        ingredient_match = re.search(r'(mengandung|dengan kandungan|kandungan)\s+([\w\s]+)', q_lower)
        if ingredient_match:
            self.context.ingredient_filter = ingredient_match.group(2).strip()
            logger.info(f"Extracted ingredient filter: {self.context.ingredient_filter}")
        
        # Deteksi filter harga
        price_match = re.search(
            r'(harga\s+)?(dibawah|di bawah|di atas|diatas|kurang|lebih|sampai|hingga|>|<|>=|<=)?\s*([\d\.]+)\s*(ribu|rbu|rb|k)?', 
            q_lower
        )
        if price_match:
            operator = price_match.group(2).lower() if price_match.group(2) else ""
            price_value = float(price_match.group(3).replace('.', ''))
            
            # Handle satuan ribu (100rb = 100000)
            if price_match.group(4) and price_match.group(4) in ['ribu', 'rbu', 'rb', 'k']:
                price_value *= 1000
                
            self.context.price_filter = (operator, price_value)
            logger.info(f"Extracted price filter: {operator} {price_value}")
        
        # Deteksi jumlah produk
        num_match = re.search(r'(\d+)\s+produk', q_lower)
        if num_match:
            self.context.requested_count = int(num_match.group(1))
            logger.info(f"Extracted requested count: {self.context.requested_count}")

    def _guess_product_name(self, question: str, product_name: str = "") -> Optional[str]:
        """Use GPT to guess product name from incomplete names using document context"""
        try:
            # Ambil daftar produk (maksimal 100 untuk efisiensi)
            product_list = list(self.context.all_products)
            if not product_list:
                return None
                
            # Batasi jumlah produk jika terlalu banyak
            if len(product_list) > 100:
                product_list = product_list[:100]
            
            prompt = f"""
            Anda adalah asisten skincare yang membantu mengidentifikasi produk dari pertanyaan pengguna.
            Daftar produk yang tersedia: {', '.join(product_list)}
            
            Pertanyaan pengguna: "{question}"
            
            Berdasarkan daftar produk di atas, produk mana yang paling mungkin dimaksud oleh pengguna?
            Jawab HANYA dengan nama produk yang lengkap. Jika tidak yakin, jawab dengan 'Tidak Diketahui'.
            
            Pertimbangkan:
            1. Kemiripan nama produk
            2. Konteks pertanyaan
            3. Produk yang sering disebut sebelumnya
            """
            
            # Jika ada petunjuk nama produk, tambahkan ke prompt
            if product_name:
                prompt = f"""
                Anda adalah asisten skincare yang membantu mengidentifikasi produk dari pertanyaan pengguna.
                Daftar produk yang tersedia: {', '.join(product_list)}
                
                Pengguna menyebut: "{product_name}" dalam konteks: "{question}"
                
                Berdasarkan daftar produk di atas, produk mana yang paling mungkin dimaksud oleh pengguna?
                Jawab HANYA dengan nama produk yang lengkap. Jika tidak yakin, jawab dengan 'Tidak Diketahui'.
                """
            
            response = self.llm.invoke(prompt).content.strip()
            
            # Bersihkan respons
            response = re.sub(r'^"|"$', '', response)  # Hapus tanda kutip jika ada
            
            # Periksa apakah respons adalah produk yang valid
            if any(p.lower() == response.lower() for p in self.context.all_products):
                return next(p for p in self.context.all_products if p.lower() == response.lower())
                    
            # Jika GPT tidak yakin, cari dengan similarity
            best_match = None
            best_score = 0
            
            for product in self.context.all_products:
                score = difflib.SequenceMatcher(
                    None, 
                    response.lower(), 
                    product.lower()
                ).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = product
                    
            if best_score > 0.7:  # Threshold kecocokan
                return best_match
                    
            return None
            
        except Exception as e:
            logger.error(f"Product guessing error: {e}")
            return None

    def analyze_question(self, question: str) -> Tuple[str, Optional[str], List[str]]:
        """Mengembalikan (question_type, product, info_types)"""
        info_types = []
        q_lower = question.lower()

        # 1. Handle pertanyaan kosong atau tidak valid
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
        
        # 2. Ekstrak filter dari pertanyaan
        self._extract_filters(question)
        
        # 3. Ekstrak produk SEKARANG untuk digunakan di logika berikutnya
        product = self._extract_product_from_question(question)
        
        # 4. Handle special cases
        if any(greeting in q_lower for greeting in GREETINGS):
            if product:
                return ("product_info", product, ['manfaat'])
            return ("sapaan", None, [])
        
        # 5. Deteksi rekomendasi lainnya (follow-up)
        rekomendasi_lain_keywords = ['rekomendasi lainnya', 'rekomendasi lain', 'lainnya', 'selanjutnya', 'lagi']
        if any(kw in q_lower for kw in rekomendasi_lain_keywords):
            return ("rekomendasi_lain", None, [])
        
        # 6. Gunakan GPT untuk klasifikasi pertanyaan
        classification = self._classify_question_with_gpt(question)
        logger.info(f"GPT classification: {classification}")
        
        # 7. Deteksi info_types
        product_info_keywords = {
            'manfaat': ['manfaat', 'kegunaan', 'guna', 'fungsi', 'untuk apa', 'guna nya'],
            'kandungan': ['kandungan', 'ingredient', 'komposisi', 'bahan', 'isi', 'content'],
            'cara_pakai': ['cara pakai', 'cara penggunaan', 'penggunaan', 'pemakaian', 'bagaimana pakai', 'instruksi'],
            'harga': ['harga', 'price', 'berapa harga', 'berapa price', 'biaya', 'cost'],
            'lokasi': ['lokasi', 'tempat beli', 'beli dimana', 'tempat pembelian', 'dijual dimana', 'pembelian', 'outlet']
        }
        
        detected_info_types = []
        for info_type, keywords in product_info_keywords.items():
            if any(kw in q_lower for kw in keywords):
                detected_info_types.append(info_type)
        
        # 8. Prioritaskan rekomendasi jika GPT mengklasifikasikan sebagai rekomendasi
        if classification == "rekomendasi":
            return ("rekomendasi", None, [])
        
        # 9. Handle product_info jika ada info_types dan produk
        if detected_info_types:
            if product:
                # Periksa apakah produk benar-benar produk spesifik atau hanya tipe produk
                if product.lower() in SUPPORTED_PRODUCT_TYPES:
                    # Ini sebenarnya tipe produk, bukan produk spesifik
                    return ("rekomendasi", None, [])
                return ("product_info", product, detected_info_types)
            elif self.context.last_product:
                return ("product_info", self.context.last_product, detected_info_types)
        
        # 10. Deteksi jenis kulit
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            if not self.context.skin_type:
                self.context.skin_type = []
            self.context.skin_type.append(skin_match)
            
            if self.context.product_type:
                return ("rekomendasi", None, [])
            return ("need_product_type", None, [])
        
        # 11. Handle product_card jika hanya menyebut produk
        if product and not detected_info_types:
            return ("product_card", product, [])
        
        # 12. Fallback ke hasil klasifikasi GPT
        if classification == "product_info":
            if product:
                return ("product_info", product, ['manfaat'])
            elif self.context.last_product:
                return ("product_info", self.context.last_product, ['manfaat'])
        
        return ("general", None, [])

    def _classify_question_with_gpt(self, question: str) -> str:
        """Klasifikasikan pertanyaan menggunakan GPT"""
        prompt = f"""
        Anda adalah asisten yang membantu mengklasifikasikan pertanyaan tentang skincare.
        Pertanyaan: "{question}"
        
        Klasifikasikan pertanyaan ini ke dalam salah satu kategori berikut:
        1. "rekomendasi" - jika pengguna meminta rekomendasi produk berdasarkan kriteria
        2. "product_info" - jika pengguna menanyakan informasi tentang produk tertentu
        
        Pertimbangkan:
        - Jika pertanyaan mengandung kata kunci seperti 'rekomendasi', 'sarankan', 'untuk kulit [jenis]', 
        'dengan kriteria', atau meminta saran produk -> "rekomendasi"
        - Jika pertanyaan menyebut nama produk spesifik dan menanyakan detail seperti manfaat, kandungan, harga -> "product_info"
        - Jika tidak ada petunjuk yang jelas, kembalikan "unknown"
        
        Jawab HANYA dengan salah satu dari: "rekomendasi", "product_info", "unknown"
        """
        
        try:
            response = self.llm.invoke(prompt).content.strip().lower()
            if "rekomendasi" in response:
                return "rekomendasi"
            elif "product_info" in response or "produk" in response:
                return "product_info"
            return "unknown"
        except Exception as e:
            logger.error(f"GPT classification error: {e}")
            return "unknown"
        
    def _extract_product_from_question(self, question: str) -> Optional[str]:
        # Pola regex untuk nama produk uppercase dengan tanda kurung
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)', # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+() ]{5,}(?:\s+[A-Z0-9&.\-+() ]{3,})*)'  # Izinkan tanda kurung
        ]
        
        # Coba ekstrak dengan pola regex
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                # Gunakan fuzzy matching untuk verifikasi (case-insensitive)
                matched = fuzzy_match(name, list(self.context.all_products), threshold=0.7)
                if matched:
                    return matched
        
        # Jika tidak ditemukan, cari dengan similarity (case-insensitive)
        words = re.findall(r'\b[\w&.\-+]{3,}\b', question)
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                if len(phrase) < 5: 
                    continue
                    
                # Fuzzy match with lower threshold
                match = fuzzy_match(phrase, list(self.context.all_products), threshold=0.7)
                if match:
                    return match
        
        return None

    def get_recommendations(self, query: str) -> str:
        try:
            # Dapatkan semua produk yang tersedia
            if 'rekomendasi lainnya' in query.lower():
                # Preserve existing filters (ingredient, price, etc.)
                logger.info(f"Maintaining filters for follow-up recommendations: {self.context.ingredient_filter}, {self.context.price_filter}")
                self.context.need_reset_filters = False  # Don't reset the filters here
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data and product_data.get('type_produk') in SUPPORTED_PRODUCT_TYPES:
                    all_products.append({
                        'name': product_name,  # Gunakan nama asli
                        'data': product_data
                    })
            
            # Log untuk debugging
            logger.info(f"Jumlah produk tersedia: {len(all_products)}")
            logger.info(f"Kriteria filter - Kulit: {self.context.skin_type}, Produk: {self.context.product_type}, Kandungan: {self.context.ingredient_filter}, Harga: {self.context.price_filter}")
            
            # Normalisasi jenis kulit yang diminta user
            user_skins = [SKIN_SYNONYMS.get(st.lower(), st.lower()) for st in self.context.skin_type] if self.context.skin_type else []
            logger.info(f"Normalized skin types: {user_skins}")
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                skin_data = [s.lower() for s in product['data'].get('jenis_kulit', [])]
                type_data = product['data'].get('type_produk', '').lower()
                
                # Skip produk yang jenisnya tidak didukung
                if type_data not in SUPPORTED_PRODUCT_TYPES:
                    continue
                
                # Logika filter jenis kulit untuk kombinasi
                skin_match = False
                if not user_skins:  # Tidak ada filter kulit
                    skin_match = True
                else:
                    # Jika produk cocok untuk semua jenis kulit
                    if 'semua' in skin_data:
                        skin_match = True
                    else:
                        # Minimal satu jenis kulit yang diminta cocok dengan produk
                        any_match = any(user_skin in skin_data for user_skin in user_skins)
                        skin_match = any_match
                
                # Logika filter jenis produk
                product_match = False
                if not self.context.product_type:
                    product_match = True
                else:
                    # Cari semua kemungkinan tipe produk
                    product_types = [self.context.product_type]
                    if self.context.product_type in PRODUCT_SYNONYMS.values():
                        # Tambahkan sinonim
                        product_types.extend([k for k, v in PRODUCT_SYNONYMS.items() if v == self.context.product_type])
                    
                    # Cek kecocokan
                    for pt in product_types:
                        if pt.lower() == type_data:
                            product_match = True
                            break
                
                # Filter kandungan - PERBAIKAN UTAMA: gunakan pencocokan substring
                ingredient_match = True
                if self.context.ingredient_filter:
                    kandungan = product['data'].get('kandungan', '').lower()
                    # Pisahkan menjadi kata-kata individual
                    ingredient_words = self.context.ingredient_filter.lower().split()
                    # Cek apakah semua kata ada dalam kandungan
                    ingredient_match = all(word in kandungan for word in ingredient_words)
                    
                # Filter harga
                price_match = True
                if self.context.price_filter:
                    harga_text = product['data'].get('harga', '0')
                    try:
                        # Ekstrak angka dari teks harga (dukungan format Rp 100.000 dan 100000)
                        harga_str = re.search(r'[\d,\.]+', harga_text.replace('.', '')).group().replace(',', '')
                        harga = float(harga_str)
                    except:
                        harga = float('inf')  # Tidak bisa diparsing, anggap mahal
                        
                    operator, value = self.context.price_filter
                    if operator in ["dibawah", "di bawah", "kurang", "<"]:
                        price_match = harga < value
                    elif operator in ["di atas", "diatas", "lebih", ">"]:
                        price_match = harga > value
                    elif operator in ["sampai", "hingga", "-"]:
                        price_match = harga <= value
                    else:  # Default: harga tepat (tidak ada operator)
                        price_match = harga == value
                
                if skin_match and product_match and ingredient_match and price_match:
                    filtered_products.append(product)
                    logger.info(f"Product matched: {product['name']} | Skin: {skin_data} | Type: {type_data}")
            
            # Buat kunci konteks baru
            new_context_key = (
                f"{','.join(sorted(user_skins))}-"
                f"{self.context.product_type}-"
                f"{self.context.ingredient_filter}-"
                f"{str(self.context.price_filter)}"
            )
            
            # Reset hanya jika ada perubahan signifikan
            if new_context_key != self.context.recommendation_context:
                self.context.recommended_products = set()
                self.context.recommendation_context = new_context_key
                logger.info(f"New recommendation context: {new_context_key}")
            
            # Filter produk yang belum direkomendasikan
            filtered_products = [p for p in filtered_products 
                                if p['name'] not in self.context.recommended_products]
            
            # Batasi jumlah produk jika diminta
            requested_count = self.context.requested_count or 5
            if len(filtered_products) > requested_count:
                # Prioritaskan produk dengan kemiripan tertinggi
                filtered_products = filtered_products[:requested_count]
            
            # Generate product cards
            products_to_show = [{'name': p['name']} for p in filtered_products]
            product_cards = self._generate_product_cards(products_to_show)
            
            # =============================================================
            # NEW RAG-BASED INTERACTIVE TEXT GENERATION
            # =============================================================
            rag_response = f""" """
            if filtered_products:
                # Build context description
                context_desc = []
                if user_skins:
                    skin_str = ", ".join(user_skins)
                    context_desc.append(f"kulit <span class='highlight'>{skin_str}</span>")
                if self.context.product_type:
                    context_desc.append(f"tipe produk <span class='highlight'>{self.context.product_type}</span>")
                if self.context.ingredient_filter:
                    context_desc.append(f"mengandung <span class='highlight'>{self.context.ingredient_filter}</span>")
                if self.context.price_filter:
                    operator, value = self.context.price_filter
                    value_str = f"Rp {value:,.0f}".replace(",", ".")
                    if operator in ['dibawah', 'di bawah', '<']:
                        context_desc.append(f"harga di bawah <span class='highlight'>{value_str}</span>")
                    elif operator in ['di atas', 'diatas', '>']:
                        context_desc.append(f"harga di atas <span class='highlight'>{value_str}</span>")
                    else:
                        context_desc.append(f"harga sekitar <span class='highlight'>{value_str}</span>")
                
                context_str = ", ".join(context_desc) if context_desc else "berbagai kriteria"
                
                # Get top 3 product names
                top_products = [p['name'].lower() for p in filtered_products[:3]]
                
                # Generate personalized message using LLM
                prompt = f"""
                Anda adalah asisten skincare yang ramah dan interaktif. Buatlah respons untuk rekomendasi produk dengan:
                - Konteks: {context_str}
                - Jumlah produk: {len(filtered_products)}
                - Produk teratas: {', '.join(top_products)}
                
                Struktur respons:
                Penjelasan singkat mengapa produk ini cocok dengan kriteria dan Ajakan untuk melihat detail produk
                
                Gunakan maksimal 3 kalimat. Format dalam HTML sederhana tanpa heading.
                """
                
                try:
                    rag_text = self.llm.invoke(prompt).content
                    rag_response = f"""
                    <div class="rag-response">
                        {rag_text}
                    </div>
                    """
                except Exception as e:
                    logger.error(f"RAG text generation error: {e}")
                    rag_response = f"""
                    <div class="rag-response">
                        <p>Saya menemukan {len(filtered_products)} produk yang cocok untuk {context_str}.</p>
                        <p>Produk-produk ini dipilih berdasarkan kecocokan dengan kebutuhan kulit Anda.</p>
                    </div>
                    """
            else:
                rag_response = """
                <div class="rag-response">
                    <h5>Tidak Ada Lagi Produk Yang Relevan <i class="sad-icon fas fa-frown"></i></h5>
                <p>Maaf, saya tidak menemukan produk yang sesuai dengan kriteria Anda.</p>
                </div>
                """
            
            # Simpan produk yang direkomendasikan
            for p in filtered_products:
                self.context.recommended_products.add(p['name'])
            
            # Response dengan jumlah produk aktual
            if filtered_products:
                response_html = f"""
                <div class="recommendation-container text-justify">
                    {rag_response}
                    <div class="product-grid" style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px;">
                        {product_cards}
                    </div>
                    <div class="recommendation-feedback">
                        <p>Bagaimana dengan rekomendasi ini?</p>
                        <div class="feedback-buttons">
                            <button class="feedback-btn more" data-feedback="more"><i class="fas fa-redo mr-6"></i> Rekomendasi Lainnya</button>
                        </div>
                        <p class="feedback-note">Atau jelaskan kebutuhan spesifik Anda untuk rekomendasi lebih akurat</p>
                    </div>
                </div>
                """
            else:
                response_html = f"""
                <div class="no-recommendation">
                    {rag_response}
                <div class="suggestion-box">
                <ul class="suggestion-list">
                  <li>
                    <i class="fa-solid fa-1"></i>
                    <span class="cursor-pointer suggesst-one">Moisturizer untuk kulit sensitif</span>
                  </li>
                  <li>
                    <i class="fa-solid fa-2"></i>
                    <span class="cursor-pointer suggesst-two">Sunscreen untuk kulit kering dibawah 100 ribu</span>
                  </li>
                  <li>
                    <i class="fa-solid fa-3"></i>
                    <span class="cursor-pointer suggesst-three">manfaat garnier sakura glow glowing face wash</span>
                  </li>
                </ul>
              </div>
                </div>
                """
            
            # Evaluate recommendations
            self._evaluate_recommendation(filtered_products)
            
            return response_html
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi."
    
    def _evaluate_recommendation(self, recommended_products: list):
        """Evaluate recommendations against document reference"""
        try:
            if not self.context.skin_type:
                return
                
            # Normalize skin type for reference lookup
            normalized_skin = SKIN_SYNONYMS.get(self.context.skin_type[0].lower(), self.context.skin_type[0].lower())
                
            # Get reference recommendations from document
            reference_products = self.processor.get_recommendation_reference(normalized_skin)
            if not reference_products:
                return
                
            # Convert to sets for comparison
            rec_set = {p['name'].lower() for p in recommended_products}
            ref_set = {p.lower() for p in reference_products}
            
            # Calculate precision and recall
            common = rec_set.intersection(ref_set)
            precision = len(common) / len(rec_set) if rec_set else 0
            recall = len(common) / len(ref_set) if ref_set else 0
            
            # Calculate BERTScore for recommendation text
            rec_text = ", ".join([p['name'] for p in recommended_products])
            ref_text = ", ".join(reference_products)
            bert_score = self.bertscorer.evaluate_answer(rec_text, ref_text)
            
            # Log evaluation result
            self.evaluation_results.append({
                'type': 'rekomendasi',
                'skin_type': self.context.skin_type[0],
                'precision': precision,
                'recall': recall,
                'bert_score': bert_score,
                'recommended': [p['name'] for p in recommended_products],
                'reference': reference_products
            })
            
            logger.info(f"Recommendation evaluation for {self.context.skin_type}: "
                         f"Precision={precision:.2f}, Recall={recall:.2f}, BERTScore={bert_score:.4f}")
            
        except Exception as e:
            logger.error(f"Recommendation evaluation error: {e}")

    def get_product_info(self, product_name: str, info_types: List[str]) -> str:
        try:
            if not product_name:
                return "❌ Produk tidak disebutkan"
            
            # Gunakan algoritma yang lebih toleran untuk pencocokan nama produk
            best_match = None
            best_score = 0
            threshold = 0.7  # Threshold untuk kecocokan
            
            for real_name in self.context.all_products:
                # Hitung similarity dengan SequenceMatcher
                score = difflib.SequenceMatcher(
                    None, 
                    product_name.lower(), 
                    real_name.lower()
                ).ratio()
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = real_name
            
            # Jika tidak ditemukan, coba cari dengan fuzzy matching
            if not best_match:
                best_match = fuzzy_match(
                    product_name, 
                    list(self.context.all_products), 
                    threshold=0.7
                )
            
            # Jika masih tidak ketemu, gunakan GPT untuk menebak
            if not best_match:
                guessed_name = self._guess_product_name(f"{product_name} - {info_types[0] if info_types else ''}")
                if guessed_name:
                    best_match = guessed_name
            
            if not best_match:
                # Cari kandidat produk yang mirip
                candidates = fuzzy_match(
                    product_name, 
                    list(self.context.all_products), 
                    threshold=0.7, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates]
                    )
                    
                    return f"""
                    <div class="candidate-container">
                        <p>Produk "{product_name}" tidak ditemukan. Mungkin maksud Anda:</p>
                        <div class="candidate-grid" style="display: flex; flex-wrap: wrap; gap: 15px;">
                            {candidate_cards}
                        </div>
                        <p>Silakan pilih salah satu atau sebutkan nama produk dengan lebih spesifik.</p>
                    </div>
                    """
                return f"❌ Produk '{product_name}' tidak ditemukan."
            
            # Gunakan nama asli dari data produk
            product_data = self.processor.get_product_data(best_match)
            if not product_data:
                return f"❌ Tidak menemukan informasi tentang {best_match}"
            
            # Mapping untuk tipe informasi
            INFO_MAPPING = {
                'kandungan': ('kandungan', 'Kandungan'),
                'manfaat': ('manfaat', 'Manfaat'),
                'cara_pakai': ('cara_pemakaian', 'Cara Pakai'),
                'harga': ('harga', 'Harga'),
                'lokasi': ('lokasi', 'Lokasi Pembelian')
            }
            
            # Filter hanya informasi yang diminta
            response_lines = []
            for info_type in info_types:
                if info_type in INFO_MAPPING:
                    field, title = INFO_MAPPING[info_type]
                    if product_data.get(field):
                        # Bersihkan konten dari bagian yang tidak relevan
                        cleaned_content = self._clean_content(
                            product_data[field],
                            [k for k in INFO_MAPPING.keys() if k != info_type]
                        )
                        response_lines.append(f"<b>{title}</b>:<br /><p class=\"text-justify\">{cleaned_content}</p>")
            
            response_text = "<br /><br />".join(response_lines) if response_lines else "❌ Tidak menemukan informasi yang diminta"
            
            # Tampilkan card produk setelah penjelasan teks
            product_card = self._generate_product_cards([{'name': best_match}])
            
            # Tambahkan informasi konteks
            context_info = ""
            if self.context.skin_type:
                context_info += f"<p>Jenis kulit: {', '.join(self.context.skin_type)}</p>"
            if self.context.product_type:
                context_info += f"<p>Jenis produk: {self.context.product_type}</p>"
            
            # Prepare for evaluation
            response_content = re.sub(r'<[^>]+>', '', response_text)  # Remove HTML tags
            self._evaluate_product_info(best_match, info_types, response_content)
            
            return f"""
            <div class="product-info-response">
                <div class="info-text">
                    {response_text}
                </div>
                <div class="product-card-container">
                    {product_card}
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Product info error: {e}")
            return f"❌ Tidak bisa menampilkan informasi untuk produk tersebut"
    
    def _evaluate_product_info(self, product_name: str, info_types: List[str], response: str):
        """Evaluate product info response against document content"""
        for info_type in info_types:
            try:
                reference_text = self.processor.get_reference_text(product_name, info_type)
                if reference_text:
                    score = self.bertscorer.evaluate_answer(response, reference_text)
                    
                    # Log evaluation result
                    self.evaluation_results.append({
                        'type': 'produk_info',
                        'product': product_name,
                        'info_type': info_type,
                        'bert_score': score,
                        'response': response[:200] + "..." if len(response) > 200 else response,
                        'reference': reference_text[:200] + "..." if len(reference_text) > 200 else reference_text
                    })
                    
                    logger.info(f"BERTScore for {product_name} ({info_type}): {score:.4f}")
            except Exception as e:
                logger.error(f"Product info evaluation error: {e}")

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Gunakan algoritma yang lebih toleran untuk pencocokan nama produk
        best_match = None
        best_score = 0
        threshold = 0.7  # Threshold untuk kecocokan
        
        for real_name in self.context.all_products:
            # Hitung similarity dengan SequenceMatcher
            score = difflib.SequenceMatcher(
                None, 
                product_name.lower(), 
                real_name.lower()
            ).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = real_name
        
        # Jika tidak ditemukan, coba cari dengan fuzzy matching
        if not best_match:
            best_match = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.7
            )
        
        # Jika masih tidak ketemu, gunakan GPT untuk menebak
        if not best_match:
            guessed_name = self._guess_product_name(f"Produk: {product_name}")
            if guessed_name:
                best_match = guessed_name
        
        if not best_match:
            # Cari kandidat produk yang mirik
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.7, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates]
                )
                
                return f"""
                <div class="candidate-container">
                    <p>Produk "{product_name}" tidak ditemukan. Mungkin maksud Anda:</p>
                    <div class="candidate-grid" style="display: flex; flex-wrap: wrap; gap: 15px;">
                        {candidate_cards}
                    </div>
                    <p>Silakan pilih salah satu atau sebutkan nama produk dengan lebih spesifik.</p>
                </div>
                """
            return f"❌ Produk '{product_name}' tidak ditemukan."
        
        # Tampilkan card produk
        product_card = self._generate_product_cards([{'name': best_match}])
        
        # Ambil info singkat jika ada
        product_data = self.processor.get_product_data(best_match)
        brief_info = ""
        if product_data:
            if product_data.get('jenis_kulit'):
                skin_types = ", ".join(product_data['jenis_kulit'])
                brief_info += f"<p><b>Jenis Kulit:</b> {skin_types}</p>"
            if product_data.get('type_produk'):
                brief_info += f"<p><b>Jenis Produk:</b> {product_data['type_produk']}</p>"
        
        return f"""
        <div class="product-card-response">
            <div class="context-info">
                <p>Produk yang dimaksud: {best_match}</p>
            </div>
            <div class="product-card-container">
                {product_card}
            </div>
            <div class="product-brief-info">
                {brief_info}
            </div>
            <p>Gunakan tombol 'Lihat Detail' untuk informasi lebih lanjut.</p>
        </div>
        """

    def handle_confirmation(self, question: str) -> str:
        """Menangani konfirmasi produk setelah typo"""
        if not self.context.product_candidates:
            return "❌ Tidak ada kandidat produk yang tersedia. Silakan sebutkan produk yang dimaksud."
        
        question_lower = question.lower()
        
        # Cek jika user mengatakan "ya" untuk kandidat pertama
        if any(kw in question_lower for kw in ['ya', 'yes', 'benar', 'betul', 'itu', 'yang pertama']):
            selected_product = self.context.product_candidates[0]
            self.context.product_candidates = []  # Reset kandidat
            return self.show_product_card(selected_product)
        
        # Cek jika user mengatakan "tidak" atau "lain"
        if any(kw in question_lower for kw in ['tidak', 'no', 'bukan', 'lain', 'yang lain', 'yang lainnya']):
            if len(self.context.product_candidates) > 1:
                # Tampilkan kandidat berikutnya
                next_candidates = self.context.product_candidates[1:]
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in next_candidates]
                )
                
                self.context.product_candidates = next_candidates  # Update kandidat
                
                return f"""
                <div class="candidate-container">
                    <p>Produk lainnya yang mungkin Anda maksud:</p>
                    <div class="candidate-grid" style="display: flex; flex-wrap: wrap; gap: 15px;">
                        {candidate_cards}
                    </div>
                    <p>Silakan pilih atau sebutkan nama produk dengan lebih spesifik.</p>
                </div>
                """
            else:
                self.context.product_candidates = []
                return "❌ Tidak ada alternatif lain. Silakan sebutkan nama produk dengan lebih spesifik."
        
        # Jika user menyebutkan nama produk secara spesifik
        product = self._extract_product_from_question(question)
        if product:
            self.context.product_candidates = []
            return self.show_product_card(product)
        
        return "❌ Silakan pilih salah satu kandidat produk atau sebutkan nama produk dengan lebih spesifik."

    def _clean_content(self, content: str, exclude_keywords: List[str]) -> str:
        """Remove unrelated sections with improved logic"""
        if not content:
            return "❌ Informasi tidak tersedia"
        
        # Hapus angka dan bullet di awal
        content = re.sub(r'^[\d\s•\-\.]+', '', content, flags=re.MULTILINE)
        
        # Split pada setiap keyword yang dikecualikan yang muncul di awal baris baru
        for keyword in exclude_keywords:
            pattern = r'\n\s*' + re.escape(keyword) + r'[\s:]'
            content = re.split(pattern, content, flags=re.IGNORECASE)[0]
        
        return content.strip()

    def get_evaluation_summary(self) -> str:
        """Get evaluation summary as HTML"""
        if not self.evaluation_results:
            return "<p>Belum ada evaluasi</p>"
        
        # Group by evaluation type
        product_info_evals = [e for e in self.evaluation_results if e['type'] == 'produk_info']
        recommendation_evals = [e for e in self.evaluation_results if e['type'] == 'rekomendasi']
        
        summary = "<h3>Hasil Evaluasi</h3>"
        
        # Product info evaluations
        if product_info_evals:
            summary += "<h4>Informasi Produk</h4>"
            summary += "<table border='1'><tr><th>Produk</th><th>Jenis Info</th><th>BERTScore</th><th>Respons Sistem</th><th>Referensi</th></tr>"
            for result in product_info_evals:
                summary += f"<tr><td>{result['product']}</td><td>{result['info_type']}</td><td>{result['bert_score']:.4f}</td><td>{result['response']}</td><td>{result['reference']}</td></tr>"
            summary += "</table>"
        
        # Recommendation evaluations
        if recommendation_evals:
            summary += "<h4>Rekomendasi</h4>"
            summary += "<table border='1'><tr><th>Jenis Kulit</th><th>Presisi</th><th>Recall</th><th>BERTScore</th><th>Rekomendasi Sistem</th><th>Referensi</th></tr>"
            for result in recommendation_evals:
                summary += f"<tr><td>{result['skin_type']}</td><td>{result['precision']:.2f}</td><td>{result['recall']:.2f}</td><td>{result['bert_score']:.4f}</td><td>{', '.join(result['recommended'])}</td><td>{', '.join(result['reference'])}</td></tr>"
            summary += "</table>"
        
        return summary

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                skin_types = ", ".join(self.context.skin_type)
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {skin_types}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "need_skin_type":
            # Gunakan konteks sebelumnya jika ada
            if self.context.skin_type and self.context.context_maintained:
                logger.info(f"Using maintained skin type: {self.context.skin_type}")
                return self.get_recommendations("")
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(SUPPORTED_PRODUCT_TYPES)})"
        
        # Handle product info requests
        if q_type == "product_info":
            # Prioritaskan produk dari pertanyaan, lalu last_product, lalu konteks
            if not product:
                if self.context.last_product:
                    product = self.context.last_product
                    logger.info(f"Using last product: {product} for follow-up")
                elif self.context.product_type:
                    return "Produk apa yang Anda maksud? Silakan sebutkan nama produknya."
            
            response = self.get_product_info(product, info_types)
            # Simpan sebagai last_product hanya jika berhasil
            if not response.startswith("❌"):
                self.context.last_product = product
                self.context.set_last_info_types(info_types)
            return response
        
        # Handle product card requests
        if q_type == "product_card":
            if not product:
                return "Produk apa yang Anda maksud? Silakan sebutkan nama produknya."
            
            response = self.show_product_card(product)
            if not response.startswith("❌"):
                self.context.last_product = product
            return response
        
        # Handle recommendations
        if q_type == "rekomendasi" or q_type == "rekomendasi_lain":
            # Gunakan konteks kulit jika ada
            if self.context.skin_type:
                logger.info(f"Using existing skin type: {self.context.skin_type}")
                return self.get_recommendations("")
                
            # Jika tidak ada konteks kulit, minta konfirmasi
            if self.context.product_type and not self.context.skin_type:
                return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
            
            return self.get_recommendations("")
        
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        try:
            start_time = time.time()
            self.context.last_question = question
            
            # Handle konfirmasi produk setelah typo
            if self.context.product_candidates:
                return self.handle_confirmation(question)
            
            # Skip empty or special character only questions
            if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
                return "Silakan ajukan pertanyaan tentang produk skincare."

            # Update context - jangan reset last_product jika tidak ada produk baru
            self.context.update(question, "")
            
            # Analyze question
            q_type, product, info_types = self.analyze_question(question)
            logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
            logger.info(f"Current context: skin_type={self.context.skin_type}, product_type={self.context.product_type}, ingredient_filter={self.context.ingredient_filter}, price_filter={self.context.price_filter}")
            
            # Deteksi pertanyaan follow-up yang lebih natural
            is_follow_up = (
                not product and 
                self.context.last_product and 
                any(keyword in question.lower() for keyword in [
                    'lagi', 'tentang itu', 'produk tadi', 'produknya',
                    'tsb', 'tersebut', 'detail', 'info', 'jelaskan lebih', 'tentang produk', 'cara pakai'
                ])
            )
            
            if is_follow_up:
                # Gunakan info types dari konteks jika tidak ada yang baru
                if not info_types and self.context.get_last_info_types():
                    info_types = self.context.get_last_info_types()
                
                # Jika ada kata kunci spesifik, gunakan itu
                if 'manfaat' in question.lower():
                    info_types = ['manfaat']
                elif 'kandungan' in question.lower():
                    info_types = ['kandungan']
                elif 'cara pakai' in question.lower() or 'penggunaan' in question.lower():
                    info_types = ['cara_pakai']
                elif 'harga' in question.lower():
                    info_types = ['harga']
                elif 'lokasi' in question.lower() or 'beli' in question.lower():
                    info_types = ['lokasi']
                
                # Jika tidak ada info_types, gunakan manfaat sebagai default
                if not info_types:
                    info_types = ['manfaat']
                
                q_type = "product_info"
                product = self.context.last_product
                logger.info(f"Follow-up question detected: {info_types} for {product}")
            
            # Save for follow-up
            self.context.last_question_type = q_type
            
            # Generate response
            response = self._handle_response(q_type, product, info_types)
            logger.info(f"Response generated in {time.time()-start_time:.2f}s")
            return response
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return "❌ Terjadi kesalahan. Silakan coba lagi."

# ======================
# MAIN APPLICATION
# ======================
def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = input("Enter OPENAI_API_KEY: ")
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return

    pdf_path = input("Masukkan path file PDF: ").strip()
    if not os.path.exists(pdf_path):
        print("File tidak ditemukan")
        return

    print("Memproses dokumen...")
    processor = PDFProcessor()
    
    if not processor.initialize_vector_store(pdf_path):
        print("Gagal memproses dokumen")
        return

    product_names = processor.get_product_names()
    print(f"Ditemukan {len(product_names)} produk dalam dokumen")
    
    qa_engine = QAEngine(processor.vector_store, processor)
    print("Sistem siap. Ketik 'exit' untuk keluar, 'evaluasi' untuk melihat hasil evaluasi.")
    
    while True:
        try:
            question = input("\nPertanyaan: ").strip()
            if question.lower() == 'exit':
                break
            if question.lower() == 'evaluasi':
                print("\nHasil Evaluasi:")
                print(qa_engine.get_evaluation_summary())
                continue
            if not question:
                continue

            start_time = time.time()
            answer = qa_engine.generate_response(question)
            print(f"\nJawaban: {answer}")
            print(f"Waktu: {time.time()-start_time:.2f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()