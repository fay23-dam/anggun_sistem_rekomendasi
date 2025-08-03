import os
import fitz
import re
import time
import logging
import json
import difflib
import unicodedata
from urllib.parse import quote
from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

# ======================
# UTILITY FUNCTIONS
# ======================
def normalize_whitespace(text: str) -> str:
    """Normalisasi spasi dan karakter khusus"""
    text = re.sub(r'\s+', ' ', text)  # Ganti multiple spaces
    text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter khusus
    return text.strip().lower()

def normalize_filename(name: str) -> str:
    """Normalisasi nama file untuk URL yang aman"""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name).strip().lower()
    name = re.sub(r'[-\s]+', '_', name)
    return name

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
    
    # Tambahkan logger khusus untuk matching
    matching_logger = logging.getLogger('matching')
    matching_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("product_matching.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    matching_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

logger = configure_logging()
load_dotenv()

# ======================
# CONSTANTS
# ======================
SUPPORTED_PRODUCT_TYPES = ["sunscreen", "tabir surya", "moisturizer", "pelembap", 
                           "serum", "facial wash", "cleanser", "pembersih"]
SKIN_TYPES = ['kering', 'berminyak', 'normal', 'sensitif', 'berjerawat', 'kombinasi', 'kemerahan', 'kusam']
GREETINGS = ["halo", "hi", "hello", "selamat", "pagi", "siang", "sore", "malam"]
INFO_TYPES = ["manfaat", "kandungan", "cara pakai", "harga", "lokasi"]
RECOMMENDATION_PHRASES = ["rekomendasi lainnya", "rekomendasi lain", "lagi", "yang lain", "lainnya"]
INGREDIENT_PHRASES = ["mengandung", "dengan kandungan", "dengan bahan", "ingredients", "kandungan"]
OUT_OF_SCOPE_KEYWORDS = ["rambut", "badan", "tubuh", "tangan", "kaki", "bibir", "mata", "shampoo", "conditioner"]
BRANDS = ["garnier", "wardah", "skintific", "glad2glow", "azarine", "scarlett", "emina", "npure", "the originote"]
EXISTENCE_PHRASES = ["apakah ada", "ada produk", "ada tidak", "tersedia tidak"]

# ======================
# CONVERSATION CONTEXT
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []
        self.product_type = None
        self.last_product = None
        self.conversation_history = []
        self.recommended_products = set()
        self.all_products = set()
        self.current_intent = None
        self.active_recommendation = False
        self.filters = {
            "max_price": None,
            "min_price": None,
            "ingredients": [],
            "brands": []  # Filter berdasarkan brand
        }
        
        # Untuk menangani multiple product selection
        self.pending_products = []  # Daftar produk yang sedang ditawarkan
        self.pending_info_types = []  # Jenis info yang diminta
        
        # Lacak intent terakhir
        self.last_intent = None  # 'statistical', 'recommendation', 'info', dll.
        
        if product_names:
            for name in product_names:
                self.add_product(name)
                
    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        
    def get_recent_history(self, max_messages=3) -> str:
        history_text = ""
        for msg in self.conversation_history[-max_messages:]:
            history_text += f"{msg['role']}: {msg['content']}\n"
        return history_text.strip()
    
    def add_product(self, product_name: str):
        self.all_products.add(product_name.strip())
    
    def get_context_summary(self) -> str:
        summary = [f"Intent Terakhir: {self.last_intent}"]
        if self.skin_type: summary.append(f"Kulit: {', '.join(self.skin_type)}")
        if self.product_type: summary.append(f"Produk: {self.product_type}")
        if self.last_product: summary.append(f"Produk Terakhir: {self.last_product}")
        if self.filters["max_price"] is not None: summary.append(f"Harga Maks: Rp{self.filters['max_price']:,}")
        if self.filters["min_price"] is not None: summary.append(f"Harga Min: Rp{self.filters['min_price']:,}")
        if self.filters["ingredients"]: summary.append(f"Kandungan: {', '.join(self.filters['ingredients'])}")
        if self.filters["brands"]: summary.append(f"Brand: {', '.join(self.filters['brands'])}")
        return ", ".join(summary)
    
    def reset_recommendations(self):
        self.recommended_products = set()
        logger.info("Riwayat rekomendasi direset")
        
    def update_skin_type(self, new_types: List[str]):
        """Update jenis kulit dengan reset filter jika jenis kulit berubah"""
        if set(new_types) != set(self.skin_type):
            logger.info(f"Perubahan jenis kulit: {self.skin_type} => {new_types}")
            self.skin_type = new_types
            self.filters["ingredients"] = []
            self.filters["max_price"] = None
            self.filters["min_price"] = None
            self.last_product = None
            self.filters["brands"] = []
            self.reset_recommendations()
        else:
            for skin in new_types:
                if skin not in self.skin_type:
                    self.skin_type.append(skin)
                
    def reset_skin_type(self):
        self.skin_type = []
        logger.info("Jenis kulit direset")
                
    def update_filters(self, max_price: Optional[float] = None, 
                       min_price: Optional[float] = None,
                       ingredients: Optional[List[str]] = None,
                       brands: Optional[List[str]] = None):
        logger.info(f"Memperbarui filter: max_price={max_price}, min_price={min_price}, ingredients={ingredients}, brands={brands}")
        logger.info(f"Filter sebelum: {self.filters}")
        
        if max_price is not None:
            self.filters["max_price"] = max_price
            # Reset min_price hanya jika kita menambahkan filter max baru
            if min_price is None:
                self.filters["min_price"] = None
            self.reset_recommendations()
            
        if min_price is not None:
            self.filters["min_price"] = min_price
            # Reset max_price hanya jika kita menambahkan filter min baru
            if max_price is None:
                self.filters["max_price"] = None
            self.reset_recommendations()
            
        if ingredients is not None:
            self.filters["ingredients"] = ingredients
            self.reset_recommendations()
            
        if brands is not None:
            self.filters["brands"] = [b.lower() for b in brands]
            self.reset_recommendations()
            
        logger.info(f"Filter setelah: {self.filters}")
            
    def reset_filters(self):
        self.filters = {
            "max_price": None,
            "min_price": None,
            "ingredients": [],
            "brands": []
        }
        logger.info("Filter direset: harga, kandungan, dan brand")
    
    def reset_context(self):
        self.reset_skin_type()
        self.reset_filters()
        self.reset_recommendations()
        self.product_type = None
        self.last_product = None
        self.active_recommendation = False
        self.last_intent = None
        logger.info("Semua konteks direset")
        
    def update_context(self, analysis: dict):
        """Update konteks berdasarkan hasil analisis pertanyaan"""
        new_intent = analysis.get("question_type")
    
        # Reset konteks jika intent berubah ke statistical/info_produk
        if new_intent in ["statistical", "info_produk"] and self.last_intent != new_intent:
            self.reset_filters()
            self.active_recommendation = False
            self.last_product = None
            
        # Reset jenis kulit jika ada perubahan jenis produk atau brand
        if (analysis.get("product_type") and analysis["product_type"] != self.product_type) or (analysis.get("brands") and set(analysis["brands"]) != set(self.filters["brands"])):
            self.skin_type = []  # Reset jenis kulit
            logger.info("Reset jenis kulit karena perubahan produk/brand")
            
        # Reset filter jika ada perubahan jenis produk
        if analysis.get("product_type") and analysis["product_type"] != self.product_type:
            logger.info(f"Perubahan jenis produk: {self.product_type} => {analysis['product_type']}")
            self.reset_filters()
            self.product_type = analysis["product_type"]
            
        # Update jenis kulit jika ada
        if analysis.get("skin_types"):
            self.update_skin_type(analysis["skin_types"])
            
        # Update filter jika ada
        if analysis.get("max_price") is not None or analysis.get("min_price") is not None or analysis.get("ingredients") or analysis.get("brands"):
            self.update_filters(
                max_price=analysis.get("max_price"),
                min_price=analysis.get("min_price"),
                ingredients=analysis.get("ingredients"),
                brands=analysis.get("brands")
            )

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
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        self.full_text = self.extract_text(pdf_path)
        if not self.full_text:
            logger.error("No text extracted from PDF")
            return False
        
        self._extract_all_product_data(self.full_text)
        
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
        product_sections = re.split(r'(?=\nProduk:\s*)', text)
        
        for section in product_sections:
            if not section.strip():
                continue
                
            name_match = re.search(
                r'Produk:\s*((?:.|\n)*?)\s*(?=harga:|type produk:|jenis kulit:|informasi produk:|kandungan:|cara pemakaian:|manfaat:|lokasi :|\nProduk:|$)',
                section, 
                re.IGNORECASE
            )
            
            if name_match:
                name = name_match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                
                if len(name) < 3 or not any(c.isalpha() for c in name):
                    continue
                    
                jenis_kulit_text = self._extract_field(section, 'jenis kulit:').lower()
                skin_types = []
                for skin in SKIN_TYPES:
                    if skin in jenis_kulit_text:
                        skin_types.append(skin)
                
                type_produk = self._extract_field(section, 'type produk:').lower()
                normalized_type = 'lain'
                for pt in SUPPORTED_PRODUCT_TYPES:
                    if pt in type_produk:
                        normalized_type = pt
                        break
                
                harga_text = self._extract_field(section, 'harga:').lower()
                harga = self._parse_price(harga_text)
                
                # Ekstrak brand dari nama produk
                brand = "unknown"
                # 1. Coba deteksi dari nama produk
                for b in BRANDS:
                    if b in name.lower():
                        brand = b
                        break
                
                # 2. Jika tidak ditemukan, cari di seluruh section
                if brand == "unknown":
                    section_lower = section.lower()
                    for b in BRANDS:
                        if b in section_lower:
                            brand = b
                            break
                
                # 3. Khusus Wardah - deteksi pola khusus
                if brand == "unknown" and ("nature daily" in name.lower() or 
                                        "c-defense" in name.lower() or
                                        "perfect bright" in name.lower() or
                                        "heartleaf" in name.lower() or
                                        "vitamin c" in name.lower()):
                    brand = "wardah"
                product_info = {
                    'name': name,
                    'brand': brand,
                    'harga': harga,
                    'harga_text': harga_text,
                    'type_produk': normalized_type,
                    'jenis_kulit': skin_types,
                    'informasi_produk': self._extract_field(section, 'informasi produk:'),
                    'kandungan': self._extract_field(section, 'kandungan:'),
                    'cara_pemakaian': self._extract_field(section, 'cara pemakaian:'),
                    'manfaat': self._extract_field(section, 'manfaat:'),
                    'lokasi': self._extract_field(section, 'lokasi :')
                }
                
                self.product_names.add(name)
                self.product_data[name.lower()] = product_info
                logger.info(f"Extracted product: {name} | Brand: {brand} | Skin: {product_info['jenis_kulit']} | Type: {normalized_type} | Price: {harga}")
        
        logger.info(f"Extracted {len(self.product_names)} product names from PDF")

    def _parse_price(self, price_str: str) -> float:
        try:
            # Handle various price formats: 100.000, 100,000, Rp 100.000, 100rb, 100k
            price_str = price_str.replace('Rp', '').replace('.', '').replace(',', '').strip()
            
            # Handle ribu/k format
            if 'rb' in price_str or 'k' in price_str:
                price_str = re.sub(r'[^\d.]', '', price_str)
                return float(price_str) * 1000 if price_str else 0
            
            # Handle juta/jt format
            if 'jt' in price_str or 'juta' in price_str:
                price_str = re.sub(r'[^\d.]', '', price_str)
                return float(price_str) * 1000000 if price_str else 0
                
            # Normal numeric format
            return float(re.sub(r'[^\d]', '', price_str)) if re.sub(r'[^\d]', '', price_str) else 0
        except:
            return 0

    def _extract_field(self, text: str, field_name: str) -> str:
        try:
            pattern = re.compile(re.escape(field_name), re.IGNORECASE)
            match = pattern.search(text)
            if not match:
                return ""
            
            start_idx = match.end()
            remaining_text = text[start_idx:]
            
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
            content = re.sub(r'^[\d\s•\-\.]+', '', content)
            content = normalize_whitespace(content)  # Normalisasi
            return content
        except Exception as e:
            logger.error(f"Error extracting field {field_name}: {e}")
            return ""

    def get_product_names(self) -> List[str]:
        return list(self.product_names)
    
    def get_product_data(self, product_name: str) -> Optional[Dict]:
        return self.product_data.get(product_name.lower())
    
    def get_product_summary(self) -> str:
        """Generate summary statistics of product data"""
        if not self.product_data:
            return "Tidak ada data produk"
            
        summary = []
        
        # Total produk
        summary.append(f"Total produk: {len(self.product_data)}")
        
        # Distribusi jenis produk
        type_counts = {}
        for product in self.product_data.values():
            product_type = product.get('type_produk', 'lain')
            type_counts[product_type] = type_counts.get(product_type, 0) + 1
        
        type_summary = ", ".join([f"{k}: {v}" for k, v in type_counts.items()])
        summary.append(f"Distribusi jenis produk: {type_summary}")
        
        # Distribusi jenis kulit
        skin_counts = {}
        for product in self.product_data.values():
            for skin_type in product.get('jenis_kulit', []):
                skin_counts[skin_type] = skin_counts.get(skin_type, 0) + 1
        
        skin_summary = ", ".join([f"{k}: {v}" for k, v in skin_counts.items()])
        summary.append(f"Distribusi jenis kulit: {skin_summary}")
        
        # Statistik khusus untuk produk populer
        sunscreen_count = type_counts.get('sunscreen', 0) + type_counts.get('tabir surya', 0)
        serum_count = type_counts.get('serum', 0)
        moisturizer_count = type_counts.get('moisturizer', 0) + type_counts.get('pelembap', 0)
        cleanser_count = type_counts.get('facial wash', 0) + type_counts.get('cleanser', 0) + type_counts.get('pembersih', 0)
        acne_count = skin_counts.get('berjerawat', 0)
        
        summary.append(f"Jumlah produk sunscreen: {sunscreen_count}")
        summary.append(f"Jumlah produk serum: {serum_count}")
        summary.append(f"Jumlah produk moisturizer: {moisturizer_count}")
        summary.append(f"Jumlah produk cleanser: {cleanser_count}")
        summary.append(f"Jumlah produk untuk jerawat: {acne_count}")
        
        # Distribusi brand
        brand_counts = {}
        for product in self.product_data.values():
            brand = product.get('brand', 'unknown')
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        brand_summary = ", ".join([f"{k}: {v}" for k, v in brand_counts.items()])
        summary.append(f"Distribusi brand: {brand_summary}")
        
        return "\n".join(summary)

    def _normalize_name(self, name: str) -> str:
        """Normalisasi nama produk untuk pencocokan yang lebih baik"""
        name = name.lower()
        # Hapus brand jika ada di tengah nama
        name = re.sub(r'\b(garnier|wardah|erha|etc)\b', '', name)
        # Hapus karakter khusus
        name = re.sub(r'[^\w\s]', '', name)
        # Hapus kata umum
        name = re.sub(r'\b(face|facial|wash|moisturizer|serum|cream|gel|glow)\b', '', name)
        # Hapus spasi berlebih
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def find_best_product_match(self, product_name: str) -> Optional[Tuple[str, float]]:
        normalized_query = self._normalize_name(product_name)
        best_match = None
        best_score = 0
        
        for real_name in self.product_names:
            # Normalisasi nama produk di database
            normalized_real = self._normalize_name(real_name)
            
            score = difflib.SequenceMatcher(
                None, 
                normalized_query,
                normalized_real
            ).ratio()
            
            # Bonus jika query merupakan substring
            if normalized_query in normalized_real:
                score = min(score + 0.2, 1.0)
                
            if score > best_score:
                best_score = score
                best_match = real_name
        
        # Log pencocokan
        matching_logger = logging.getLogger('matching')
        matching_logger.info(f"Matching: '{product_name}' -> '{best_match}' (score: {best_score:.2f})")
        
        if best_match and best_score >= 0.4:
            return best_match, best_score
        return None, best_score

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
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )
        self.context = ConversationContext(processor.get_product_names())
        self.last_recommendation_criteria = {}
        
    def _generate_product_cards(self, products: List[Dict]) -> str:
        cards = []
        for product in products:
            # Gunakan nama produk asli (dari data)
            name = product['name']
            # Normalisasi nama file untuk URL yang aman
            image_name = normalize_filename(name)
            image_url = quote(name)
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{image_url}.jpg" alt="{name}" class="product-image">
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

    def _analyze_question_with_llm(self, question: str) -> dict:
        # Deteksi sapaan sederhana
        if any(greeting in question.lower() for greeting in GREETINGS) and len(question.split()) < 4:
            return {
                "question_type": "sapaan",
                "skin_types": [],
                "product_type": None,
                "product_name": "",
                "info_types": [],
                "is_face_skincare": True,
                "max_price": None,
                "min_price": None,
                "ingredients": [],
                "brands": []
            }
            
        # Deteksi rekomendasi lainnya
        if any(phrase in question.lower() for phrase in RECOMMENDATION_PHRASES):
            return {
                "question_type": "rekomendasi_lagi",
                "skin_types": [],
                "product_type": None,
                "product_name": None,
                "info_types": [],
                "is_face_skincare": True,
                "max_price": None,
                "min_price": None,
                "ingredients": [],
                "brands": []
            }
            
        # Ekstrak harga dengan lebih akurat
        max_price = None
        min_price = None
        ingredients = []
        brands = []
        
        # Ekstrak harga maksimum (dibawah X)
        price_match = re.search(r'(?:dibawah|kurang dari|max|maks|harga)\s*[rp]?\s*([\d.,]+)\s*(?:rb|ribu|k)?', question, re.IGNORECASE)
        if price_match:
            price_value = price_match.group(1).replace('.', '').replace(',', '')
            try:
                max_price = float(price_value) * 1000
            except:
                logger.warning(f"Failed to parse max price: {price_match.group(1)}")
            
        # Ekstrak harga minimum (diatas X)
        price_match = re.search(r'(?:diatas|minimal|mulai|lebih dari)\s*[rp]?\s*([\d.,]+)\s*(?:rb|ribu|k)?', question, re.IGNORECASE)
        if price_match:
            price_value = price_match.group(1).replace('.', '').replace(',', '')
            try:
                min_price = float(price_value) * 1000
            except:
                logger.warning(f"Failed to parse min price: {price_match.group(1)}")
            
        # Ekstrak kandungan
        if any(phrase in question.lower() for phrase in INGREDIENT_PHRASES):
            content_match = re.search(r'mengandung\s+([^\?\.]+)', question, re.IGNORECASE)
            if content_match:
                ingredients = [word.strip().lower() for word in content_match.group(1).split() if len(word) > 3]
        
        # Ekstrak brand
        for brand in BRANDS:
            if brand in question.lower():
                brands.append(brand)
        
        # Deteksi pertanyaan keberadaan produk
        if any(phrase in question.lower() for phrase in EXISTENCE_PHRASES):
            # Ekstrak nama produk dari pertanyaan
            product_name = question.lower()
            if not product_name:
                product_name = self.context.last_product
            for phrase in EXISTENCE_PHRASES:
                if phrase in product_name:
                    product_name = product_name.split(phrase)[-1].strip()
                    break
            return {
                "question_type": "info_produk",
                "skin_types": [],
                "product_type": None,
                "product_name": product_name,
                "info_types": ["existence"],
                "is_face_skincare": True,
                "max_price": max_price,
                "min_price": min_price,
                "ingredients": ingredients,
                "brands": brands
            }
        
        # Deteksi perubahan konteks signifikan
        context_change = False
        if self.context.last_intent == "rekomendasi":
            # Deteksi perubahan merek
            current_brands = set(self.context.filters.get("brands", []))
            new_brands = set()
            for brand in BRANDS:
                if brand in question.lower():
                    new_brands.add(brand)
            
            # Jika ada perubahan merek signifikan
            if new_brands and current_brands and not new_brands.issubset(current_brands):
                context_change = True
                logger.info(f"Deteksi perubahan merek: {current_brands} => {new_brands}")
        
        # Prompt untuk analisis LLM
        prompt = f"""
[INSTRUKSI]
Analisis pertanyaan pengguna: "{question}"

LANGKAH:
1. Identifikasi jenis pertanyaan:
   - sapaan: salam, perkenalan
   - rekomendasi: minta saran produk skincare wajah, baik dari jenis kulit, produk, ataupun brand
   - rekomendasi_lagi: permintaan rekomendasi tambahan
   - info_produk: tanya detail produk spesifik, berdasarkan nama produk
   - filter_update: update kriteria filter (harga, kandungan, brand)
   - statistical: pertanyaan tentang jumlah, statistik produk
   - lainnya: pertanyaan umum skincare
   - out_of_scope: di luar skincare wajah

2. Identifikasi jenis kulit WAJAH: {SKIN_TYPES}
3. Identifikasi tipe produk WAJAH: {SUPPORTED_PRODUCT_TYPES}
4. Identifikasi brand: {BRANDS}
CATATAN TAMBAHAN:
- Jika hanya ada 1 kata dan termasuk dalam tipe produk wajah ({SUPPORTED_PRODUCT_TYPES}), anggap sebagai permintaan statistik jika sebelumnya ada konteks statistik, atau rekomendasi jika tidak.
- Jika pertanyaan singkat (<=3 kata) dan mengandung tipe produk, dan sebelumnya ada konteks statistik, set sebagai statistical.
PERHATIAN KHUSUS:
- Jika ada kata kunci: rambut, badan, tubuh, tangan, kaki, bibir, mata, shampoo, conditioner atau hal lainnya yang tidak untuk skincare kulit wajah
  -> set is_face_skincare=false dan question_type=out_of_scope
- Kata 'kering' bisa berarti kulit atau rambut, bedakan berdasarkan konteks
PERHATIAN:
- Jika user pertanyaan user hanya mengandung {BRANDS} dan jenis {SUPPORTED_PRODUCT_TYPES}, anggap sebagai rekomendasi produk dari brand tersebut (misal: "serum scarlett", "serum glad2glow").
- Jika user menyebutkan {INFO_TYPES} seperti "manfaat", "kandungan", "cara pakai", "harga", "lokasi", anggap sebagai info_produk.
- Jika user menyebutkan "brand [nama]" atau "dari [brand]", anggap sebagai filter brand untuk rekomendasi.
- Jika user menyebutkan nama produk lengkap (misal: "SKINTIFIC NIACINAMIDE SERUM"), anggap sebagai info_produk.
- Jika user menyebutkan jenis produk + brand (misal: "serum dari skintific"), anggap sebagai rekomendasi.
Jika pertanyaan singkat seperti "berapa harganya", "manfaatnya apa", atau "cara pakainya", dan konteks sebelumnya adalah info_produk, maka pertahankan product_name dari konteks.
5. Identifikasi nama produk spesifik (jika ada)
6. Identifikasi jenis informasi yang diminta: {INFO_TYPES}
7. Untuk sapaan dan identitas, selalu set is_face_skincare=true

FORMAT RESPONS: JSON
{{
    "question_type": "sapaan|rekomendasi|rekomendasi_lagi|info_produk|filter_update|statistical|out_of_scope|lainnya|reset",
    "skin_types": ["list", "kulit", "wajah"],
    "product_type": "tipe produk",
    "product_name": "nama produk",
    "info_types": ["list", "jenis", "info"],
    "is_face_skincare": true/false,
    "max_price": null|number,
    "min_price": null|number,
    "ingredients": ["list", "ingredients"],
    "brands": ["list", "brands"]
}}

CONTOH:
Pertanyaan: "rekomendasi serum untuk kulit kering dibawah 100 ribu"
Output: {{"question_type": "rekomendasi", "skin_types": ["kering"], "product_type": "serum", "product_name": null, "info_types": [], "is_face_skincare": true, "max_price": 100000, "min_price": null, "ingredients": [], "brands": []}}

Pertanyaan: "serum scarlett"
Output: {{"question_type": "rekomendasi", "skin_types": [], "product_type": "serum", "product_name": null, "info_types": [], "is_face_skincare": true, "max_price": null, "min_price": null, "ingredients": [], "brands": ["scarlett"]}}

Pertanyaan: "serum glad2glow"
Output: {{"question_type": "rekomendasi", "skin_types": [], "product_type": "serum", "product_name": null, "info_types": [], "is_face_skincare": true, "max_price": null, "min_price": null, "ingredients": [], "brands": ["glad2glow"]}}

Pertanyaan: "dibawah 100 ribu"
Output: {{"question_type": "filter_update", "skin_types": [], "product_type": null, "product_name": null, "info_types": [], "is_face_skincare": true, "max_price": 100000, "min_price": null, "ingredients": [], "brands": []}}

Pertanyaan: "rekomendasi produk wardah"
Output: {{"question_type": "rekomendasi", "skin_types": [], "product_type": null, "product_name": null, "info_types": [], "is_face_skincare": true, "max_price": null, "min_price": null, "ingredients": [], "brands": ["wardah"]}}
"""
        
        try:
            response = self.llm.invoke(prompt).content
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            analysis = json.loads(json_str)
            
            # Validasi jenis kulit
            valid_skin_types = []
            for skin in analysis.get("skin_types", []):
                if skin in SKIN_TYPES:
                    valid_skin_types.append(skin)
            analysis["skin_types"] = valid_skin_types
            
            # Tambahkan hasil ekstraksi manual
            if max_price is not None:
                analysis["max_price"] = max_price
            if min_price is not None:
                analysis["min_price"] = min_price
            if ingredients:
                analysis["ingredients"] = ingredients
            if brands:
                analysis["brands"] = brands
            
            # Jika hanya ada filter harga/brand, anggap sebagai update filter
            if (analysis.get("max_price") or analysis.get("min_price") or analysis.get("brands")) and not analysis.get("skin_types") and not analysis.get("product_type") and not analysis.get("product_name"):
                analysis["question_type"] = "filter_update"
            
            # Jika deteksi sebagai perubahan konteks signifikan
            if context_change:
                analysis["question_type"] = "rekomendasi"
                analysis["brands"] = list(new_brands)
                logger.info("Override question type to rekomendasi due to context change")
            
            return analysis
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {
                "question_type": "unknown",
                "skin_types": [],
                "product_type": None,
                "product_name": None,
                "info_types": [],
                "is_face_skincare": True,
                "max_price": None,
                "min_price": None,
                "ingredients": [],
                "brands": []
            }

    def _get_rag_response(self, question: str, context: str, documents: List[str]) -> str:
        # Dapatkan ringkasan produk
        product_summary = self.processor.get_product_summary()
        
        docs_text = "\n\n".join(documents[:3])[:3000]
        
        prompt = f"""
[KONTEKS] 
{context}

[INFORMASI PRODUK]
{docs_text}

[STATISTIK PRODUK]
{product_summary}

[PERINTAH]
Jawab hanya tentang skincare wajah dalam 1-3 kalimat.
Gunakan bahasa Indonesia informal.
Jika tidak tahu, katakan tidak tahu.
Gunakan statistik produk jika relevan dengan pertanyaan.

Pertanyaan: {question}
"""
        
        try:
            response = self.llm.invoke(prompt).content
            return f"""<div class="rag-response">{response}</div>"""
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return f"""<div class="rag-response">Maaf, saya sedang tidak bisa menjawab. Bisa diulang?</div>"""

    def _get_out_of_scope_response(self, question: str) -> str:
        """Respons interaktif untuk pertanyaan di luar scope skincare wajah"""
        prompt = f"""
Anda adalah asisten virtual yang ramah. Anda hanya membantu untuk produk skincare wajah.
Jika ada pertanyaan di luar skincare wajah, jelaskan dengan ramah bahwa Anda hanya fokus pada skincare wajah.
Jawab dengan 1-2 kalimat.

Pertanyaan: {question}
"""
        try:
            response = self.llm.invoke(prompt).content
            return f"""<div class="out-of-scope">{response}</div>"""
        except:
            return f"""<div class="out-of-scope">Maaf, saya hanya membantu untuk skincare wajah. Ada yang bisa saya bantu tentang perawatan wajah?</div>"""

    def get_recommendations(self, question: str) -> str:
        try:
            # Dapatkan semua produk yang tersedia
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data and product_data.get('type_produk') in SUPPORTED_PRODUCT_TYPES:
                    all_products.append({
                        'name': product_name,
                        'data': product_data
                    })
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                if product['name'] in self.context.recommended_products:
                    continue
                    
                skin_data = [s.lower() for s in product['data'].get('jenis_kulit', [])]
                type_data = product['data'].get('type_produk', '').lower()
                
                # Filter jenis kulit
                skin_match = True
                if self.context.skin_type:
                    skin_match = any(user_skin in skin_data for user_skin in self.context.skin_type)
                
                # Filter jenis produk
                product_match = True
                if self.context.product_type:
                    product_match = self.context.product_type.lower() == type_data
                
                # Filter harga - validasi ketat
                price_match = True
                product_price = product['data'].get('harga', 0)
                
                if self.context.filters["max_price"] is not None:
                    if product_price > self.context.filters["max_price"]:
                        price_match = False
                
                if self.context.filters["min_price"] is not None:
                    if product_price < self.context.filters["min_price"]:
                        price_match = False
                
                # Filter kandungan
                ingredient_match = True
                if self.context.filters["ingredients"]:
                    kandungan = product['data'].get('kandungan', '').lower()
                    found = False
                    for ingredient in self.context.filters["ingredients"]:
                        if ingredient.lower() in kandungan:
                            found = True
                            break
                    if not found:
                        ingredient_match = False
                
                # Filter brand
                brand_match = True
                if self.context.filters["brands"]:
                    product_brand = product['data'].get('brand', '').lower()
                    brand_match = any(
                        brand.lower() in product_brand 
                        for brand in self.context.filters["brands"]
                    )
                
                if skin_match and product_match and price_match and ingredient_match and brand_match:
                    filtered_products.append(product)
            
            # Simpan kriteria untuk rekomendasi berikutnya
            self.last_recommendation_criteria = {
                "skin_types": self.context.skin_type.copy(),
                "product_type": self.context.product_type,
                "filters": {
                    "max_price": self.context.filters["max_price"],
                    "min_price": self.context.filters["min_price"],
                    "ingredients": self.context.filters["ingredients"].copy(),
                    "brands": self.context.filters["brands"].copy()
                }
            }
            
            # Jika tidak ada produk baru, reset rekomendasi
            if not filtered_products:
                self.context.reset_recommendations()
                filtered_products = []
                for product in all_products:
                    skin_data = [s.lower() for s in product['data'].get('jenis_kulit', [])]
                    type_data = product['data'].get('type_produk', '').lower()
                    
                    skin_match = True
                    if self.context.skin_type:
                        skin_match = any(user_skin in skin_data for user_skin in self.context.skin_type)
                    
                    product_match = True
                    if self.context.product_type:
                        product_match = self.context.product_type.lower() == type_data
                    
                    price_match = True
                    product_price = product['data'].get('harga', 0)
                    
                    if self.context.filters["max_price"] is not None:
                        if product_price > self.context.filters["max_price"]:
                            price_match = False
                    
                    if self.context.filters["min_price"] is not None:
                        if product_price < self.context.filters["min_price"]:
                            price_match = False
                    
                    ingredient_match = True
                    if self.context.filters["ingredients"]:
                        kandungan = product['data'].get('kandungan', '').lower()
                        found = False
                        for ingredient in self.context.filters["ingredients"]:
                            if ingredient.lower() in kandungan:
                                found = True
                                break
                        if not found:
                            ingredient_match = False
                    
                    brand_match = True
                    if self.context.filters["brands"]:
                        product_brand = product['data'].get('brand', '').lower()
                        brand_match = any(
                            brand.lower() in product_brand 
                            for brand in self.context.filters["brands"]
                        )
                    
                    if skin_match and product_match and price_match and ingredient_match and brand_match:
                        filtered_products.append(product)
            
            # Jika masih tidak ada produk
            if not filtered_products:
                return """
                <div class="no-product-response">
                    <p>😔 Maaf, tidak ada produk yang sesuai dengan kriteria Anda.</p>
                    <p>Silakan coba dengan kriteria yang lain.</p>
                </div>
                """
            
            # Ambil 5 produk baru
            products_to_show = filtered_products[:5]
            
            # Log produk yang direkomendasikan
            logger.info("===== REKOMENDASI PRODUK =====")
            logger.info(f"Kriteria: Jenis Kulit: {self.context.skin_type}, Tipe Produk: {self.context.product_type}")
            logger.info(f"Filter: Max Price: {self.context.filters['max_price']}, Min Price: {self.context.filters['min_price']}")
            
            for p in products_to_show:
                self.context.recommended_products.add(p['name'])
                product_data = self.processor.get_product_data(p['name'])
                logger.info(f"Produk: {p['name']} | Brand: {product_data['brand']} | Jenis: {product_data['type_produk']} | Kulit: {product_data['jenis_kulit']} | Harga: {product_data['harga']}")
            logger.info("===============================")
            
            # Generate product cards
            product_cards = self._generate_product_cards([
                {'name': p['name']} for p in products_to_show
            ])
            
            # Generate natural explanation with LLM
            explanation = self._get_recommendations_explanation(products_to_show)
            
            # Tombol lebih banyak
            more_button = ""
            if len(filtered_products) > 5:
                more_button = """
                <div class="recommendation-feedback">
                    <button class="feedback-btn more" data-feedback="more">
                        <i class="fas fa-redo mr-6"></i> Rekomendasi Lainnya
                    </button>
                </div>
                """
            
            return f"""
            <div class="recommendation-container text-justify">
                <div class="rag-response">
                    {explanation}
                </div>
                <div class="product-grid" style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px;">
                    {product_cards}
                </div>
                {more_button}
            </div>
            """
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return f"""<div class="error-response">❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi.</div>"""

    def _get_recommendations_explanation(self, products: List[dict]) -> str:
        """Generate natural language explanation for recommendations"""
        products_info = []
        for product in products[:5]:
            name = product['name']
            product_data = self.processor.get_product_data(name)
            if product_data:
                products_info.append({
                    "name": name,
                    "brand": product_data.get('brand', ''),
                    "type": product_data.get('type_produk', ''),
                    "price": product_data.get('harga', 0),
                    "benefits": product_data.get('manfaat', '')[:100] + "..." if product_data.get('manfaat') else "Tidak ada informasi manfaat",
                    "skin_types": product_data.get('jenis_kulit', [])
                })
        
        if not products_info:
            return f"""<p>Maaf, saya tidak menemukan produk yang sesuai dengan kebutuhan Anda.</p>"""
        
        skin_types = ", ".join(self.context.skin_type) if self.context.skin_type else "berbagai jenis kulit"
        product_type = self.context.product_type if self.context.product_type else "produk skincare"
        
        # Filter info untuk prompt
        filter_info = []
        if self.context.filters["max_price"] is not None:
            filter_info.append(f"Batas Harga: Rp {self.context.filters['max_price']:,}")
        if self.context.filters["min_price"] is not None:
            filter_info.append(f"Harga Minimal: Rp {self.context.filters['min_price']:,}")
        if self.context.filters["ingredients"]:
            filter_info.append(f"Kandungan: {', '.join(self.context.filters['ingredients'])}")
        if self.context.filters["brands"]:
            filter_info.append(f"Brand: {', '.join(self.context.filters['brands'])}")
        
        filter_text = "\n".join(filter_info) if filter_info else "Tidak ada filter tambahan"
        
        prompt = f"""
[Kriteria Pengguna]
Jenis Kulit: {skin_types}
Tipe Produk: {product_type}
{filter_text}

[Produk yang Direkomendasikan]
{json.dumps(products_info, indent=2, ensure_ascii=False)}

[Instruksi]
Buat penjelasan rekomendasi dalam format HTML dengan:
1. Langsung berikan rekomendasi tanpa salam
2. Sebutkan 3-5 produk yang paling relevan
3. Untuk setiap produk:
- Gunakan struktur: <p><b>[Nama Produk]</b>: [Penjelasan singkat]</p>
4. Berikan alasan singkat (1 kalimat) mengapa produk ini cocok
5. Gunakan maksimal 3 kalimat untuk keseluruhan penjelasan
6. Judul rekomendasi dalam tag <h4>
7. Jangan tambahkan tag html, head, atau body

Contoh output:
<h4>Rekomendasi Serum untuk Kulit Kering</h4>
<p><b>SKINTIFIC NIACINAMIDE BRIGHTENING SERUM 20ML</b>: Produk ini efektif mencerahkan kulit dan menenangkan, sehingga cocok untuk kulit kering.</p>
<p><b>AZARINE MIRACLE OIL SERUM</b>: Serum ini memberikan hidrasi dan mendukung perawatan antiaging, ideal untuk menjaga kelembapan kulit kering.</p>
"""
        
        try:
            return self.llm.invoke(prompt).content
        except:
            # Fallback to simple HTML listing
            products_list = "".join([f"<p><b>{p['name']}</b></p>" for p in products[:3]])
            return f"<h4>Rekomendasi Produk</h4>{products_list}"

    def _find_product_candidates(self, product_name: str) -> List[dict]:
        """Cari produk tanpa menampilkan UI, hanya untuk pengecekan keberadaan"""
        normalized_query = product_name.lower()
        candidate_products = []
        
        # Prioritas 1: Exact match
        for real_name in self.processor.product_names:
            if normalized_query == real_name.lower():
                return [{'name': real_name}]
        
        # Prioritas 2: Substring match
        for real_name in self.processor.product_names:
            if normalized_query in real_name.lower():
                candidate_products.append({'name': real_name})
        
        # Prioritas 3: Similarity match
        if not candidate_products:
            for real_name in self.processor.product_names:
                score = difflib.SequenceMatcher(
                    None, 
                    product_name.lower(), 
                    real_name.lower()
                ).ratio()
                
                if score > 0.6:
                    candidate_products.append({
                        'name': real_name,
                        'score': score
                    })
        
        return candidate_products

    def get_product_info(self, product_name: str, info_types: List[str]) -> str:
        """Mendapatkan informasi detail produk berdasarkan nama produk dan tipe info yang diminta"""
        try:
            # Fallback ke produk terakhir jika nama tidak diberikan
            if not product_name or product_name.strip() == "":
                product_name = self.context.last_product
            
            # Validasi eksplisit
            if not product_name or product_name.strip() == "":
                return """<div class="error-response">❌ Produk tidak dikenali. Silakan sebutkan nama produknya.</div>"""
            
            # Karena product_name sekarang ada, kita gunakan untuk pencarian
            # Catatan: product_name bisa dalam bentuk apa pun (mungkin uppercase atau lowercase)
            # Kita akan cari di database dengan lowercase, tapi simpan nama asli untuk ditampilkan nanti
            # Tapi untuk pencarian, kita gunakan versi lowercase
            product_name_for_search = product_name.strip().lower()
            
            # Cari produk dengan similarity tinggi
            candidate_products = self._find_product_candidates(product_name_for_search)

            # Jika hanya menanyakan keberadaan
            if "existence" in info_types:
                if not candidate_products:
                    return f"""<div class="info-response">❌ Tidak ditemukan produk dengan nama '{product_name}'</div>"""
                elif len(candidate_products) == 1:
                    return f"""<div class="info-response">✅ Ya, produk '{candidate_products[0]['name']}' tersedia</div>"""
                else:
                    product_list = ", ".join([p['name'] for p in candidate_products[:3]])
                    return f"""<div class="info-response">✅ Ya, ditemukan {len(candidate_products)} produk: {product_list}</div>"""

            # Jika tidak ada produk yang cocok
            if not candidate_products:
                return f"""<div class="error-response">❌ Produk '{product_name}' tidak ditemukan. Coba gunakan nama lengkap produk.</div>"""

            # Jika hanya satu produk, langsung tampilkan detail
            if len(candidate_products) == 1:
                best_match = candidate_products[0]['name']
                return self._display_product_info(best_match, info_types)

            # Jika ada beberapa produk, tampilkan pilihan
            return self._display_product_selection(candidate_products, info_types)
                
        except Exception as e:
            logger.error(f"Product info error: {e}")
            return """<div class="error-response">❌ Tidak bisa menampilkan informasi untuk produk tersebut</div>"""

    def _display_product_info(self, product_name: str, info_types: List[str]) -> str:
        """Tampilkan informasi spesifik yang diminta untuk satu produk"""
        if not product_name:
            return """<div class="error-response">❌ Nama produk tidak valid.</div>"""

        # Simpan nama asli untuk ditampilkan
        display_name = product_name
        # Untuk pencarian, gunakan lowercase
        product_name_lower = product_name.lower()
        product_data = self.processor.get_product_data(product_name_lower)

        if not product_data:
            return f"""<div class="error-response">❌ Produk '{product_name}' tidak ditemukan dalam database.</div>"""

        # Mapping info types ke field data
        fields_map = {
            "harga": ("harga_text", "Harga"),
            "manfaat": ("manfaat", "Manfaat"),
            "kandungan": ("kandungan", "Kandungan"),
            "cara pakai": ("cara_pemakaian", "Cara Pakai"),
            "lokasi": ("lokasi", "Lokasi")
        }

        # Bangun HTML response hanya untuk info yang diminta
        response_html = f"""<div class="product-info-summary">
            <h3>Informasi Produk: {display_name}</h3>
            <p><b>Brand:</b> {product_data.get('brand', 'Tidak diketahui')}</p>"""

        # Tampilkan hanya info yang diminta
        displayed_info = False
        for info_type in info_types:
            if info_type in fields_map:
                field_key, display_name = fields_map[info_type]
                value = product_data.get(field_key, 'Tidak tersedia')
                
                # Format khusus untuk harga
                if info_type == "harga" and value.replace('.', '').isdigit():
                    try:
                        value = f"Rp {int(float(value)):,}"
                    except:
                        pass
                
                response_html += f"""<p><b>{display_name}:</b> {value}</p>"""
                displayed_info = True
        
        # Jika tidak ada info yang diminta, tampilkan semua
        if not displayed_info:
            for info_type, (field_key, display_name) in fields_map.items():
                value = product_data.get(field_key, 'Tidak tersedia')
                # Format harga khusus jika ditampilkan semua
                if info_type == "harga" and value.replace('.', '').isdigit():
                    try:
                        value = f"Rp {int(float(value)):,}"
                    except:
                        pass
                response_html += f"""<p><b>{display_name}:</b> {value}</p>"""
        
        response_html += "</div>"

        # Tampilkan card produk dengan nama asli
        product_card = self._generate_product_cards([{'name': product_name}])

        # Update last product di context
        self.context.last_product = display_name

        return f"""
        <div class="product-info-response">
            {response_html}
            <div class="product-card-container">
                {product_card}
            </div>
        </div>
        """

    def _display_product_selection(self, candidate_products: List[dict], info_types: List[str]) -> str:
        """Tampilkan pilihan produk ketika ada beberapa kemungkinan"""
        # Simpan daftar produk di context untuk referensi selanjutnya
        self.context.pending_products = [cp['name'] for cp in candidate_products]
        self.context.pending_info_types = info_types
        
        # Generate product cards untuk semua kandidat
        product_cards = self._generate_product_cards([
            {'name': cp['name']} for cp in candidate_products[:5]  # Batasi max 5 produk
        ])
        
        info_text = ", ".join(info_types) if info_types else "detail"
        
        return f"""
        <div class="multiple-products">
            <p>🔍 Ditemukan {len(candidate_products)} produk dengan nama mirip:</p>
            <div class="product-grid" style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px;">
                {product_cards}
            </div>
            <p class="selection-hint">
                Klik <strong>"Lihat Detail"</strong> pada produk yang Anda maksud
            </p>
        </div>
        """
    
    def _handle_statistical_question(self, analysis: dict) -> str:
        """Generate statistics based on the analysis result"""
        try:
            # Pastikan all_products ada dan berisi data
            if not self.context.all_products or len(self.context.all_products) == 0:
                logger.warning("Tidak ada produk yang tersedia untuk analisis statistik")
                return "<div class='stat-response'>Belum ada data produk yang tersedia</div>"
            
            # Initialize counters
            total_count = 0
            skin_counts = {skin: 0 for skin in SKIN_TYPES}
            brand_counts = {brand: 0 for brand in BRANDS}
            type_counts = {pt: 0 for pt in SUPPORTED_PRODUCT_TYPES}
            
            # Extract criteria from analysis
            product_type = analysis.get("product_type", "")
            skin_types = analysis.get("skin_types", [])
            brands = analysis.get("brands", [])
            
            # Count products matching the criteria
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if not product_data:
                    continue
                    
                # Check if product matches the criteria
                type_match = (not product_type) or (product_type.lower() in product_data.get('type_produk', '').lower())
                skin_match = (not skin_types) or any(
                    skin in [s.lower() for s in product_data.get('jenis_kulit', [])] 
                    for skin in skin_types
                )
                brand_match = (not brands) or any(
                    brand.lower() in product_data.get('brand', '').lower()
                    for brand in brands
                )
                
                if type_match and skin_match and brand_match:
                    total_count += 1
                    
                    # Count skin types
                    for skin in product_data.get('jenis_kulit', []):
                        if skin in skin_counts:
                            skin_counts[skin] += 1
                    
                    # Count brands
                    brand = product_data.get('brand', 'unknown')
                    if brand in brand_counts:
                        brand_counts[brand] += 1
                    
                    # Count product types
                    for pt in SUPPORTED_PRODUCT_TYPES:
                        if pt in product_data.get('type_produk', '').lower():
                            type_counts[pt] += 1
            
            # Build response based on criteria
            response_parts = []
            
            # Total products
            if not product_type and not skin_types and not brands:
                response_parts.append(f"📊 Total produk terdaftar: <b>{total_count}</b>")
            
            # Product type statistics
            if product_type:
                count = type_counts.get(product_type.lower(), 0)
                response_parts.append(f"📊 Total produk {product_type}: <b>{count}</b>")
            
            # Skin type statistics
            if skin_types:
                skin_text = ", ".join(skin_types)
                response_parts.append(f"📊 Produk untuk kulit {skin_text}: <b>{total_count}</b>")
            
            # Brand statistics
            if brands:
                brand_text = ", ".join(brands)
                response_parts.append(f"📊 Produk dari {brand_text}: <b>{total_count}</b>")
            
            # Combination statistics
            if product_type and skin_types:
                skin_text = ", ".join(skin_types)
                count = type_counts.get(product_type.lower(), 0)
                response_parts.append(f"📊 Produk {product_type} untuk kulit {skin_text}: <b>{count}</b>")
            
            if product_type and brands:
                brand_text = ", ".join(brands)
                count = type_counts.get(product_type.lower(), 0)
                response_parts.append(f"📊 Produk {product_type} dari {brand_text}: <b>{count}</b>")
            
            # Add skin type distribution if relevant
            if skin_types or product_type:
                skin_info = [f"{skin}: {count}" for skin, count in skin_counts.items() if count > 0]
                if skin_info:
                    response_parts.append(f"Jenis kulit: {', '.join(skin_info)}")
            
            # Add brand distribution if relevant
            if brands or product_type:
                brand_info = [f"{brand}: {count}" for brand, count in brand_counts.items() if count > 0]
                if brand_info:
                    response_parts.append(f"Merek: {', '.join(brand_info)}")
            
            if not response_parts:
                return "<div class='stat-response'>Tidak ada data statistik yang ditemukan.</div>"
            
            return f"""
            <div class="stat-response">
                {'<br>'.join([f'<p>{part}</p>' for part in response_parts])}
            </div>
            """
        except Exception as e:
            logger.error(f"Error in statistical handling: {e}")
            return f"<div class='error-response'>Terjadi kesalahan dalam memproses data statistik: {e}</div>"

    def generate_response(self, question: str) -> str:
        try:
            # Tambahkan pertanyaan ke history
            self.context.add_message("user", question)
            
            # Analisis pertanyaan dengan LLM
            analysis = self._analyze_question_with_llm(question)
            
            # Override untuk follow-up questions
            if (self.context.last_product and 
                len(question.split()) <= 3 and 
                any(word in question.lower() for word in INFO_TYPES)):
                logger.info(f"Overriding question type to info_produk for follow-up question: {question}")
                analysis['question_type'] = 'info_produk'
                analysis['product_name'] = self.context.last_product
                # Set info_types from the question
                analysis['info_types'] = [word for word in INFO_TYPES if word in question.lower()]
            
            # Override khusus untuk permintaan brand setelah rekomendasi
            if (self.context.last_intent == "rekomendasi" and 
                analysis.get("brands") and 
                not analysis.get("product_name") and
                analysis["question_type"] != "rekomendasi"):
                logger.info(f"Override question type to rekomendasi for brand request after recommendation")
                analysis["question_type"] = "rekomendasi"
            
            # Override untuk pola "brand + produk"
            if (analysis.get("brands") and 
                analysis.get("product_type") and 
                not analysis.get("product_name") and
                analysis["question_type"] != "rekomendasi"):
                logger.info(f"Override question type to rekomendasi for brand+product pattern")
                analysis["question_type"] = "rekomendasi"
            
            # Log hasil analisis (hanya satu kali)
            logger.info(f"Hasil analisis: {json.dumps(analysis, indent=2)}")
            logger.info(f"Konteks saat ini: {self.context.get_context_summary()}")
            
            # Handle out-of-scope questions
            if not analysis.get("is_face_skincare", True):
                response = self._get_out_of_scope_response(question)
                self.context.add_message("assistant", response)
                return response
            
            # Handle reset request
            if analysis["question_type"] == "reset":
                self.context.reset_context()
                response = f"""<div class="reset-response">Konfigurasi telah direset. Silakan mulai percakapan baru.</div>"""
                self.context.add_message("assistant", response)
                return response
            
            # Update konteks berdasarkan analisis
            self.context.update_context(analysis)
            logger.info(f"Konteks setelah update: {self.context.get_context_summary()}")
            
            # Handle question types
            q_type = analysis["question_type"]
            
            # Special handling for "rekomendasi_lagi"
            if q_type == "rekomendasi_lagi":
                if not self.context.active_recommendation:
                    response = f"""<div class="error-response">Anda belum meminta rekomendasi sebelumnya. Silakan mulai dengan 'rekomendasi serum' atau sejenisnya.</div>"""
                    self.context.add_message("assistant", response)
                    return response
                
                # Gunakan kriteria terakhir jika tidak ada perubahan
                if not any([
                    set(analysis.get("skin_types", [])) != set(self.context.skin_type),
                    analysis.get("product_type") != self.context.product_type,
                    analysis.get("max_price") != self.context.filters["max_price"],
                    analysis.get("min_price") != self.context.filters["min_price"],
                    set(analysis.get("ingredients", [])) != set(self.context.filters["ingredients"]),
                    set(analysis.get("brands", [])) != set(self.context.filters["brands"])
                ]) and self.last_recommendation_criteria:
                    self.context.skin_type = self.last_recommendation_criteria["skin_types"].copy()
                    self.context.product_type = self.last_recommendation_criteria["product_type"]
                    self.context.filters = {
                        "max_price": self.last_recommendation_criteria["filters"]["max_price"],
                        "min_price": self.last_recommendation_criteria["filters"]["min_price"],
                        "ingredients": self.last_recommendation_criteria["filters"]["ingredients"].copy(),
                        "brands": self.last_recommendation_criteria["filters"]["brands"].copy()
                    }
                
                response = self.get_recommendations(question)
                self.context.last_intent = "rekomendasi"
                self.context.add_message("assistant", response)
                return response
            
            # Handle filter updates
            if q_type == "filter_update":
                if not self.context.active_recommendation:
                    response = f"""<div class="error-response">Anda belum meminta rekomendasi sebelumnya. Silakan mulai dengan 'rekomendasi serum' atau sejenisnya.</div>"""
                    self.context.add_message("assistant", response)
                    return response
                
                response = self.get_recommendations(question)
                self.context.last_intent = "rekomendasi"
                self.context.add_message("assistant", response)
                return response
            
            # Simple responses for greetings
            if q_type == "sapaan":
                response = f"""<div class="greeting">Halo! Saya asisten skincare, ada yang bisa saya bantu?</div>"""
                self.context.last_intent = "sapaan"
                self.context.add_message("assistant", response)
                return response
                
            if q_type == "identity":
                response = f"""<div class="identity">Saya asisten virtual khusus rekomendasi skincare wajah. Ada yang bisa saya bantu?</div>"""
                self.context.last_intent = "identity"
                self.context.add_message("assistant", response)
                return response
            
            # Recommendation request
            if q_type == "rekomendasi":
                # Nonaktifkan mode rekomendasi jika sebelumnya aktif untuk info produk
                self.context.active_recommendation = True
                
                # Jika tidak ada jenis kulit, minta dulu
                if not self.context.skin_type and not self.context.product_type:
                    response = f"""<div class="clarification">Tentu, bisa Anda sebutkan jenis kulit Anda? (misal: kering, berminyak, kombinasi, sensitif, berjerawat)</div>"""
                    self.context.last_intent = "rekomendasi"
                    self.context.add_message("assistant", response)
                    return response
                else:
                    response = self.get_recommendations(question)
                    self.context.last_intent = "rekomendasi"
                    self.context.add_message("assistant", response)
                    return response
                
            # Handle skin type updates in recommendation context
            if self.context.active_recommendation and analysis["skin_types"]:
                response = self.get_recommendations(question)
                self.context.last_intent = "rekomendasi"
                self.context.add_message("assistant", response)
                return response
                
            # Handle filter updates in recommendation context
            if self.context.active_recommendation and (
                analysis["max_price"] is not None or 
                analysis["min_price"] is not None or 
                analysis["ingredients"] or
                analysis["brands"]
            ):
                response = self.get_recommendations(question)
                self.context.last_intent = "rekomendasi"
                self.context.add_message("assistant", response)
                return response
                
            # Product info request
            elif q_type == "info_produk":
                # Nonaktifkan mode rekomendasi
                if self.context.last_intent in ["rekomendasi", "statistical"]:
                    logger.info("Reset konteks untuk pertanyaan info produk")
                    self.context.reset_filters()
                    self.context.active_recommendation = False
                
                product_name = analysis.get("product_name", "").strip()
                info_types = analysis.get("info_types", [])
                
                # Jika ada pending products (user memilih dari daftar)
                if self.context.pending_products and ("detail" in question.lower() or "pilih" in question.lower()):
                    # Cari produk yang dipilih dari daftar pending
                    for p_name in self.context.pending_products:
                        if p_name.lower() in question.lower():
                            response = self._display_product_info(p_name, self.context.pending_info_types)
                            self.context.last_product = p_name
                            self.context.last_intent = "info_produk"
                            self.context.pending_products = []  # Reset pending
                            break
                    else:
                        response = f"""<div class="error-response">🚫 Produk tidak dikenali. Silakan pilih dari daftar sebelumnya.</div>"""
                else:
                    # Jika tidak ada pending, proses normal
                    if not product_name and self.context.last_product:
                        product_name = self.context.last_product
                    
                    if not product_name:
                        response = f"""<div class="clarification">🔍 Produk apa yang Anda maksud? Silakan sebutkan nama produknya.</div>"""
                    else:
                        response = self.get_product_info(product_name, info_types)
                        # Update last product jika ditemukan tepat satu
                        if "multiple-products" not in response and "error-response" not in response:
                            self.context.last_product = product_name
                        self.context.last_intent = "info_produk"
                
                self.context.add_message("assistant", response)
                return response
            
            # Statistical question
            elif q_type == "statistical":
                if self.context.last_intent != "statistical":
                    logger.info("Reset konteks untuk pertanyaan statistik")
                    self.context.reset_filters()
                    self.context.active_recommendation = False
                    self.context.last_product = None
                
                response = self._handle_statistical_question(analysis)
                self.context.last_intent = "statistical"
                self.context.add_message("assistant", response)
                return response
                    
            # General question about product info
            elif self.context.last_product and any(word in question.lower() for word in INFO_TYPES):
                info_types = [word for word in INFO_TYPES if word in question.lower()]
                response = self.get_product_info(self.context.last_product, info_types)
                self.context.last_intent = "info_produk"
                self.context.add_message("assistant", response)
                return response
            
            # General question
            else:
                documents = self.vector_store.similarity_search(question, k=3)
                doc_texts = [d.page_content for d in documents]
                context = self.context.get_recent_history()
                
                response = self._get_rag_response(
                    question=question,
                    context=context,
                    documents=doc_texts
                )
                self.context.active_recommendation = False
                self.context.last_intent = "general"
                self.context.add_message("assistant", response)
                return response
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return f"""<div class="error-response">❌ Terjadi kesalahan. Silakan coba lagi.</div>"""

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
    print("Sistem siap. Ketik 'exit' untuk keluar.")
    
    while True:
        try:
            question = input("\nPertanyaan: ").strip()
            if question.lower() == 'exit':
                break
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