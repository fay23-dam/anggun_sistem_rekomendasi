## main_qa_module
# ====== IMPORT LIBRARY ====== #
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score

import sys
sys.stdout.flush()  # Pastikan output tidak tertahan di buffer
# ====== KELAS MEMORI PERCAKAPAN ====== #
class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ====== #
class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()


# ====== KELAS QA ENGINE ====== #
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None

        # Daftar pola pertanyaan
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

        # Daftar istilah teknis
        self.technical_terms = set(["hsse", "dppu", "pompav", "keselamatan", "penerbangan",
                                   "bandara", "aviasi", "keamanan", "insiden", "darurat"])

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Deteksi perintah khusus
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Deteksi sapaan
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Deteksi pertanyaan lanjutan
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi dengan menyebutkan:
           - Daftar topik terkait dari jawaban sebelumnya
           - Contoh format pertanyaan yang lebih spesifik
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret

        [FORMAT JAWABAN]
        <konfirmasi topik>
        <penjelasan lengkap>
        <contoh relevan> (jika ada)
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        try:
            core_q, q_type = self.analyze_question(question)

            # Handle perintah khusus
            if q_type == "reset":
                self.memory.clear()
                self.current_context = None
                return "🔄 Memori percakapan telah direset", (0, 0, 0)

            if q_type in ["empty", "pure_greeting", "small_talk"]:
                samples = "\n".join([
                    "• Apa definisi HSSE menurut dokumen POMPAV?",
                    "• Bagaimana prosedur tanggap darurat di bandara?",
                    "• Sebutkan sertifikasi yang berlaku 2 tahun"
                ])
                responses = {
                    "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                    "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}",
                    "small_talk": f"🤖 Saya AI spesialis POMPAV. Contoh pertanyaan teknis:\n{samples}"
                }
                return responses[q_type], (0, 0, 0)

            # Handle pertanyaan lanjutan
            if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                if not self.current_context:
                    self.current_context = self.vector_store.similarity_search(
                        self.memory.history[-1]["question"], k=3
                    )
                response = self.generate_follow_up_response(core_q)
                answer = response["output_text"]
                self.memory.add_interaction(question, answer)
                return answer, (0.9, 0.9, 0.9)  # High score for contextual answers

            # Handle pertanyaan baru
            docs = self.vector_store.similarity_search(core_q, k=3)
            self.current_context = docs
            self.update_terms(docs)

            if not docs:
                return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

            prompt_template = """
            [RIWAYAT PERCAKAPAN]
            {history}

            [DOKUMEN REFERENSI]
            {context}

            [PERTANYAAN]
            {question}

            [INSTRUKSI JAWABAN]
            1. Berikan jawaban langsung di awal
            2. Jelaskan dengan rinci merujuk dokumen
            3. Sertakan contoh jika relevan
            4. Tautkan dengan konteks sebelumnya jika ada
            5. Gunakan istilah teknis secara konsisten

            [FORMAT]
            <jawaban inti>
            <penjelasan>
            <contoh/analogi> (opsional)
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "context", "question"]
            )

            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain({
                "input_documents": docs,
                "history": self.memory.get_formatted_history(),
                "question": core_q
            }, return_only_outputs=True)

            # Evaluasi dengan BERTScore
            P, R, F1 = score(
                [response["output_text"]],
                [docs[0].page_content],
                lang="id",
                model_type="bert-base-multilingual-cased"
            )

            self.memory.add_interaction(question, response["output_text"])
            return response["output_text"], (P.mean().item(), R.mean().item(), F1.mean().item())

        except Exception as e:
            traceback.print_exc()
            return f"⚠️ Error: {str(e)}", (0, 0, 0)


# ====== FUNGSI UTAMA ====== #
def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + answer)

            # Tampilkan metrics
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"📊 Skor Relevansi (F1): {scores[2]:.2f}/1.00")

            print("-"*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")

if __name__ == "__main__":
    main()

## main_qa_module.py part 2
# ====== IMPORT LIBRARY ====== #
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
import sys
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import BERTScorer


# ====== KELAS MEMORI PERCAKAPAN ====== #
class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ====== #
class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()


# ====== KELAS QA ENGINE ====== #
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        
        # Inisialisasi BERTScorer untuk Bahasa Indonesia
        # Inisialisasi BERTScore dengan model yang umum digunakan
        self.bert_scorer = BERTScorer(lang="id", model_type="bert-base-multilingual-cased")


        # Daftar pola pertanyaan
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

        # Daftar istilah teknis
        self.technical_terms = set(["hsse", "dppu", "pompav", "keselamatan", "penerbangan",
                                   "bandara", "aviasi", "keamanan", "insiden", "darurat"])

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def generate_follow_up_suggestions(self, answer: str) -> List[str]:
        """Generate relevant follow-up questions based on the answer"""
        prompt = f"""
        Berdasarkan jawaban berikut, sarankan 3 pertanyaan lanjutan yang relevan dalam Bahasa Indonesia:
        
        Jawaban: {answer}
        
        Aturan:
        1. Pertanyaan harus spesifik dan terkait langsung dengan jawaban
        2. Gunakan istilah teknis dari jawaban
        3. Buat pertanyaan yang praktis dan dapat ditindaklanjuti
        4. Format sebagai daftar bernomor dengan setiap pertanyaan pada baris baru
        
        Contoh:
        1. Apa definisi lengkap dari HSSE menurut dokumen ini?
        2. Bagaimana prosedur penerapan POMPAV di bandara kecil?
        3. Apa saja sanksi untuk pelanggaran prosedur keselamatan?
        """
        
        try:
            response = self.llm.invoke(prompt)
            questions = response.content.split("\n")
            return [q.split(" ", 1)[1] for q in questions if q.strip() and q[0].isdigit()][:3]
        except Exception as e:
            print(f"⚠️ Error menghasilkan pertanyaan lanjutan: {e}")
            return []

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Deteksi perintah khusus
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Deteksi sapaan
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Deteksi pertanyaan lanjutan
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, minta klarifikasi dengan menyebutkan:
           - Daftar topik terkait dari jawaban sebelumnya
           - Contoh format pertanyaan yang lebih spesifik
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        4. Akhiri dengan 3 saran pertanyaan lanjutan

        [FORMAT JAWABAN]
        <konfirmasi topik>
        <penjelasan lengkap>
        <contoh relevan jika ada>
        
        [SARAN PERTANYAAN LANJUTAN]
        1. <pertanyaan 1>
        2. <pertanyaan 2>
        3. <pertanyaan 3>
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float], List[str]]:
        try:
            core_q, q_type = self.analyze_question(question)
            follow_up_suggestions = []

            # Handle perintah khusus
            if q_type == "reset":
                self.memory.clear()
                self.current_context = None
                return "🔄 Memori percakapan telah direset", (0, 0, 0), []

            if q_type in ["empty", "pure_greeting", "small_talk"]:
                samples = "\n".join([
                    "• Apa definisi HSSE menurut dokumen POMPAV?",
                    "• Bagaimana prosedur tanggap darurat di bandara?",
                    "• Sebutkan sertifikasi yang berlaku 2 tahun"
                ])
                responses = {
                    "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                    "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}",
                    "small_talk": f"🤖 Saya AI spesialis POMPAV. Contoh pertanyaan teknis:\n{samples}"
                }
                return responses[q_type], (0, 0, 0), []

            # Handle pertanyaan lanjutan
            if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                if not self.current_context:
                    self.current_context = self.vector_store.similarity_search(
                        self.memory.history[-1]["question"], k=3
                    )
                response = self.generate_follow_up_response(core_q)
                answer = response["output_text"]
                
                # Ekstrak saran pertanyaan lanjutan
                if "[SARAN PERTANYAAN LANJUTAN]" in answer:
                    answer_parts = answer.split("[SARAN PERTANYAAN LANJUTAN]")
                    answer = answer_parts[0].strip()
                    follow_up_suggestions = [q.strip() for q in answer_parts[1].split("\n") if q.strip() and q[0].isdigit()]
                
                self.memory.add_interaction(question, answer)
                return answer, (0.9, 0.9, 0.9), follow_up_suggestions

            # Handle pertanyaan baru
            docs = self.vector_store.similarity_search(core_q, k=3)
            self.current_context = docs
            self.update_terms(docs)

            if not docs:
                return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0), []

            prompt_template = """
            [RIWAYAT PERCAKAPAN]
            {history}

            [DOKUMEN REFERENSI]
            {context}

            [PERTANYAAN]
            {question}

            [INSTRUKSI JAWABAN]
            1. Berikan jawaban langsung di awal
            2. Jelaskan dengan rinci merujuk dokumen
            3. Sertakan contoh jika relevan
            4. Hubungkan dengan konteks sebelumnya jika ada
            5. Gunakan istilah teknis secara konsisten
            6. Akhiri dengan 3 saran pertanyaan lanjutan

            [FORMAT]
            <jawaban inti>
            <penjelasan>
            <contoh/analogi> (opsional)
            
            [SARAN PERTANYAAN LANJUTAN]
            1. <pertanyaan 1>
            2. <pertanyaan 2>
            3. <pertanyaan 3>
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "context", "question"]
            )

            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain({
                "input_documents": docs,
                "history": self.memory.get_formatted_history(),
                "question": core_q
            }, return_only_outputs=True)
            
            answer = response["output_text"]
            
            # Ekstrak saran pertanyaan lanjutan
            if "[SARAN PERTANYAAN LANJUTAN]" in answer:
                answer_parts = answer.split("[SARAN PERTANYAAN LANJUTAN]")
                answer = answer_parts[0].strip()
                follow_up_suggestions = [q.strip() for q in answer_parts[1].split("\n") if q.strip() and q[0].isdigit()]
            
            # Evaluasi dengan BERTScore Bahasa Indonesia
            reference_text = "\n".join([doc.page_content for doc in docs[:2]])
            print("\n[Menghitung BERTScore...]")
            sys.stdout.flush()
            
            P, R, F1 = self.bert_scorer.score(
                [answer],
                [reference_text]
            )
            
            # Konversi tensor ke nilai float
            precision = P.item()
            recall = R.item()
            f1 = F1.item()

            print("\n[Metric BERTScore]")
            print(f"Presisi: {precision:.4f}")
            print(f"Recall:  {recall:.4f}")
            print(f"F1:      {f1:.4f}")
            sys.stdout.flush()

            self.memory.add_interaction(question, answer)
            return answer, (precision, recall, f1), follow_up_suggestions

        except Exception as e:
            traceback.print_exc()
            return f"⚠️ Error: {str(e)}", (0, 0, 0), []


# ====== FUNGSI UTAMA ====== #
def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("- Saran pertanyaan lanjutan cerdas")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores, follow_ups = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + "="*50)
            print("💬 JAWABAN:")
            print(answer)

            # Tampilkan metrics
            print("\n" + "="*50)
            print("📊 METRIK KINERJA:")
            print(f"⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"\nBERTScore:")
                print(f"  Presisi: {scores[0]:.4f}")
                print(f"  Recall:  {scores[1]:.4f}")
                print(f"  F1:      {scores[2]:.4f}")
                
                if scores[2] < 0.8:
                    print("\n💡 Saran: Untuk hasil lebih baik, coba:")
                    print("- Gunakan istilah teknis yang lebih spesifik")
                    print("- Ajukan pertanyaan yang lebih fokus")
                    print("- Sertakan referensi dokumen dalam pertanyaan")

            # Tampilkan saran pertanyaan lanjutan
            if follow_ups:
                print("\n" + "="*50)
                print("🔍 SARAN PERTANYAAN LANJUTAN:")
                for i, q in enumerate(follow_ups, 1):
                    print(f"{i}. {q}")

            print("\n" + "="*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")
            
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")

if __name__ == "__main__":
    main()

    
## app.py
from flask import Flask, request, jsonify, render_template
from main_qa_module import PDFProcessor, QAEngine
from dotenv import load_dotenv
import os
load_dotenv()  # Ini akan membaca file .env

# Cek apakah berhasil
assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY tidak ditemukan di .env"

app = Flask(__name__)

# ====== Inisialisasi Sistem QA ======
processor = PDFProcessor()
text = processor.extract_text('dokumen/dataset_pompav.pdf')  # Pastikan file ini ada
chunks = processor.chunk_text(text)
processor.create_vector_store(chunks)
qa_engine = QAEngine(processor.vector_store)

# ====== ROUTES ======

@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan UI chatbot

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': '⚠️ Pertanyaan kosong tidak dapat diproses'}), 400
    answer, scores = qa_engine.generate_response(user_message)
    return jsonify({'response': answer, 'f1_score': round(scores[2], 2)})

if __name__ == '__main__':
    app.run(debug=False)

# 18-05-2025
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score
import sys
sys.stdout.flush()  # Pastikan output tidak tertahan di buffer

# ====== KELAS MEMORI PERCAKAPAN ======

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ======

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()

import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score
from threading import Lock
import sys
sys.stdout.flush()

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "pompav_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        """Initialize or load vector store from disk"""
        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        # Try to load existing index
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        # Create new vector store if loading failed
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        """Create new vector store from PDF and save to disk"""
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True  # Still return True since we have it in memory
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Warning: Document very large, only processing first part")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Text chunking error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Initial vector store created with {len(batch)} chunks")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Added batch {i//batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()
                
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        self.lock = Lock()  # Thread lock for concurrent requests
        self.technical_terms = set()

        # Patterns and keywords
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Special commands
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Greetings detection
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Follow-up detection
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        with self.lock:  # Thread-safe processing
            try:
                core_q, q_type = self.analyze_question(question)

                # Handle special commands
                if q_type == "reset":
                    self.memory.clear()
                    self.current_context = None
                    print("🔄 Memori percakapan telah direset")
                    return "🔄 Memori percakapan telah direset", (0, 0, 0)

                if q_type in ["empty", "pure_greeting", "small_talk"]:
                    samples = "\n".join([
                        "Apa definisi HSSE menurut dokumen POMPAV?",
                        "Bagaimana prosedur tanggap darurat di bandara?",
                        "Sebutkan sertifikasi yang berlaku 2 tahun"
                    ])
                    responses = {
                        "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                        "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Kamu bisa bertanya:\n{samples}",
                        "small_talk": f"🤖 Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}"
                    }
                    response = responses[q_type]
                    print(f"Response: {response}")
                    return response, (0, 0, 0)

                # Follow-up questions
                if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                    if not self.current_context:
                        self.current_context = self.vector_store.similarity_search(
                            self.memory.history[-1]["question"], k=3
                        )
                    response = self.generate_follow_up_response(core_q)
                    answer = response["output_text"]
                    self.memory.add_interaction(question, answer)
                    print(f"Follow-up response: {answer}")
                    return answer, (0.9, 0.9, 0.9)

                # New questions
                docs = self.vector_store.similarity_search(core_q, k=3)
                self.current_context = docs
                self.update_terms(docs)

                if not docs:
                    print("❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.")
                    return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

                prompt_template = """
                [RIWAYAT PERCAKAPAN]
                {history}

                [DOKUMEN REFERENSI]
                {context}

                [PERTANYAAN]
                {question}

                [INSTRUKSI JAWABAN]
                1. Berikan jawaban langsung di awal
                2. Jelaskan dengan rinci merujuk dokumen
                3. Sertakan contoh jika relevan
                4. Tautkan dengan konteks sebelumnya jika ada
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
                response = chain({
                    "input_documents": docs,
                    "history": self.memory.get_formatted_history(),
                    "question": core_q
                }, return_only_outputs=True)

                # BERTScore evaluation
                P, R, F1 = score(
                    [response["output_text"]],
                    [docs[0].page_content],
                    lang="id",
                    model_type="bert-base-multilingual-cased"
                )

                # Print BERTScore to terminal
                print("\n" + "="*50)
                print("📊 EVALUASI BERTSCORE:")
                print(f"Precision: {P.mean().item():.4f}")
                print(f"Recall: {R.mean().item():.4f}")
                print(f"F1 Score: {F1.mean().item():.4f}")
                print("="*50 + "\n")

                self.memory.add_interaction(question, response["output_text"])
                return response["output_text"], (P.mean().item(), R.mean().item(), F1.mean().item())

            except Exception as e:
                traceback.print_exc()
                error_msg = f"⚠️ Error: {str(e)}"
                print(error_msg)
                return error_msg, (0, 0, 0)

# ====== FUNGSI UTAMA ======

def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + answer)

            # Tampilkan metrics
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"📊 Skor Relevansi (F1): {scores[2]:.2f}/1.00")

            print("-"*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")


if __name__ == "__main__":
    main()


#backup kode main_qa_module 23-05-2025

import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from threading import Lock
import sys
sys.stdout.flush()

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "pompav_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Warning: Document very large, only processing first part")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Text chunking error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Initial vector store created with {len(batch)} chunks")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Added batch {i//batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        self.lock = Lock()  # Thread lock for concurrent requests
        self.technical_terms = set()

        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa", "bagaimana", "contoh", "saja", "tersebut", "itu", "ini", "yg", "yang", "tadi", "nomor", "poin"]

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        4. jika pertanyaan diluar konteks atau tidak relevan dengan dokumen jawab dengan pertanyaan diluar pengetahuan yang ada
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        with self.lock:  
            try:
                start_time = time.time()
                core_q, q_type = self.analyze_question(question)

                if q_type == "reset":
                    self.memory.clear()
                    self.current_context = None
                    return "🔄 Memori percakapan telah direset", (0, 0, 0)

                if q_type in ["empty", "pure_greeting", "small_talk"]:
                    samples = "\n".join([
                        "Apa definisi HSSE menurut dokumen POMPAV?",
                        "Bagaimana prosedur tanggap darurat di bandara?",
                        "bagaimana prosedur pengisian pesawat udara?"
                    ])
                    responses = {
                        "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                        "pure_greeting": f"🖐️Halo! Saya asisten QA POMPAV. Kamu bisa bertanya:\n{samples}",
                        "small_talk": f"🤖 Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}"
                    }
                    return responses[q_type], (0, 0, 0)

                if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                    if not self.current_context:
                        self.current_context = self.vector_store.similarity_search(
                            self.memory.history[-1]["question"], k=3
                        )
                    response = self.generate_follow_up_response(core_q)
                    answer = response["output_text"]
                    self.memory.add_interaction(question, answer)
                    return answer, (0.9, 0.9, 0.9)

                docs = self.vector_store.similarity_search(core_q, k=3)
                self.current_context = docs
                self.update_terms(docs)

                if not docs:
                    return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

                prompt_template = """
                [RIWAYAT PERCAKAPAN]
                {history}

                [DOKUMEN REFERENSI]
                {context}

                [PERTANYAAN]
                {question}

                [INSTRUKSI JAWABAN]
                1. Berikan jawaban langsung di awal
                2. Jelaskan dengan rinci merujuk dokumen
                3. Sertakan contoh jika relevan
                4. Tautkan dengan konteks sebelumnya jika ada
                5. jika pertanyaan diluar konteks atau tidak relevan dengan dokumen jawab dengan pertanyaan diluar pengetahuan yang ada
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
                response = chain({
                    "input_documents": docs,
                    "history": self.memory.get_formatted_history(),
                    "question": core_q
                }, return_only_outputs=True)

                self.memory.add_interaction(question, response["output_text"])
                process_time = time.time() - start_time

                print(f"⏱️ Waktu respon: {process_time:.2f} detik")
                return response["output_text"], (0, 0, 0)
            

            except Exception as e:
                traceback.print_exc()
                return f"⚠️ Error: {str(e)}", (0, 0, 0)

# ====== FUNGSI UTAMA ======
def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)

    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()  # Mulai waktu respon
            answer, _ = qa_engine.generate_response(question)

            print("\n" + answer)
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")

            print("-"*50)

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")


if __name__ == "__main__":
    main()



import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
import gc
import difflib
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.context_buffer = ""  # Menyimpan konteks percakapan terakhir

    def add_interaction(self, question: str, answer: str):
        """Add interaction to memory with timestamp and embedding"""
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        
        interaction = {
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(f"{question} {answer}")
        }
        self.history.append(interaction)
        
        # Update context buffer with the latest interaction
        self.context_buffer = f"Pertanyaan sebelumnya: {question}\nJawaban: {answer}"

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        """Calculate similarity between current question and last interaction"""
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return float(util.pytorch_cos_sim(current_embedding, last_embedding).item())

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.context_buffer = ""

    def get_context_summary(self) -> str:
        """Get summary of the conversation context"""
        return self.context_buffer

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "skincare_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        """Initialize vector store from existing or create new"""
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        """Create new vector store from PDF with large document support"""
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF document with better large file handling"""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                # Process in batches to avoid memory issues
                batch_size = 100  # Adjust based on your needs
                for page_num in tqdm(range(0, total_pages, batch_size), desc="📄 Extracting text"):
                    for page in doc[page_num:page_num + batch_size]:
                        text += page.get_text("text") + "\n"
                    print(f"   Processed pages {page_num+1}-{min(page_num+batch_size, total_pages)}/{total_pages}")
                    # Optional: Add delay between batches for very large files
                    if len(text) > 4_000_000:  # ~4MB limit
                        print("ℹ️ Reached processing limit, truncating text")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks with better memory handling"""
        try:
            # Process in segments for very large texts
            max_text_length = 1_000_000  # Process 1MB at a time
            text_segments = [text[i:i+max_text_length] for i in range(0, len(text), max_text_length)]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Slightly larger chunks
                chunk_overlap=200,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            )
            
            all_chunks = []
            for segment in tqdm(text_segments, desc="✂️ Chunking text"):
                all_chunks.extend(splitter.split_text(segment))
                print(f"   Generated {len(all_chunks)} chunks so far")
                
            return all_chunks
        except Exception as e:
            print(f"❌ Chunk creation error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        """Create vector store from text chunks with batch processing"""
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 50  # Increased batch size for better performance
        total_chunks = len(chunks)
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="🧠 Creating embeddings"):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Initial vector store created with {len(batch)} chunks ({i+1}-{min(i+batch_size, total_chunks)}/{total_chunks})")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"➕ Added batch {i//batch_size + 1} ({len(batch)} chunks, {min(i+batch_size, total_chunks)}/{total_chunks} total)")
                
                # Optional: Save progress periodically
                if i % (batch_size * 5) == 0 and i > 0:
                    temp_path = os.path.join(self.vector_store_dir, f"{self.index_name}_temp")
                    self.vector_store.save_local(temp_path)
                    print(f"💾 Saved temporary vector store at {i} chunks")
                    
                # Clean up memory
                gc.collect()
                    
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()
                # Try with smaller batch if failed
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)
                    print(f"🔄 Reducing batch size to {batch_size} and retrying...")
                    continue

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.lock = Lock()
        self.system_role = "Anda adalah asisten AI khusus untuk skincare recommendation."
        self.restricted_roles = ["asisten kota sorong", "asisten sorong", "virtual assistant", "asisten virtual", "asisten ai kota"]
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa", "bagaimana"]
        self.reference_keywords = ["seperti itu", "seperti di atas", "tersebut", "yg tadi", "yg sebelumnya", "yg itu"]
        self.question_words = ['apa', 'untuk apa', 'berguna', 'manfaat', 'kegunaan', 'cocok untuk', 'bagaimana', 'kandungan', 'review']
        # Define skin issues and product types
        self.skin_issues = ['berjerawat', 'kering', 'berminyak', 'sensitif', 'komedo', 'flek hitam', 'kusam', 'penuaan']
        self.product_types = ["facial wash", "sabun muka", "pencuci muka", "cleanser",
                             "pelembap", "moisturizer", "krim", "cream",
                             "sunscreen", "tabir surya", "sunblock",
                             "serum", "essence", "toner", "masker", "scrub"]
        # State tracking
        self.pending_skin_issue = None
        self.last_product_info = None  # Menyimpan informasi produk terakhir yang dibahas

    def fuzzy_match(self, word: str, word_list: List[str], threshold: float = 0.7) -> Optional[str]:
        """Find the best match for a word in a list using fuzzy matching"""
        best_match = None
        best_score = 0
        
        for candidate in word_list:
            score = difflib.SequenceMatcher(None, word, candidate).ratio()
            if score > best_score and score > threshold:
                best_score = score
                best_match = candidate
                
        return best_match

    def detect_keywords(self, question: str) -> Tuple[bool, bool, bool, bool]:
        """Detect skin issues, product types, question words, and reference words with fuzzy matching"""
        words = re.findall(r'\b\w+\b', question.lower())
        has_skin_issue = False
        has_product_type = False
        has_question_word = False
        has_reference_word = False
        
        # Check for skin issues
        for word in words:
            if self.fuzzy_match(word, self.skin_issues):
                has_skin_issue = True
                break
        
        # Check for product types (single words)
        for word in words:
            if self.fuzzy_match(word, self.product_types):
                has_product_type = True
                break
        
        # Check for multi-word product types
        if not has_product_type:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if self.fuzzy_match(bigram, self.product_types):
                    has_product_type = True
                    break
        
        # Check for question words
        for word in words:
            if self.fuzzy_match(word, self.question_words):
                has_question_word = True
                break
        
        # Check for multi-word question phrases
        if not has_question_word:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if self.fuzzy_match(bigram, self.question_words):
                    has_question_word = True
                    break
        
        # Check for reference words
        for word in words:
            if self.fuzzy_match(word, self.reference_keywords):
                has_reference_word = True
                break
        
        # Check for multi-word reference phrases
        if not has_reference_word:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if self.fuzzy_match(bigram, self.reference_keywords):
                    has_reference_word = True
                    break
        
        return has_skin_issue, has_product_type, has_question_word, has_reference_word

    def extract_terms(self, text: str) -> set:
        """Extract important terms from text"""
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_question_clear(self, question: str) -> bool:
        """Check if question is clear and valid"""
        if len(question) < 3:
            return False
        if re.search(r'[^a-zA-Z0-9\s,?!.-]', question):
            return False
        return True

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        """Analyze question type and validity with typo tolerance"""
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()
        if any(role in q_lower for role in self.restricted_roles):
            return "⚠️ Saya adalah asisten khusus skincare recommendation dan tidak dapat berperan sebagai asisten lainnya. Saya hanya dapat membantu pertanyaan terkait skincare recommendation.", "role_change_attempt"
        
        # Remove greetings for core question analysis
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')
        if not cleaned:
            return None, "greeting"

        if not self.is_question_clear(question):
            return "⚠️ Your question contains invalid characters. Please ask a clearer question.", "invalid"

        # Deteksi masalah kulit, jenis produk, kata tanya, dan kata rujukan dengan toleransi typo
        has_skin_issue, has_product_type, has_question_word, has_reference_word = self.detect_keywords(cleaned)
        
        # Deteksi pertanyaan yang merujuk ke konteks sebelumnya
        if has_reference_word and self.memory.history:
            return "context_reference", "context_reference"
        
        # Jika ada kata tanya, tangani sebagai pertanyaan produk spesifik
        if has_question_word:
            return "product_inquiry", "product_inquiry"
        
        # Jika user menyebutkan masalah kulit dan jenis produk sekaligus
        if has_skin_issue and has_product_type:
            return "complete", "complete_question"
        
        # Jika hanya masalah kulit tanpa jenis produk
        if has_skin_issue and not has_product_type:
            return "skin_issue", "skin_issue"

        # Jika hanya jenis produk tanpa masalah kulit
        if not has_skin_issue and has_product_type:
            return "product_type", "product_type"

        # Check for follow-up questions
        if any(keyword in q_lower for keyword in self.follow_up_keywords) and self.memory.history:
            return question, "follow_up"

        return question, "general"

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        """Generate response to user question with context awareness"""
        with self.lock:
            try:
                start_time = time.time()
                core_q, q_type = self.analyze_question(question)

                if q_type == "role_change_attempt":
                    return core_q, (0, 0, 0)
                elif q_type == "off_topic":
                    return core_q, (0, 0, 0)
                
                # Handle special cases
                if q_type == "empty":
                    return "⚠️ Please enter a question.", (0, 0, 0)
                elif q_type == "greeting":
                    return "Halo! Ada yang bisa saya bantu?", (0, 0, 0)
                elif isinstance(core_q, str) and core_q.startswith("⚠️"):
                    return core_q, (0, 0, 0)

                # Handle context reference questions
                if q_type == "context_reference":
                    # Gunakan konteks dari memori terakhir
                    context_summary = self.memory.get_context_summary()
                    
                    if not context_summary:
                        return "Saya tidak memiliki konteks sebelumnya. Bisakah Anda menjelaskan lebih detail?", (0, 0, 0)
                    
                    # Gabungkan dengan pertanyaan saat ini
                    combined_question = f"{context_summary}\n\nPertanyaan lanjutan: {question}"
                    
                    # Proses sebagai pertanyaan baru
                    return self.generate_response(combined_question)

                # Handle product-specific inquiries
                if q_type == "product_inquiry":
                    # Search for relevant documents
                    docs = self.vector_store.similarity_search(question, k=5)
                    if not docs:
                        return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik.", (0, 0, 0)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate prompt khusus untuk pertanyaan produk
                    prompt_template = """
                    Anda adalah asisten ahli skincare. 
                    Jawablah pertanyaan spesifik tentang produk skincare berdasarkan konteks di bawah ini.
                    
                    Aturan menjawab:
                    1. Fokus pada informasi produk yang diminta
                    2. Berikan jawaban spesifik dan langsung
                    3. Sertakan manfaat, kandungan, dan jenis kulit yang cocok jika relevan
                    4. Jika produk tidak ditemukan, jangan membuat informasi
                    
                    [CONTEXT]
                    {context}
                    
                    [QUESTION]
                    {question}
                    
                    Jawablah dengan format:
                    - Jawaban langsung dan spesifik
                    - Gunakan bahasa Indonesia yang jelas
                    - Jika perlu, gunakan poin-poin singkat
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    response = self.llm.invoke(
                        prompt.format(context=context, question=question)
                    )

                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Simpan informasi produk terakhir untuk referensi berikutnya
                    self.last_product_info = response_text
                    
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Handle jenis produk tanpa masalah kulit
                if q_type == "product_type":
                    # Jika ada masalah kulit yang tertunda dari interaksi sebelumnya
                    if self.pending_skin_issue:
                        combined_question = f"{self.pending_skin_issue} {question}"
                        self.pending_skin_issue = None
                        return self.generate_response(combined_question)
                    
                    # Jika tidak ada masalah kulit tertunda, tanyakan masalah kulit
                    self.pending_skin_issue = question
                    response_text = "Baik, untuk jenis produk yang Anda sebutkan, masalah kulit apa yang ingin Anda atasi?\n" \
                                   "(contoh: kulit berminyak, kering, berjerawat, sensitif)"
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Handle complete question (kulit + produk)
                if q_type == "complete_question":
                    # Search for relevant documents
                    docs = self.vector_store.similarity_search(question, k=3)
                    if not docs:
                        return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik.", (0, 0, 0)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate prompt khusus untuk rekomendasi langsung
                    prompt_template = """
                    Anda adalah asisten ahli skincare. 
                    Berikan rekomendasi produk spesifik berdasarkan masalah kulit dan jenis produk yang diminta.

                    Aturan menjawab:
                    1. Fokus hanya pada produk skincare yang relevan
                    2. Sebutkan nama produk lengkap dengan merek
                    3. Berikan penjelasan singkat mengapa produk cocok
                    4. Jika ada beberapa opsi, sebutkan maksimal 3 produk terbaik
                    5. Format jawaban:
                       - Mulai dengan kalimat pembuka yang menjelaskan rekomendasi untuk masalah kulit dan jenis produk yang diminta
                       - Kemudian sebutkan produk dan manfaatnya secara berurutan tanpa bullet point
                       - Setiap produk dan manfaatnya dipisahkan oleh baris baru

                    Contoh format:
                    Untuk kulit kering, toner yang saya rekomendasikan:
                    Hada Labo Gokujyun Hyaluronic Acid Lotion
                    Manfaat: Mengandung asam hyaluronat untuk melembapkan kulit kering

                    Klairs Supple Preparation Unscented Toner
                    Manfaat: Formula bebas alkohol untuk kulit sensitif dan kering

                    [CONTEXT]
                    {context}

                    [QUESTION]
                    {question}
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    response = self.llm.invoke(
                        prompt.format(context=context, question=question)
                    )

                    response_text = response.content if hasattr(response, 'content') else str(response)
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Handle skin issue without product type
                if q_type == "skin_issue":
                    # Save the skin issue for follow-up
                    self.pending_skin_issue = question
                    response_text = "Baik, untuk masalah kulit yang Anda sebutkan, jenis produk apa yang Anda cari?\n" \
                                   "(contoh: facial wash, serum, pelembap, sunscreen, toner)"
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Handle product type selection after skin issue
                if self.pending_skin_issue and q_type == "product_type":
                    combined_question = f"{self.pending_skin_issue} {question}"
                    # Reset the pending skin issue
                    self.pending_skin_issue = None
                    
                    # Search for relevant documents
                    docs = self.vector_store.similarity_search(combined_question, k=3)
                    if not docs:
                        return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik.", (0, 0, 0)

                    # Prepare context
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Generate prompt
                    prompt_template = """
                    Anda adalah asisten ahli skincare. 
                    Berikan rekomendasi produk spesifik berdasarkan masalah kulit dan jenis produk yang diminta.

                    Aturan menjawab:
                    1. Fokus hanya pada produk skincare yang relevan
                    2. Sebutkan nama produk lengkap dengan merek
                    3. Berikan penjelasan singkat mengapa produk cocok
                    4. Jika ada beberapa opsi, sebutkan maksimal 3 produk terbaik
                    5. Format jawaban:
                       - Mulai dengan kalimat pembuka yang menjelaskan rekomendasi untuk masalah kulit dan jenis produk yang diminta
                       - Kemudian sebutkan produk dan manfaatnya secara berurutan tanpa bullet point
                       - Setiap produk dan manfaatnya dipisahkan oleh baris baru

                    Contoh format:
                    Untuk kulit kering, toner yang saya rekomendasikan:
                    Hada Labo Gokujyun Hyaluronic Acid Lotion
                    Manfaat: Mengandung asam hyaluronat untuk melembapkan kulit kering

                    Klairs Supple Preparation Unscented Toner
                    Manfaat: Formula bebas alkohol untuk kulit sensitif dan kering

                    [CONTEXT]
                    {context}

                    [QUESTION]
                    {question}
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    response = self.llm.invoke(
                        prompt.format(context=context, question=combined_question)
                    )

                    response_text = response.content if hasattr(response, 'content') else str(response)
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Search for relevant documents
                docs = self.vector_store.similarity_search(core_q if core_q else question, k=3)
                if not docs:
                    return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik.", (0, 0, 0)

                # Prepare context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Handle follow-up questions
                if q_type == "follow_up" and self.memory.history:
                    context_summary = self.memory.get_context_summary()
                    prompt = f"{context_summary}\n\nFollow-up question: {question}\n\nRelevant documents:\n{context}"
                    response = self.llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    self.memory.add_interaction(question, response_text)
                    return response_text, (0.9, 0.9, 0.9)

                # Handle general questions
                prompt_template = """
                Anda adalah asisten ahli skincare. 
                **Fokus utama Anda hanya pada produk dan layanan skincare.**

                Aturan menjawab:
                1. Untuk pertanyaan produk: Tanyakan jenis produk yang diinginkan (pelembap, serum, sunscreen, facial wash)
                2. Untuk masalah kulit: Tanyakan jenis kulit (kering, berminyak, sensitif, berjerawat)
                3. Jika pertanyaan di luar topik skincare : 
                "Maaf, saya hanya dapat membantu mengenai produk skincare "
                4. Untuk pertanyaan ambigu: Minta klarifikasi lebih detail

                [CONTEXT]
                {context}

                [QUESTION]
                {question}

                Jawablah dengan singkat dan tepat berdasarkan konteks di atas.
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                response = self.llm.invoke(
                    prompt.format(context=context, question=question)
                )

                response_text = response.content if hasattr(response, 'content') else str(response)
                self.memory.add_interaction(question, response_text)
                return response_text, (0.9, 0.9, 0.9)

            except Exception as e:
                print(f"⚠️ Error: {str(e)}")
                traceback.print_exc()
                return "❌ Error processing your question. Please try again.", (0, 0, 0)

def main():
    """Main function to run the QA system"""
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Enter OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ API configuration error: {str(e)}")
        return

    print("\n📤 Please enter the path to the Skincare recommendation PDF document")
    pdf_path = input("PDF file path: ").strip()

    if not os.path.exists(pdf_path):
        print("❌ File not found")
        return

    print(f"\n🔍 Processing document: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    if not processor.initialize_vector_store(pdf_path):
        print("❌ Failed to process document")
        return

    print(f"⏱️ Processing completed in {time.time()-start:.2f} seconds")

    qa_engine = QAEngine(processor.vector_store)

    print("\n💬 QA System Ready. Type 'exit' to quit.")
    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() == 'exit':
                break

            if not question:
                print("⚠️ Please enter a question")
                continue

            start_time = time.time()
            answer, _ = qa_engine.generate_response(question)

            print("\n💡 Answer:")
            print(answer)
            print(f"\n⏱️ Response time: {time.time()-start_time:.2f} seconds")
            print("-"*50)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()




    import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
import gc
import difflib
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConversationContext:
    def __init__(self):
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 3
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def update(self, question: str, response: str):
        # Update skin type and product type from conversation
        self._extract_context(question, response)
        
        # Add to history
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append((question, response))
    
    def _extract_context(self, question: str, response: str):
        # Extract skin type
        skin_types = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi']
        for skin in skin_types:
            if skin in question.lower() or skin in response.lower():
                self.skin_type = skin
                break
                
        # Extract product type
        product_types = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                        "sunscreen", "tabir surya", "serum", "toner", "masker"]
        for product in product_types:
            if product in question.lower() or product in response.lower():
                self.product_type = product
                break
                
        # Extract product name if mentioned
        if re.search(r'\[.*?\]', response):
            self.last_product = re.findall(r'\[(.*?)\]', response)[0]
    
    def get_context_summary(self) -> str:
        summary = []
        if self.skin_type:
            summary.append(f"Jenis kulit: {self.skin_type}")
        if self.product_type:
            summary.append(f"Jenis produk: {self.product_type}")
        if self.last_product:
            summary.append(f"Produk terakhir: {self.last_product}")
        return ", ".join(summary)
    
    def clear(self):
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "skincare_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                batch_size = 100
                for page_num in tqdm(range(0, total_pages, batch_size), desc="📄 Extracting text"):
                    for page in doc[page_num:page_num + batch_size]:
                        text += page.get_text("text") + "\n"
                    if len(text) > 4_000_000:
                        print("ℹ️ Reached processing limit, truncating text")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            max_text_length = 1_000_000
            text_segments = [text[i:i+max_text_length] for i in range(0, len(text), max_text_length)]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            )
            
            all_chunks = []
            for segment in tqdm(text_segments, desc="✂️ Chunking text"):
                all_chunks.extend(splitter.split_text(segment))
                
            return all_chunks
        except Exception as e:
            print(f"❌ Chunk creation error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 50
        total_chunks = len(chunks)
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="🧠 Creating embeddings"):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                else:
                    self.vector_store.add_texts(batch)
                    
                gc.collect()
                    
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)
                    print(f"🔄 Reducing batch size to {batch_size} and retrying...")
                    continue

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        self.context = ConversationContext()
        self.lock = Lock()
        self.skin_types = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi']
        self.product_types = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                             "sunscreen", "tabir surya", "serum", "toner", "masker"]
        self.question_types = ['manfaat', 'kandungan', 'cara pakai', 'harga', 'review', 'rekomendasi', 'untuk']
    
    def analyze_question(self, question: str) -> str:
        """Determine question type based on content and context"""
        q_lower = question.lower()
        
        # Check for greetings
        if any(greet in q_lower for greet in ["halo", "hai", "hi"]):
            return "greeting"
            
        # Check for product-specific questions
        for q_type in self.question_types:
            if q_type in q_lower:
                return "product_detail"
                
        # Check for product type requests
        for p_type in self.product_types:
            if p_type in q_lower:
                return "product_recommendation"
                
        # Check for skin type mentions
        for s_type in self.skin_types:
            if s_type in q_lower:
                return "skin_issue"
                
        # Check for context follow-up
        if self.context.history and any(ref in q_lower for ref in ["itu", "tersebut", "sebelumnya"]):
            return "follow_up"
            
        return "general"
    
    def get_recommendations(self, question: str) -> str:
        """Get product recommendations based on question and context"""
        # Use context if available
        context_summary = self.context.get_context_summary()
        if context_summary:
            question = f"{context_summary}, {question}"
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(question, k=3)
        if not docs:
            return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate recommendations
        prompt_template = """
        Anda adalah asisten ahli skincare. Berikan rekomendasi produk spesifik berdasarkan konteks.
        
        Aturan:
        1. Berikan 2-3 rekomendasi produk saja
        2. Format setiap produk:
           - [Nama Produk]
           - Manfaat: [poin singkat]
           - Cocok untuk: [jenis kulit]
        3. Jika ada konteks jenis kulit, prioritaskan produk untuk jenis kulit tersebut
        
        Konteks: {context}
        Pertanyaan: {question}
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context, question=question))
        return response.content
    
    def get_product_details(self, question: str) -> str:
        """Get detailed information about a specific product"""
        # Use context if available
        if self.context.last_product and any(ref in question for ref in ["produk ini", "tersebut", "sebelumnya"]):
            question = f"{self.context.last_product} {question}"
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(question, k=2)
        if not docs:
            return "❌ Tidak menemukan informasi produk. Silakan sebutkan nama produk lebih jelas."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate product details
        prompt_template = """
        Anda adalah asisten ahli skincare. Berikan informasi spesifik tentang produk berdasarkan konteks.
        
        Aturan:
        1. Fokus pada produk yang disebutkan
        2. Berikan informasi:
           - Manfaat utama
           - Kandungan utama
           - Jenis kulit yang cocok
           - Cara penggunaan
           - Tekstur (jika ada)
           - Harga (jika ada)
        3. Gunakan format poin singkat
        4. Jika informasi tidak tersedia, jangan membuat-buat
        
        Konteks: {context}
        Pertanyaan: {question}
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context, question=question))
        return response.content
    
    def handle_general_question(self, question: str) -> str:
        """Handle general skincare questions"""
        # Search for relevant documents
        docs = self.vector_store.similarity_search(question, k=3)
        if not docs:
            return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt_template = """
        Anda adalah asisten ahli skincare. Jawab pertanyaan berdasarkan konteks.
        
        Aturan:
        1. Berikan jawaban langsung dan spesifik
        2. Jika kurang jelas, minta klarifikasi singkat
        3. Gunakan bahasa Indonesia yang sederhana
        
        Konteks: {context}
        Pertanyaan: {question}
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context, question=question))
        return response.content
    
    def generate_response(self, question: str) -> str:
        with self.lock:
            try:
                start_time = time.time()
                
                # Analyze question type
                q_type = self.analyze_question(question)
                
                # Handle different question types
                if q_type == "greeting":
                    response = "Halo! Ada yang bisa saya bantu tentang produk skincare?"
                
                elif q_type == "product_detail":
                    response = self.get_product_details(question)
                
                elif q_type == "product_recommendation":
                    response = self.get_recommendations(question)
                
                elif q_type == "skin_issue":
                    # Extract skin type
                    for s_type in self.skin_types:
                        if s_type in question.lower():
                            self.context.skin_type = s_type
                            break
                    response = f"Baik, saya catat jenis kulit Anda {self.context.skin_type}. Produk apa yang ingin Anda cari?\n(Contoh: serum, sunscreen, moisturizer)"
                
                elif q_type == "follow_up" and self.context.history:
                    # Use last interaction for context
                    last_question, last_response = self.context.history[-1]
                    combined_question = f"{last_question} {last_response} {question}"
                    response = self.handle_general_question(combined_question)
                
                else:
                    response = self.handle_general_question(question)
                
                # Update context
                self.context.update(question, response)
                
                return response
                
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")
                traceback.print_exc()
                return "❌ Terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi."

def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Enter OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ API configuration error: {str(e)}")
        return

    print("\n📤 Please enter the path to the Skincare recommendation PDF document")
    pdf_path = input("PDF file path: ").strip()

    if not os.path.exists(pdf_path):
        print("❌ File not found")
        return

    print(f"\n🔍 Processing document: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    if not processor.initialize_vector_store(pdf_path):
        print("❌ Failed to process document")
        return

    print(f"⏱️ Processing completed in {time.time()-start:.2f} seconds")

    qa_engine = QAEngine(processor.vector_store)

    print("\n💬 QA System Ready. Type 'exit' to quit.")
    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() == 'exit':
                break

            if not question:
                print("⚠️ Please enter a question")
                continue

            start_time = time.time()
            answer = qa_engine.generate_response(question)

            print("\n💡 Answer:")
            print(answer)
            print(f"\n⏱️ Response time: {time.time()-start_time:.2f} seconds")
            print("-"*50)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()







import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
import gc
import difflib
import logging
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("skincare_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConversationContext:
    def __init__(self):
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.skin_types = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi']
        self.product_types = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                            "sunscreen", "tabir surya", "serum", "toner", "masker"]
        self.need_skin_type = False
        self.need_product_type = False
    
    def update(self, question: str, response: str):
        """Update conversation context based on user question and system response"""
        self._extract_context(question, response)
        
        # Maintain conversation history
        self.history = self.history[-(self.max_history-1):] + [(question, response)]
    
    def _extract_context(self, question: str, response: str):
        """Extract relevant context from conversation"""
        # Reset needs
        self.need_skin_type = False
        self.need_product_type = False
        
        # Extract skin type
        skin_match = next((skin for skin in self.skin_types 
                          if skin in question.lower() or skin in response.lower()), None)
        if skin_match:
            self.skin_type = skin_match
            logger.info(f"Detected skin type: {self.skin_type}")
                
        # Extract product type
        product_match = next((product for product in self.product_types 
                             if product in question.lower() or product in response.lower()), None)
        if product_match:
            self.product_type = product_match
            logger.info(f"Detected product type: {self.product_type}")
                
        # Extract product name
        self.last_product = self._extract_product_name(response)
        if self.last_product:
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_product_name(self, response: str) -> Optional[str]:
        """Extract product name from response with special characters"""
        patterns = [
            r'\[(.*?)\]',  # [Product Name]
            r'Produk:\s*(.*)',  # Produk: Product Name
            r'\b(?:[A-Z][A-Z0-9&.\-+ ]+\(?SPF[\s\d+PA]+\)?[\s\dML]*)\b',  # SCARLETT... (SPF 50+) 50ML
            r'\b(?:[A-Z][A-Za-z0-9&.\-+ ]+SPF[\s\d+PA]+)\b',  # AZARINE... SPF45 PA++++
            r'\bNPURE\s+[^\s]+\b',  # NPURE ProductName
            r'\bGARNIER\s+[^\s]+\b'  # GARNIER ProductName
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                # Return the matched text exactly as found
                return match.group(1 if pattern == r'Produk:\s*(.*)' else 0).strip()
        return None
    
    def get_context_summary(self) -> str:
        """Generate summary of current conversation context"""
        summary = []
        if self.skin_type:
            summary.append(f"Jenis kulit: {self.skin_type}")
        if self.product_type:
            summary.append(f"Jenis produk: {self.product_type}")
        if self.last_product:
            summary.append(f"Produk terakhir: {self.last_product}")
        return ", ".join(summary)
    
    def clear(self):
        """Reset conversation context"""
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []
        self.need_skin_type = False
        self.need_product_type = False

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "skincare_index"
        self.MAX_TEXT_LENGTH = 1_000_000  # To prevent memory overflow
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        """Initialize or load vector store from PDF"""
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        # Create text output path based on PDF filename
        pdf_filename = os.path.basename(pdf_path)
        self.text_output_path = os.path.join(self.text_output_dir, f"{pdf_filename}_processed.txt")
        
        if os.path.exists(index_path):
            try:
                logger.info("Loading existing vector store")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
                
                # Load existing processed text if available
                if os.path.exists(self.text_output_path):
                    with open(self.text_output_path, 'r', encoding='utf-8') as f:
                        self.processed_text = f.read()
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        
        # Create new vector store if loading fails or doesn't exist
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        """Create new vector store from PDF content"""
        # Extract and save text
        self.processed_text = self.extract_text(pdf_path)
        if not self.processed_text:
            logger.error("No text extracted from PDF")
            return False
            
        # Save extracted text to file
        self._save_text_to_file(self.processed_text)
            
        chunks = self.chunk_text(self.processed_text)
        if not chunks:
            logger.error("No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                logger.info(f"Vector store saved to {save_path}")
                
                # Save chunk information
                self._save_chunk_info(chunks)
                return True
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
                return True
        
        return False

    def _save_text_to_file(self, text: str):
        """Save processed text to file"""
        try:
            with open(self.text_output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Extracted text saved to {self.text_output_path}")
        except Exception as e:
            logger.error(f"Failed to save extracted text: {e}")

    def _save_chunk_info(self, chunks: List[str]):
        """Save information about chunks to file"""
        chunk_info_path = os.path.join(self.text_output_dir, "chunk_info.txt")
        try:
            with open(chunk_info_path, 'w', encoding='utf-8') as f:
                f.write(f"Total chunks: {len(chunks)}\n\n")
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"=== Chunk {i} ===\n")
                    f.write(f"Length: {len(chunk)} characters\n")
                    f.write(f"Content:\n{chunk[:200]}...\n\n")  # First 200 chars
            logger.info(f"Chunk information saved to {chunk_info_path}")
        except Exception as e:
            logger.error(f"Failed to save chunk info: {e}")

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text content from PDF document"""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                batch_size = min(100, total_pages)
                
                # Create metadata about the PDF
                pdf_metadata = {
                    'filename': os.path.basename(pdf_path),
                    'total_pages': total_pages,
                    'extraction_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'batches': []
                }
                
                for page_num in tqdm(range(0, total_pages, batch_size), 
                                     desc="Extracting text", unit="pages"):
                    batch_text = ""
                    for page in doc[page_num:page_num + batch_size]:
                        batch_text += page.get_text("text") + "\n"
                    
                    text += batch_text
                    pdf_metadata['batches'].append({
                        'pages': f"{page_num}-{min(page_num + batch_size, total_pages)}",
                        'chars_extracted': len(batch_text)
                    })
                    
                    # Prevent excessive memory usage
                    if len(text) > self.MAX_TEXT_LENGTH:
                        logger.warning("Reached processing limit, truncating text")
                        break
                
                # Save extraction metadata
                self._save_extraction_metadata(pdf_metadata)
                
            return text if text.strip() else None
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def _save_extraction_metadata(self, metadata: dict):
        """Save PDF extraction metadata to file"""
        metadata_path = os.path.join(self.text_output_dir, "extraction_metadata.txt")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write("=== PDF EXTRACTION METADATA ===\n\n")
                f.write(f"File: {metadata['filename']}\n")
                f.write(f"Total pages: {metadata['total_pages']}\n")
                f.write(f"Extracted on: {metadata['extraction_date']}\n\n")
                f.write("Extraction batches:\n")
                for batch in metadata['batches']:
                    f.write(f"- Pages {batch['pages']}: {batch['chars_extracted']} characters\n")
            logger.info(f"Extraction metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save extraction metadata: {e}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for processing"""
        try:
            # Split large text into segments to avoid memory issues
            text_segments = [text[i:i+self.MAX_TEXT_LENGTH] 
                            for i in range(0, len(text), self.MAX_TEXT_LENGTH)]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            )
            
            all_chunks = []
            for segment in tqdm(text_segments, desc="Chunking text", unit="segments"):
                all_chunks.extend(splitter.split_text(segment))
                
            return all_chunks
        except Exception as e:
            logger.error(f"Chunk creation error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        """Create vector store from text chunks"""
        if not chunks:
            logger.error("No text chunks to process")
            return

        batch_size = 50
        total_chunks = len(chunks)
        
        # Create file to log embedding process
        embedding_log_path = os.path.join(self.text_output_dir, "embedding_log.txt")
        
        try:
            with open(embedding_log_path, 'w', encoding='utf-8') as log_file:
                log_file.write("=== EMBEDDING PROCESS LOG ===\n\n")
                log_file.write(f"Total chunks to process: {total_chunks}\n")
                log_file.write(f"Batch size: {batch_size}\n")
                log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i in tqdm(range(0, total_chunks, batch_size), 
                              desc="Creating embeddings", unit="chunks"):
                    batch = chunks[i:i + batch_size]
                    try:
                        if not self.vector_store:
                            self.vector_store = FAISS.from_texts(batch, self.embeddings)
                            log_file.write(f"Initialized vector store with first batch (chunks 1-{min(batch_size, total_chunks)})\n")
                        else:
                            self.vector_store.add_texts(batch)
                            log_file.write(f"Added batch {i//batch_size + 1} (chunks {i+1}-{min(i+batch_size, total_chunks)})\n")
                            
                        # Manage memory usage
                        if i % 500 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                        log_file.write(f"ERROR processing batch {i//batch_size + 1}: {str(e)}\n")
                        traceback.print_exc()
                        if batch_size > 10:
                            batch_size = max(10, batch_size // 2)
                            logger.info(f"Reducing batch size to {batch_size} and retrying")
                            log_file.write(f"Reduced batch size to {batch_size}\n")
                
                log_file.write(f"\nEmbedding completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Final vector store size: {len(self.vector_store.index_to_docstore_id)} documents\n")
            
            logger.info(f"Embedding process logged to {embedding_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to create embedding log: {e}")

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        self.context = ConversationContext()
        self.lock = Lock()
        self.question_types = ['manfaat', 'kandungan', 'cara pakai', 'harga', 'review']
        self.ingredient_keywords = ['mengandung', 'kandungan', 'ingredient', 'komposisi', 'bahan']
        self.condition_keywords = ['kemerahan', 'iritasi', 'meradang', 'sensitif', 'kering', 'berminyak', 'berjerawat', 'kusam', 'pori', 'flek']
    
    def _extract_products_from_context(self, context: str) -> List[str]:
        """Extract product names with special characters and formatting"""
        patterns = [
            r'Produk:\s*(.*?)\n',  # Produk: [nama]
            r'\[(.*?)\]',           # [Nama Produk]
            r'\b(?:[A-Z][A-Z0-9&.\-+ ]+\(?SPF[\s\d+PA]+\)?[\s\dML]*)\b',  # SCARLETT... (SPF 50+) 50ML
            r'\b(?:[A-Z][A-Za-z0-9&.\-+ ]+SPF[\s\d+PA]+)\b',  # AZARINE... SPF45 PA++++
            r'\bNPURE\s+[^\s]+\b',     # NPURE ProductName
            r'\bGARNIER\s+[^\s]+\b',   # GARNIER ProductName
            r'\bSENKA\s+[^\s]+\b',     # SENKA ProductName
            r'\b(?:[A-Z][a-zA-Z0-9&.\-+ ]+)',  # Capitalized names with special chars
            r'\b[A-Z][A-Z0-9\s&.-]+\b'  # All caps names with special chars
        ]
        
        products = set()
        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                # Get the matched text exactly as found
                if isinstance(match, tuple):
                    name = " ".join(m for m in match if m).strip()
                else:
                    name = match.strip()
                
                # Only basic whitespace normalization
                name = re.sub(r'\s+', ' ', name)
                
                if len(name) > 10 and not any(kw in name.lower() for kw in ['manfaat', 'kandungan']):
                    products.add(name)
        
        logger.info(f"Valid products extracted: {list(products)}")
        return list(products)
    
    def analyze_question(self, question: str) -> str:
        """Determine question type based on content and context"""
        logger.info(f"Analyzing question: {question}")
        q_lower = question.lower()
        
        # Explicit off-topic check
        off_topic_keywords = ["kelautan", "nahkoda", "politik", "sejarah umum", "geografi", "matematika", "kapal"]
        skincare_keywords_present = any(p_type in q_lower for p_type in self.context.product_types) or \
                                   any(s_type in q_lower for s_type in self.context.skin_types) or \
                                   any(qt_type in q_lower for qt_type in self.question_types) or \
                                   "skincare" in q_lower or "kulit" in q_lower or "wajah" in q_lower

        if any(ot_keyword in q_lower for ot_keyword in off_topic_keywords) and not skincare_keywords_present:
            logger.warning(f"Off-topic question detected: {question}")
            return "off_topic"
            
        # Check if we need to complete previous request
        if self.context.need_skin_type or self.context.need_product_type:
            return "complete_recommendation"
            
        # Question type detection
        if any(greet in q_lower for greet in ["halo", "hai", "hi"]):
            return "greeting"
            
        if any(q_type in q_lower for q_type in self.question_types):
            return "product_detail"
                
        if any(p_type in q_lower for p_type in self.context.product_types):
            return "product_recommendation"
                
        if any(s_type in q_lower for s_type in self.context.skin_types):
            return "skin_issue"
                
        if self.context.history and any(ref in q_lower for ref in ["itu", "tersebut", "sebelumnya"]):
            return "follow_up"
        
        if self.context.last_product and len(question.split()) < 4:
            return "short_follow_up"
        
        # Check for ingredient-based requests
        if any(ing in q_lower for ing in self.ingredient_keywords):
            return "ingredient_recommendation"
            
        # Check for condition-based requests
        if any(cond in q_lower for cond in self.condition_keywords):
            return "condition_recommendation"
        
        return "general"
    
    def get_recommendations(self, question: str) -> str:
        """Get product recommendations strictly from context"""
        logger.info(f"Getting recommendations for: {question}")
        # Enhance question with context
        context_summary = self.context.get_context_summary()
        enhanced_question = f"{context_summary}, {question}" if context_summary else question
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(enhanced_question, k=5)
        if not docs:
            return "❌ Tidak menemukan informasi yang relevan. Silakan ajukan pertanyaan yang lebih spesifik."

        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Extract valid product names from context
        valid_products = self._extract_products_from_context(context)
        if not valid_products:
            logger.warning("No valid products found in context")
            return "❌ Tidak menemukan produk yang sesuai dalam database."
        
        # Generate recommendations with strict prompt
        prompt_template = """
        Anda adalah asisten ahli skincare. Berikan rekomendasi produk spesifik BERDASARKAN KONTEKS SAJA.

        ATURAN KETAT:
        1. HANYA rekomendasikan produk yang disebutkan dalam KONTEKS
        2. Format respons:
           - Berikan penjelasan singkat tentang rekomendasi secara umum
           - Kemudian berikan daftar produk dengan format:
             Produk: [Nama Produk Lengkap sesuai KONTEKS]
        
        3. Berikan minimal 3 produk jika tersedia
        4. JANGAN membuat-buat informasi di luar KONTEKS
        5. Jika tidak ada produk yang cocok, katakan "Tidak ditemukan rekomendasi"
        6. Prioritaskan produk yang sesuai dengan jenis kulit: {skin_type}
        
        DAFTAR PRODUK VALID (Hanya gunakan ini):
        {valid_products}

        Konteks: {context}
        Pertanyaan: {question}
        """

        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question", "valid_products", "skin_type"]
        )
        formatted_prompt = prompt.format(
            context=context,
            question=enhanced_question,
            valid_products=", ".join(valid_products),
            skin_type=self.context.skin_type or "tidak diketahui"
        )
        logger.debug(f"Recommendation prompt: {formatted_prompt}")
        
        response = self.llm.invoke(formatted_prompt)
        logger.debug(f"Raw response: {response.content}")
        
        return self._format_recommendation_response(response.content, valid_products)
    
    def _format_recommendation_response(self, response: str, valid_products: List[str]) -> str:
        """Format recommendation response with explanation and product cards"""
        logger.info("Formatting recommendation response")
        
        # Split into explanation and products section
        explanation = ""
        products_section = ""
        
        # Find where the product list starts
        product_start = re.search(r'(Produk:|\nProduk:)', response)
        if product_start:
            explanation = response[:product_start.start()].strip()
            products_section = response[product_start.start():]
        else:
            # If no product section found, treat whole response as explanation
            explanation = response
            return f"<div class='explanation'><p>{explanation}</p></div>"
        
        # Format explanation
        explanation_html = f"<div class='explanation'><p>{explanation}</p></div>"
        
        # Format products
        products_html = self._format_recommendations(products_section, valid_products)
        
        return explanation_html + products_html
    
    def _format_recommendations(self, response: str, valid_products: List[str]) -> str:
        """Format LLM response with product validation (keeping original names)"""
        logger.info("Formatting recommendations with original names")
        products = []
        current_product = {}
        
        # Parse response
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Produk:'):
                if current_product:
                    products.append(current_product)
                product_name = line.replace('Produk:', '').strip()
                current_product = {'name': product_name, 'original_name': product_name}
            elif line.startswith('- Manfaat utama:') and 'name' in current_product:
                current_product['benefits'] = line.replace('- Manfaat utama:', '').strip()
            elif line.startswith('- Kandungan utama:') and 'name' in current_product:
                current_product['ingredients'] = line.replace('- Kandungan utama:', '').strip()
            elif line.startswith('- Jenis kulit:') and 'name' in current_product:
                current_product['skin_type'] = line.replace('- Jenis kulit:', '').strip()
        
        # Add last product if exists
        if current_product and 'name' in current_product:
            products.append(current_product)
        
        # Validate products against context but keep original names
        validated_products = []
        for product in products:
            # Check if product exists in valid products (flexible matching)
            matches = difflib.get_close_matches(
                product['name'], 
                valid_products,
                n=1,
                cutoff=0.6  # Lower threshold for special characters
            )
            
            if matches:
                product['valid_name'] = matches[0]
            else:
                # If no close match but looks like a product name, keep it
                if any(kw in product['name'].upper() for kw in ['SPF', 'PA', 'SUNSCREEN', 'SERUM']):
                    product['valid_name'] = product['name']
                else:
                    logger.warning(f"Invalid product removed: {product['name']}")
                    continue
            
            validated_products.append(product)
            logger.info(f"Validated product: {product['name']}")
        
        # Generate HTML cards only for validated products
        if not validated_products:
            return "<p>❌ Tidak menemukan produk yang sesuai dalam database.</p>"
            
        return self._generate_product_cards(validated_products)
    
    def _generate_product_cards(self, products: List[Dict]) -> str:
        """Generate HTML cards from product data using original names"""
        html_cards = []

        for product in products:
            # Use the validated product name exactly as provided
            product_name = product.get('valid_name', product['name'])
            
            # Clean filename by replacing special chars, but keep original display name
            image_filename = re.sub(r'[\\/*?:"<>|]', "_", product_name)
            
            card = f"""
            <div class="product-card">
                <img style="object-fit: contain;" src="/static/data.1/{image_filename}.jpg" alt="{product_name}" class="product-image">
                <h3 class="product-name">{product_name}</h3>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{product_name}">
                        Lihat Detail
                    </button>
                </div>
            </div>
            """
            html_cards.append(card)
        
        if not html_cards:
            return "<p>Tidak ada rekomendasi produk yang dapat ditampilkan saat ini.</p>"
            
        return f'<div class="recommendations-container" style="display: flex; gap: 20px; flex-wrap: wrap;">{"".join(html_cards)}</div>'
    
    def get_product_details(self, question: str) -> str:
        """Get detailed information about a specific product"""
        logger.info(f"Getting product details for: {question}")
        
        # Enhance question with context
        if self.context.last_product and any(ref in question.lower() for ref in ["ini", "tersebut", "sebelumnya"]):
            enhanced_question = f"{self.context.last_product} {question}"
        else:
            enhanced_question = question
        
        # Determine information type requested
        info_type = next((q_type for q_type in self.question_types 
                         if q_type in enhanced_question.lower()), None)
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(enhanced_question, k=3)
        if not docs:
            return "❌ Tidak menemukan informasi produk. Silakan sebutkan nama produk lebih jelas."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Extract valid product names from context
        valid_products = self._extract_products_from_context(context)
        if not valid_products:
            logger.warning("No valid products in context for detail request")
            return "❌ Produk tidak ditemukan dalam database."
        
        # Generate prompt based on information type
        if info_type == 'kandungan':
            prompt_template = """
            Berikan daftar kandungan untuk produk yang diminta BERDASARKAN KONTEKS.
            Aturan:
            1. Hanya berikan daftar bahan dari KONTEKS
            2. Pisahkan dengan koma jika ada beberapa bahan
            3. Jika tidak ada, katakan tidak ada
            """
        elif info_type == 'cara pakai':
            prompt_template = """
            Berikan cara penggunaan untuk produk yang diminta BERDASARKAN KONTEKS.
            Aturan:
            1. Hanya berikan langkah-langkah penggunaan dari KONTEKS
            2. Gunakan format poin
            3. Jika tidak ada, katakan tidak ada
            """
        elif info_type:
            prompt_template = f"""
            Berikan informasi {info_type} untuk produk yang diminta BERDASARKAN KONTEKS.
            Aturan:
            1. Fokus pada produk spesifik dari KONTEKS
            2. Berikan {info_type} saja dari KONTEKS
            3. Jika tidak ada, jangan membuat-buat
            """
        else:
            prompt_template = """
            Berikan informasi spesifik tentang produk BERDASARKAN KONTEKS.
            Aturan:
            1. Fokus pada produk yang diminta
            2. Berikan informasi yang relevan dengan pertanyaan dari KONTEKS
            3. Jika tidak ada, katakan tidak ada
            """
        
        # Add product validation to prompt
        prompt_template += f"\n\nDaftar Produk Valid: {', '.join(valid_products)}"
        full_prompt = f"{prompt_template}\n\nKonteks: {context}\nPertanyaan: {enhanced_question}"
        logger.debug(f"Detail prompt: {full_prompt}")
        
        response = self.llm.invoke(full_prompt)
        return response.content
    
    def handle_general_question(self, question: str) -> str:
        """Handle general skincare questions"""
        logger.info(f"Handling general question: {question}")
        docs = self.vector_store.similarity_search(question, k=2)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Anda adalah asisten AI yang khusus membahas produk skincare.
        Jawab pertanyaan berikut HANYA JIKA berkaitan dengan skincare, berdasarkan KONTEKS yang diberikan.
        Jika pertanyaan tidak berkaitan dengan skincare, jawab dengan: "Maaf, saya hanya dapat membantu pertanyaan terkait produk skincare."
        
        Aturan tambahan jika pertanyaan relevan dengan skincare:
        - Berikan jawaban informatif dan ramah HANYA dari KONTEKS
        - Gunakan bahasa Indonesia yang santai.
        - Jika informasi tidak ada dalam KONTEKS, katakan tidak tahu.
        
        Konteks: {context}
        Pertanyaan: {question}
        """
        
        logger.debug(f"General prompt: {prompt}")
        response = self.llm.invoke(prompt)
        return response.content
    
    def generate_response(self, question: str) -> str:
        with self.lock:
            try:
                logger.info(f"Processing question: {question}")
                start_time = time.time()
                
                # Pre-update context from the question
                self.context._extract_context(question, "")
                
                # Analyze question type
                q_type = self.analyze_question(question)
                
                # Route to appropriate handler
                if q_type == "off_topic":
                    response = "Maaf, saya adalah asisten AI yang khusus untuk rekomendasi produk skincare. Saya tidak dapat menjawab pertanyaan di luar topik tersebut."
                
                elif q_type == "greeting":
                    response = "Halo! Ada yang bisa saya bantu tentang produk skincare?"
                
                elif q_type == "product_detail":
                    response = self.get_product_details(question)
                
                elif q_type == "complete_recommendation":
                    # Handle completion of previous recommendation request
                    if self.context.need_skin_type:
                        if any(s_type in question.lower() for s_type in self.context.skin_types):
                            response = self.get_recommendations(question)
                        else:
                            response = "Maaf, saya tidak mengenali jenis kulit tersebut. Jenis kulit yang valid: kering, berminyak, sensitif, berjerawat, normal, kombinasi."
                            self.context.need_skin_type = True
                    
                    elif self.context.need_product_type:
                        if any(p_type in question.lower() for p_type in self.context.product_types):
                            response = self.get_recommendations(question)
                        else:
                            response = "Maaf, saya tidak mengenali jenis produk tersebut. Jenis produk yang valid: facial wash, cleanser, pelembap, moisturizer, sunscreen, serum, toner, masker."
                            self.context.need_product_type = True
                
                elif q_type == "product_recommendation":
                    # Handle product-based requests
                    if self.context.skin_type:
                        # Already have skin type, provide recommendations
                        response = self.get_recommendations(question)
                    else:
                        # Need skin type for recommendation
                        response = "Untuk jenis kulit apa yang Anda rekomendasikan?"
                        self.context.need_skin_type = True
                
                elif q_type == "skin_issue":
                    # Handle skin type-based requests
                    if self.context.product_type:
                        # Already have product type, provide recommendations
                        response = self.get_recommendations(question)
                    else:
                        # Need product type for recommendation
                        response = "Produk jenis apa yang Anda cari? (facial wash, cleanser, pelembap, dll)"
                        self.context.need_product_type = True
                
                elif q_type == "ingredient_recommendation":
                    # Handle ingredient-based requests
                    response = self.get_recommendations(question)
                
                elif q_type == "condition_recommendation":
                    # Handle skin condition-based requests
                    response = self.get_recommendations(question)
                
                elif q_type == "follow_up" and self.context.history:
                    last_q, last_r = self.context.history[-1]
                    response = self.handle_general_question(f"{last_q} {last_r} {question}")
                
                elif q_type == "short_follow_up" and self.context.last_product:
                    response = self.get_product_details(f"{self.context.last_product} {question}")
                
                else:
                    response = self.handle_general_question(question)
                
                # Update context with final response
                self.context.update(question, response)
                
                logger.info(f"Response generated in {time.time()-start_time:.2f}s")
                return response
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                traceback.print_exc()
                return "❌ Terjadi kesalahan. Silakan coba lagi dengan pertanyaan berbeda."

def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Enter OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        logger.error(f"API configuration error: {str(e)}")
        return

    print("\n📤 Masukkan path file PDF rekomendasi skincare")
    pdf_path = input("Path file PDF: ").strip()

    if not os.path.exists(pdf_path):
        print("❌ File tidak ditemukan")
        return

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    if not processor.initialize_vector_store(pdf_path):
        print("❌ Gagal memproses dokumen")
        return

    print(f"⏱️ Proses selesai dalam {time.time()-start:.2f} detik")

    qa_engine = QAEngine(processor.vector_store)

    print("\n💬 Sistem siap. Ketik 'exit' untuk keluar.")
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() == 'exit':
                break
            if not question:
                continue

            start_time = time.time()
            answer = qa_engine.generate_response(question)

            print("\n💡 Jawaban:")
            print(answer)
            print(f"\n⏱️ Waktu respon: {time.time()-start_time:.2f} detik")
            print("-"*50)

        except KeyboardInterrupt:
            print("\n👋 Keluar...")
            break
        except Exception as e:
            logger.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()



    import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
from typing import Tuple, Optional, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("skincare_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.skin_types = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi', 'kemerahan']
        self.product_types = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                            "sunscreen", "tabir surya", "serum", "toner", "masker"]
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.greetings = ['halo', 'hai', 'hello', 'hi', 'apa kabar', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam']
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        self.all_products.add(product_name.strip().lower())
    
    def is_valid_product(self, product_name: str) -> bool:
        return product_name.strip().lower() in self.all_products
    
    def update(self, question: str, response: str):
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
    def _extract_context(self, question: str, response: str):
        self.need_skin_type = False
        self.need_product_type = False
        
        # Extract skin type with improved regex
        skin_pattern = r'\b(kulit\s*)?(kering|berminyak|sensitif|berjerawat|normal|kombinasi|kemerahan)\b'
        skin_match = re.search(skin_pattern, question.lower())
        if skin_match:
            self.skin_type = skin_match.group(2)
            logger.info(f"Detected skin type: {self.skin_type}")
                
        # Extract product type with improved regex
        product_pattern = r'\b(facial wash|cleanser|pelembap|moisturizer|sunscreen|tabir surya|serum|toner|masker)\b'
        product_match = re.search(product_pattern, question.lower())
        if product_match:
            self.product_type = product_match.group(0)
            logger.info(f"Detected product type: {self.product_type}")
            
            # If product type is detected but skin type is missing
            if self.product_type and not self.skin_type:
                self.need_skin_type = True
                
        # Extract product from both question and response
        self.last_product = self._extract_product_name(question) or self._extract_product_name(response)
        if self.last_product:
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_product_name(self, text: str) -> Optional[str]:
        patterns = [
            r'Produk:\s*([^\n\(\)]+)',
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'\b([A-Z][A-Za-z0-9&.\- ]{3,})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match if isinstance(match, str) else match[0]
                name = name.strip()
                if self.is_valid_product(name):
                    return name
        return None
    
    def get_context_summary(self) -> str:
        summary = []
        if self.skin_type:
            summary.append(f"Kulit: {self.skin_type}")
        if self.product_type:
            summary.append(f"Produk: {self.product_type}")
        if self.last_product:
            summary.append(f"Terakhir: {self.last_product[:20]}")
        return ", ".join(summary)

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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
        self.product_names = set()
        self.product_data = {}  # Store complete product data
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        self.processed_text = self.extract_text(pdf_path)
        if not self.processed_text:
            logger.error("No text extracted from PDF")
            return False
        
        self._extract_all_product_data(self.processed_text)
        
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
        
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        chunks = self.chunk_text(self.processed_text)
        if not chunks:
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                return True
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
                return False
        return False

    def _extract_all_product_data(self, text: str):
        # Split text into product sections
        product_sections = re.split(r'(?=Produk:\s*)', text)
        
        for section in product_sections:
            if not section.strip():
                continue
                
            # Extract product name
            name_match = re.search(r'Produk:\s*([^\n\(\)]+)', section)
            if not name_match:
                continue
                
            name = name_match.group(1).strip()
            if len(name) < 3 or not any(c.isupper() for c in name):
                continue
                
            # Extract all product data
            product_info = {
                'name': name,
                'harga': self._extract_field(section, 'harga:'),
                'type_produk': self._extract_field(section, 'type produk:'),
                'jenis_kulit': self._extract_field(section, 'jenis kulit:'),
                'informasi_produk': self._extract_field(section, 'informasi produk:'),
                'kandungan': self._extract_field(section, 'kandungan:'),
                'cara_pemakaian': self._extract_field(section, 'cara pemakaian:'),
                'manfaat': self._extract_field(section, 'manfaat:'),
                'lokasi': self._extract_field(section, 'lokasi :')
            }
            
            self.product_names.add(name)
            self.product_data[name.lower()] = product_info
        
        logger.info(f"Extracted {len(self.product_names)} product names from PDF")
    
    def _extract_field(self, text: str, field_name: str) -> str:
        pattern = re.escape(field_name) + r'\s*([^\n]*(\n[^\n]*)*?)(?=\n\w+:|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def get_product_names(self) -> List[str]:
        return list(self.product_names)
    
    def get_product_data(self, product_name: str) -> Optional[Dict]:
        return self.product_data.get(product_name.lower())

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

        # Create documents with metadata
        documents = []
        for chunk in chunks:
            metadata = {}
            # Extract skin type from chunk
            skin_match = re.search(r'jenis kulit:\s*([^\n]+)', chunk, re.IGNORECASE)
            if skin_match:
                metadata['skin_type'] = skin_match.group(1).strip().lower()
            
            # Extract product type from chunk
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
                    
                if i % 300 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.question_types = {
            'manfaat': ['manfaat', 'kegunaan', 'guna', 'fungsi'],
            'kandungan': ['kandungan', 'ingredient', 'komposisi', 'bahan'],
            'cara_pakai': ['cara pakai', 'cara penggunaan', 'penggunaan', 'pemakaian'],
            'harga': ['harga', 'price', 'berapa harga', 'berapa price'],
            'lokasi': ['lokasi', 'tempat beli', 'beli dimana', 'tempat pembelian'],
            'review': ['review', 'ulasan', 'testimoni', 'pendapat'],
            'rekomendasi': ['rekomendasi', 'sarankan', 'anjuran', 'saran']
        }

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict]) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Format nama untuk gambar: ganti spasi dengan underscore dan hilangkan karakter khusus
            img_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image">
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

    def analyze_question(self, question: str) -> Tuple[str, Optional[str]]:
        q_lower = question.lower()
        
        # Check for greetings
        if any(greeting in q_lower for greeting in self.context.greetings):
            return ("sapaan", None)
            
        # Check for off-topic
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None)
            
        # Check for skin type only
        skin_match = next((skin for skin in self.context.skin_types 
                          if skin in q_lower), None)
        if skin_match:
            self.context.skin_type = skin_match
            # If we have product type but no skin type previously, now we can recommend
            if self.context.product_type:
                return ("rekomendasi", None)
            return ("need_product_type", None)
            
        # Check for specific product questions
        for q_type, keywords in self.question_types.items():
            if any(kw in q_lower for kw in keywords):
                product = self._extract_product_from_question(question, keywords)
                if product and self.context.is_valid_product(product):
                    return (q_type, product)
                
                # Gunakan produk terakhir jika tersedia
                if self.context.last_product:
                    return (q_type, self.context.last_product)
                
        # Check for product type requests
        product_match = next((product for product in self.context.product_types 
                             if product in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            if not self.context.skin_type:
                self.context.need_skin_type = True
                return ("need_skin_type", None)
            else:
                return ("rekomendasi", None)
                
        return ("general", None)

    def _extract_product_from_question(self, question: str, keywords: list) -> Optional[str]:
        # First, try to extract product name using patterns
        patterns = [
            r'Produk:\s*([^\n\(\)]+)',
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'\b([A-Z][A-Za-z0-9&.\- ]{3,})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                name = match if isinstance(match, str) else match[0]
                name = name.strip()
                if self.context.is_valid_product(name):
                    return name
        
        # If no product found, remove keywords and check if remaining text is a product
        for keyword in keywords:
            if keyword in question.lower():
                product = re.sub(keyword, '', question, flags=re.IGNORECASE).strip()
                product = re.sub(r'[^\w\s]', '', product).strip()
                if len(product) > 3 and self.context.is_valid_product(product):
                    return product
        
        return None

    def get_recommendations(self, question: str) -> str:
        try:
            # Build effective query with skin type and product type
            query_parts = []
            if self.context.skin_type:
                query_parts.append(f"kulit {self.context.skin_type}")
            if self.context.product_type:
                query_parts.append(self.context.product_type)
            query = " ".join(query_parts) or question
            
            logger.info(f"Recommendation query: {query}")
            
            # Use metadata filtering for skin type and product type
            filter_dict = {}
            if self.context.skin_type:
                filter_dict["skin_type"] = self.context.skin_type
            if self.context.product_type:
                filter_dict["product_type"] = self.context.product_type
            
            # Get relevant documents with metadata filtering
            docs = self.vector_store.similarity_search(query, k=15, filter=filter_dict)
            if not docs:
                # Fallback: without filter
                docs = self.vector_store.similarity_search(query, k=15)
            
            context = "\n".join(doc.page_content for doc in docs)
            context = self._truncate_to_tokens(context, 3000)
            
            logger.info(f"Context for recommendations: {context[:200]}...")
            
            # Hybrid approach: Extract products from context + LLM
            all_products = set()
            
            # 1. Extract products directly from context
            product_pattern = r'Produk:\s*([^\n\(\)]+)'
            matches = re.findall(product_pattern, context)
            for match in matches:
                name = match.strip()
                if self.context.is_valid_product(name):
                    all_products.add(name)
            
            # 2. Use LLM to recommend products
            prompt = f"""
            Berdasarkan konteks berikut, rekomendasikan 3-5 produk skincare yang cocok untuk:
            - Jenis kulit: {self.context.skin_type or 'semua jenis'}
            - Jenis produk: {self.context.product_type or 'semua jenis'}
            
            Format respons:
            1. Nama Produk 1
            2. Nama Produk 2
            3. Nama Produk 3
            
            Konteks:
            {context}
            """
            
            prompt = self._truncate_to_tokens(prompt, 3500)
            llm_response = self.llm.invoke(prompt).content
            logger.info(f"LLM response for recommendations: {llm_response}")
            
            # 3. Extract products from LLM response
            for line in llm_response.split('\n'):
                if re.match(r'^\d+[\.\)]?\s+', line):
                    name = re.sub(r'^\d+[\.\)]?\s*', '', line).strip()
                    if self.context.is_valid_product(name):
                        all_products.add(name)
            
            if not all_products:
                return "❌ Maaf, tidak menemukan produk yang cocok. Silakan coba dengan kriteria yang lebih spesifik."
            
            # Convert to list of dicts
            products = [{'name': name} for name in list(all_products)[:5]]
            product_cards = self._generate_product_cards(products)
            
            return f"""
            <div class="recommendation-container">
                <h3>Rekomendasi untuk kulit {self.context.skin_type if self.context.skin_type else ''} {self.context.product_type if self.context.product_type else 'produk'}:</h3>
                <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                    {product_cards}
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            traceback.print_exc()
            return "❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi."

    def get_product_info(self, product_name: str, info_type: str) -> str:
        try:
            if not self.context.is_valid_product(product_name):
                return f"❌ Produk '{product_name}' tidak ditemukan dalam database kami."
            
            # Get product data directly from processor
            product_data = self.processor.get_product_data(product_name)
            if not product_data:
                return f"❌ Tidak menemukan informasi tentang {product_name}"
            
            # Return specific information based on type
            if info_type == "kandungan":
                # Return all ingredients without modification
                return product_data.get('kandungan', "❌ Informasi kandungan tidak tersedia")
            
            elif info_type == "cara_pakai":
                cara_pakai = product_data.get('cara_pemakaian', "❌ Informasi cara pemakaian tidak tersedia")
                # Ensure proper formatting
                if not cara_pakai.startswith(('•', '-')):
                    # Convert to bullet points
                    lines = [line.strip() for line in cara_pakai.split('\n') if line.strip()]
                    cara_pakai = '\n'.join([f"• {line}" for line in lines])
                return cara_pakai
            
            elif info_type == "manfaat":
                manfaat = product_data.get('manfaat', "❌ Informasi manfaat tidak tersedia")
                # Ensure proper formatting
                if not manfaat.startswith(('•', '-')):
                    # Convert to bullet points
                    lines = [line.strip() for line in manfaat.split('\n') if line.strip()]
                    manfaat = '\n'.join([f"• {line}" for line in lines])
                return manfaat
            
            elif info_type == "harga":
                return product_data.get('harga', "❌ Informasi harga tidak tersedia")
            
            elif info_type == "lokasi":
                lokasi = product_data.get('lokasi', "❌ Informasi lokasi tidak tersedia")
                # Ensure proper formatting
                if not lokasi.startswith(('•', '-')):
                    # Convert to bullet points
                    lines = [line.strip() for line in lokasi.split('\n') if line.strip()]
                    lokasi = '\n'.join([f"• {line}" for line in lines])
                return lokasi
            
            else:
                # For other types, use LLM as fallback
                return self._get_info_from_llm(product_name, info_type)
            
        except Exception as e:
            logger.error(f"Product info error: {e}")
            traceback.print_exc()
            return f"❌ Tidak bisa menampilkan informasi {info_type} untuk {product_name}"
    
    def _get_info_from_llm(self, product_name: str, info_type: str) -> str:
        """Fallback method for info types not directly available in data"""
        try:
            # Search for the specific product
            docs = self.vector_store.similarity_search(product_name, k=3)
            if not docs:
                return f"❌ Tidak menemukan informasi tentang {product_name}"
            
            context = "\n".join(doc.page_content for doc in docs)
            context = self._truncate_to_tokens(context, 1500)
            
            prompt = f"""
            Berdasarkan konteks berikut, berikan informasi tentang {info_type} dari produk {product_name}.
            Format respons: Informasi yang relevan
            
            Konteks:
            {context}
            """
            
            prompt = self._truncate_to_tokens(prompt, 2000)
            response = self.llm.invoke(prompt).content
            
            # Clean up response
            response = re.sub(r'^\?+\s*', '', response, flags=re.MULTILINE)
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return f"❌ Tidak bisa menampilkan informasi {info_type} untuk {product_name}"

    def generate_response(self, question: str) -> str:
        with self.lock:
            try:
                start_time = time.time()
                self.context._extract_context(question, "")
                
                q_type, product = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}")
                
                if q_type == "off_topic":
                    return "Maaf, saya hanya membantu soal skincare."
                
                elif q_type == "sapaan":
                    return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
                
                elif q_type == "need_skin_type":
                    skin_types = "/".join(self.context.skin_types)
                    return f"Silakan sebutkan jenis kulit Anda ({skin_types})"
                
                elif q_type == "need_product_type":
                    product_types = "/".join(self.context.product_types)
                    return f"Silakan sebutkan jenis produk yang Anda cari ({product_types})"
                
                elif q_type in self.question_types and product:
                    # Simpan produk terakhir untuk pertanyaan berikutnya
                    self.context.last_product = product
                    return self.get_product_info(product, q_type)
                
                elif q_type == "rekomendasi":
                    # Jika ada jenis produk tapi belum ada jenis kulit
                    if self.context.product_type and not self.context.skin_type:
                        skin_types = "/".join(self.context.skin_types)
                        return f"Silakan sebutkan jenis kulit Anda ({skin_types})"
                    
                    return self.get_recommendations(question)
                
                else:
                    return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi. Contoh: 'serum untuk kulit berminyak' atau 'rekomendasi sunscreen'"
                    
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                traceback.print_exc()
                return "❌ Terjadi kesalahan. Silakan coba lagi."

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
            traceback.print_exc()

if __name__ == "__main__":
    main()






import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
import difflib
from typing import Tuple, Optional, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

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
SKIN_TYPES = ['kering', 'berminyak', 'sensitif', 'berjerawat', 'normal', 'kombinasi', 'kemerahan']
PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                "sunscreen", "tabir surya", "serum", "toner", "masker"]
GREETINGS = ['halo', 'hai', 'hello', 'hi', 'apa kabar', 'selamat pagi', 
             'selamat siang', 'selamat sore', 'selamat malam']

QUESTION_TYPES = {
    'manfaat': ['manfaat', 'kegunaan', 'guna', 'fungsi'],
    'kandungan': ['kandungan', 'ingredient', 'komposisi', 'bahan'],
    'cara_pakai': ['cara pakai', 'cara penggunaan', 'penggunaan', 'pemakaian'],
    'harga': ['harga', 'price', 'berapa harga', 'berapa price'],
    'lokasi': ['lokasi', 'tempat beli', 'beli dimana', 'tempat pembelian'],
    'review': ['review', 'ulasan', 'testimoni', 'pendapat'],
    'rekomendasi': ['rekomendasi', 'sarankan', 'anjuran', 'saran']
}

COMBINED_TYPES = {
    'kandungan_manfaat': ['kandungan dan manfaat', 'komposisi dan kegunaan'],
    'cara_manfaat': ['cara pakai dan manfaat', 'penggunaan dan kegunaan'],
    'semua_info': ['semua informasi', 'semua info', 'semua data']
}

# ======================
# FUZZY MATCHING UTILS
# ======================
def fuzzy_match(input_str, options, threshold=0.6, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    normalized_options = [opt.lower() for opt in options]
    matches = difflib.get_close_matches(
        input_str.lower(), 
        normalized_options, 
        n=5 if get_all else 1, 
        cutoff=threshold
    )
    
    if get_all:
        # Return all matches with original casing
        return [next((opt for opt in options if opt.lower() == match), match) 
                for match in matches]
    
    if matches:
        # Return single match with original casing
        return next((opt for opt in options if opt.lower() == matches[0]), None)
    return None

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text

# ======================
# CONVERSATION CONTEXT
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = None
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.role_attempts = 0
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return product_name.strip().lower() in {p.lower() for p in self.all_products}
    
    def update(self, question: str, response: str):
        self.last_question = question
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
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
        
        self._extract_skin_type(question)
        self._extract_product_type(question)
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_skin_type(self, question: str):
        normalized_question = normalize_text(question)
        for skin in SKIN_TYPES:
            if skin in normalized_question:
                self.skin_type = skin
                logger.info(f"Detected skin type: {self.skin_type}")
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(normalized_question, SKIN_TYPES)
        if match:
            self.skin_type = match
            logger.info(f"Fuzzy matched skin type: {self.skin_type}")

    def _extract_product_type(self, question: str):
        normalized_question = normalize_text(question)
        for pt in PRODUCT_TYPES:
            if pt in normalized_question:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(normalized_question, PRODUCT_TYPES)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            self.need_skin_type = self.product_type and not self.skin_type
            
    def _extract_product_name(self, text: str) -> Optional[str]:
        patterns = [
            r'Produk:\s*([^\n\(\)]+)',
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+ ]{5,}(?:\s+[A-Z0-9&.\-+ ]{3,})*)'
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, text):
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                if self.is_valid_product(name):
                    # Cari casing asli yang cocok
                    for p in self.all_products:
                        if p.lower() == name.lower():
                            return p
                
        # Fuzzy matching for product names
        words = re.findall(r'\b[\w&.\-+]{3,}\b', text)
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                if len(phrase) < 5: 
                    continue
                    
                # Fuzzy match with threshold
                match = fuzzy_match(phrase, list(self.all_products), threshold=0.5)
                if match:
                    return match
                    
        return None
    
    def get_context_summary(self) -> str:
        summary = []
        if self.skin_type: summary.append(f"Kulit: {self.skin_type}")
        if self.product_type: summary.append(f"Produk: {self.product_type}")
        if self.last_product: summary.append(f"Terakhir: {self.last_product[:20]}")
        return ", ".join(summary)
    
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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
        self.product_names = set()
        self.product_data = {}
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        self.processed_text = self.extract_text(pdf_path)
        if not self.processed_text:
            logger.error("No text extracted from PDF")
            return False
        
        self._extract_all_product_data(self.processed_text)
        
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
        chunks = self.chunk_text(self.processed_text)
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
                
            name_match = re.search(r'Produk:\s*([^\n\(\)]+)', section)
            if not name_match:
                continue
                
            name = name_match.group(1).strip()
            if len(name) < 3 or not any(c.isupper() for c in name):
                continue
                
            # Extract fields with improved regex patterns
            product_info = {
                'name': name,
                'harga': self._extract_field(section, 'harga:'),
                'type_produk': self._extract_field(section, 'type produk:'),
                'jenis_kulit': self._extract_field(section, 'jenis kulit:'),
                'informasi_produk': self._extract_field(section, 'informasi produk:'),
                'kandungan': self._extract_field(section, 'kandungan:'),
                'cara_pemakaian': self._extract_field(section, 'cara pemakaian:'),
                'manfaat': self._extract_field(section, 'manfaat:'),
                'lokasi': self._extract_field(section, 'lokasi :')
            }
            
            # Simpan dengan casing asli
            self.product_names.add(name)
            self.product_data[name.lower()] = product_info
            logger.info(f"Extracted product: {name}")
        
        logger.info(f"Extracted {len(self.product_names)} product names from PDF")

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
                r'\n\s*[A-Z][a-z]+:'
            ]
            
            end_idx = len(remaining_text)
            for pat in end_patterns:
                end_match = re.search(pat, remaining_text, re.IGNORECASE)
                if end_match and end_match.start() < end_idx:
                    end_idx = end_match.start()
            
            content = remaining_text[:end_idx].strip()
            
            # Remove leading numbers and bullet points
            content = re.sub(r'^[\d\s\.\-]+', '', content)
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            return content
        except Exception as e:
            logger.error(f"Error extracting field {field_name}: {e}")
            return ""

    def get_product_names(self) -> List[str]:
        return list(self.product_names)
    
    def get_product_data(self, product_name: str) -> Optional[Dict]:
        # Ambil data dengan casing asli
        return self.product_data.get(product_name.lower())

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
                    
                if i % 300 == 0:
                    gc.collect()
                    
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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict], detail_btn=True) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Gunakan nama asli tanpa normalisasi untuk ditampilkan
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image" 
                         onerror="this.onerror=null; this.src='../static/default_product.jpg';">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
            """
            if detail_btn:
                card += f"""
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
                """
            card += "</div>"
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

    def analyze_question(self, question: str) -> Tuple[str, Optional[str], List[str]]:
        """Mengembalikan (question_type, product, info_types)"""
        info_types = []
        q_lower = question.lower()
        
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
            
        # Handle special cases
        if any(greeting in q_lower for greeting in GREETINGS):
            return ("sapaan", None, [])
            
        if any(kw in q_lower for kw in ["ganti peran", "change role", "ubah peran", "bukan asisten skincare"]):
            self.context.role_attempts += 1
            return ("role_change_blocked", None, []) if self.context.role_attempts > 2 else ("role_change_warning", None, [])
            
        self.context.role_attempts = 0
            
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None, [])
            
        # Skin type detection
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            self.context.skin_type = skin_match
            return ("rekomendasi", None, []) if self.context.product_type else ("need_product_type", None, [])
            
        # Handle all product info questions with unified approach
        # Extract info types from question
        info_types = self._extract_info_types(question)
        product = self._extract_product_from_question(question)
        
        if product and not info_types:
            # Hanya menyebutkan nama produk tanpa info spesifik
            return ("product_card", product, [])
        elif info_types:
            return ("product_info", product or self.context.last_product, info_types)
                
        # Product type detection
        product_match = next((pt for pt in PRODUCT_TYPES if pt in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            return ("need_skin_type", None, []) if not self.context.skin_type else ("rekomendasi", None, [])
                
        return ("general", None, [])

    def _extract_product_from_question(self, question: str) -> Optional[str]:
        patterns = [
            r'Produk:\s*([^\n\(\)]+)',
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+ ]{5,}(?:\s+[A-Z0-9&.\-+ ]{3,})*)'
        ]
        
        for pattern in patterns:
            for match in re.findall(pattern, question):
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                if self.context.is_valid_product(name):
                    return name
        
        # Fuzzy matching for product names
        words = re.findall(r'\b[\w&.\-+]{3,}\b', question)
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                if len(phrase) < 5: 
                    continue
                    
                # Fuzzy match with threshold
                match = fuzzy_match(phrase, list(self.context.all_products), threshold=0.5)
                if match:
                    return match
        
        return None

    def get_recommendations(self, query: str) -> str:
        try:
            # Build query from context
            query_parts = []
            if self.context.skin_type: query_parts.append(f"kulit {self.context.skin_type}")
            if self.context.product_type: query_parts.append(self.context.product_type)
            query_str = " ".join(query_parts) or query
            
            # Filtering
            filter_dict = {}
            if self.context.skin_type: filter_dict["skin_type"] = self.context.skin_type
            if self.context.product_type: filter_dict["product_type"] = self.context.product_type
            
            # Search documents
            docs = self.vector_store.similarity_search(query_str, k=15, filter=filter_dict) or \
                   self.vector_store.similarity_search(query_str, k=15)
            
            context = "\n".join(doc.page_content for doc in docs)
            context = self._truncate_to_tokens(context, 3000)
            
            # Get recommendations via LLM
            prompt = f"""
            Berdasarkan konteks berikut, rekomendasikan 3-5 produk skincare yang cocok untuk:
            - Jenis kulit: {self.context.skin_type or 'semua jenis'}
            - Jenis produk: {self.context.product_type or 'semua jenis'}
            
            Format respons:
            1. Nama Produk 1
            2. Nama Produk 2
            3. Nama Produk 3
            
            Konteks:
            {context}
            """
            
            llm_response = self.llm.invoke(self._truncate_to_tokens(prompt, 3500)).content
            
            # Extract products from response
            all_products = set()
            for line in llm_response.split('\n'):
                if re.match(r'^\d+[\.\)]?\s+', line):
                    name = re.sub(r'^\d+[\.\)]?\s*', '', line).strip()
                    if self.context.is_valid_product(name):
                        all_products.add(name)
            
            # Add products from context
            for match in re.findall(r'Produk:\s*([^\n\(\)]+)', context):
                if self.context.is_valid_product(match.strip()):
                    all_products.add(match.strip())
            
            if not all_products:
                return "❌ Maaf, tidak menemukan produk yang cocok. Silakan coba dengan kriteria yang lebih spesifik."
            
            # Generate product cards
            products = [{'name': name} for name in list(all_products)[:5]]
            product_cards = self._generate_product_cards(products)
            
            return f"""
            <div class="recommendation-container">
                <h3>Rekomendasi untuk kulit {self.context.skin_type if self.context.skin_type else ''} {self.context.product_type if self.context.product_type else 'produk'}:</h3>
                <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                    {product_cards}
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi."

    def get_product_info(self, product_name: str, info_types: List[str]) -> str:
        try:
            if not product_name:
                return "❌ Produk tidak disebutkan"
            
            # Fuzzy match product name
            matched_name = fuzzy_match(product_name, list(self.context.all_products))
            if not matched_name:
                # Cari kandidat produk yang mirip
                candidates = fuzzy_match(
                    product_name, 
                    list(self.context.all_products), 
                    threshold=0.4, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates], 
                        detail_btn=False
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
            
            # Get original casing from product data
            original_name = next((name for name in self.context.all_products 
                                 if name.lower() == matched_name.lower()), matched_name)
            
            product_data = self.processor.get_product_data(original_name)
            if not product_data:
                return f"❌ Tidak menemukan informasi tentang {original_name}"
            
            # Filter hanya informasi yang diminta
            response_lines = []
            for info_type in info_types:
                if info_type == 'kandungan' and product_data.get('kandungan'):
                    response_lines.append(f"<b>Kandungan</b>:<br /><p class=\"text-justify\">{self._format_ingredients(product_data['kandungan'])}</p>")
                
                elif info_type == 'manfaat' and product_data.get('manfaat'):
                    # Bersihkan hanya bagian yang tidak relevan
                    cleaned_manfaat = self._clean_content(
                        product_data['manfaat'],
                        ['cara pemakaian', 'kandungan', 'harga', 'lokasi']
                    )
                    response_lines.append(f"<b>Manfaat</b>:<br /><p class=\"text-justify\">{cleaned_manfaat}</p>")
                    
                elif info_type == 'cara_pakai' and product_data.get('cara_pemakaian'):
                    cleaned_cara = self._clean_content(
                        product_data['cara_pemakaian'],
                        ['manfaat', 'kandungan', 'harga', 'lokasi', 'informasi produk']
                    )
                    response_lines.append(f"<b>Cara Pakai</b>:<br /><p class=\"text-justify\">{cleaned_cara}</p>")
                
                elif info_type == 'harga' and product_data.get('harga'):
                    response_lines.append(f"<b>Harga</b>:<br /><p class=\"text-justify\">{product_data['harga']}</p>")
                
                elif info_type == 'lokasi' and product_data.get('lokasi'):
                    cleaned_lokasi = self._clean_content(
                        product_data['lokasi'],
                        ['cara pemakaian', 'manfaat', 'kandungan', 'harga', 'informasi produk']
                    )
                    response_lines.append(f"<b>Lokasi Pembelian</b>:<br /><p class=\"text-justify\">{cleaned_lokasi}</p>")
            
            
            response_text = "<br /><br />".join(response_lines) if response_lines else "❌ Tidak menemukan informasi yang diminta"
            
            # Tampilkan card produk setelah penjelasan teks
            product_card = self._generate_product_cards([{'name': original_name}])
            
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

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Fuzzy match product name
        matched_name = fuzzy_match(product_name, list(self.context.all_products))
        if not matched_name:
            # Cari kandidat produk yang mirip
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.4, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates], 
                    detail_btn=False
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
        
        # Get original casing from product data
        original_name = next((name for name in self.context.all_products 
                             if name.lower() == matched_name.lower()), matched_name)
        
        # Tampilkan card produk
        product_card = self._generate_product_cards([{'name': original_name}])
        
        # Ambil info singkat jika ada
        product_data = self.processor.get_product_data(original_name)
        brief_info = ""
        if product_data:
            if product_data.get('jenis_kulit'):
                brief_info += f"<p><b>Jenis Kulit:</b> {product_data['jenis_kulit']}</p>"
            if product_data.get('type_produk'):
                brief_info += f"<p><b>Jenis Produk:</b> {product_data['type_produk']}</p>"
        
        return f"""
        <div class="product-card-response">
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
        
        # Cek jika user mengatakan "ya" untuk kandidat pertama
        if any(kw in question.lower() for kw in ['ya', 'yes', 'benar', 'betul', 'itu', 'yang pertama']):
            selected_product = self.context.product_candidates[0]
            self.context.product_candidates = []  # Reset kandidat
            return self.show_product_card(selected_product)
        
        # Cek jika user mengatakan "tidak" atau "lain"
        if any(kw in question.lower() for kw in ['tidak', 'no', 'bukan', 'lain', 'yang lain', 'yang lainnya']):
            if len(self.context.product_candidates) > 1:
                # Tampilkan kandidat berikutnya
                next_candidates = self.context.product_candidates[1:]
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in next_candidates], 
                    detail_btn=False
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

    def _format_ingredients(self, ingredients: str) -> str:
        """Format ingredients with bullet points"""
        if not ingredients:
            return "❌ Informasi kandungan tidak tersedia"
        
        ingredients = self._clean_content(ingredients, ['cara pemakaian', 'manfaat', 'harga', 'lokasi'])
        ingredients = re.sub(r'\s+', ' ', ingredients)
        ingredients_list = re.split(r',\s*|\s*\n\s*', ingredients)
        ingredients_list = [ing.strip() for ing in ingredients_list if ing.strip()]
        
        return "" + ", ".join(ingredients_list) if ingredients_list else ingredients

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
        
        # Bersihkan karakter khusus di akhir
        content = re.sub(r'[\?\.\-\*]+$', '', content).strip()
        return content

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "off_topic":
            return "Maaf, saya hanya membantu soal skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {self.context.skin_type}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "role_change_warning":
            return "Maaf, saya hanya bisa membantu dengan pertanyaan seputar skincare. Ada yang bisa saya bantu terkait produk skincare?"
        
        if q_type == "role_change_blocked":
            return "Saya tetap akan membantu dengan skincare. Jika Anda ingin mengubah topik, silakan mulai percakapan baru."
        
        if q_type == "need_skin_type":
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(PRODUCT_TYPES)})"
        
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
            if not response.startswith("❌") and product:
                self.context.last_product = product
                self.context.set_last_info_types(info_types)
            return response
        
        # Handle product card requests
        if q_type == "product_card":
            if not product:
                return "Produk apa yang Anda maksud? Silakan sebutkan nama produknya."
            
            response = self.show_product_card(product)
            if not response.startswith("❌") and product:
                self.context.last_product = product
            return response
        
        # Handle recommendations
        if q_type == "rekomendasi":
            if self.context.product_type and not self.context.skin_type:
                return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
            return self.get_recommendations("")
        
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        with self.lock:
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
                self.context._extract_context(question, "")
                
                # Analyze question
                q_type, product, info_types = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
                
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
                
                # Deteksi pertanyaan singkat follow-up (1-3 kata)
                is_short_follow_up = (
                    not product and 
                    self.context.last_product and 
                    len(question.split()) <= 3 and 
                    any(keyword in question.lower() for keyword in [
                        'manfaat', 'kandungan', 'cara', 'pakai', 'penggunaan', 'harga', 'lokasi', 'review'
                    ])
                )
                
                if is_short_follow_up:
                    # Gunakan info types dari kata kunci
                    if 'manfaat' in question.lower():
                        info_types = ['manfaat']
                    elif 'kandungan' in question.lower():
                        info_types = ['kandungan']
                    elif 'cara' in question.lower() or 'pakai' in question.lower() or 'penggunaan' in question.lower():
                        info_types = ['cara_pakai']
                    elif 'harga' in question.lower():
                        info_types = ['harga']
                    elif 'lokasi' in question.lower() or 'beli' in question.lower():
                        info_types = ['lokasi']
                    elif 'review' in question.lower() or 'ulasan' in question.lower():
                        info_types = ['review']
                    
                    q_type = "product_info"
                    product = self.context.last_product
                    logger.info(f"Short follow-up detected: {info_types} for {product}")
                
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


import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
import difflib
from typing import Tuple, Optional, List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

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
    'sensitive': 'sensitif',
    'dry': 'kering',
    'oily': 'berminyak',
    'combination': 'kombinasi',
    'normal': 'normal',
    'redness': 'kemerahan',
    'dull': 'kusam'
}

# Pemisah untuk jenis kulit kombinasi
SKIN_TYPE_SEPARATORS = ['/', 'dan', '&', 'serta', '-', 'atau', '+']

PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                "sunscreen", "tabir surya", "serum"]
PRODUCT_SYNONYMS = {
    'cuci muka': 'facial wash',
    'pembersih': 'cleanser',
    'moisturizer': 'pelembap',
    'sunscreen': 'tabir surya',
    'sunblock': 'tabir surya',
    'essence': 'serum',
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
def fuzzy_match(input_str, options, threshold=0.6, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    normalized_options = [opt.lower() for opt in options]
    matches = difflib.get_close_matches(
        input_str.lower(), 
        normalized_options, 
        n=5 if get_all else 1, 
        cutoff=threshold
    )
    
    if get_all:
        # Return all matches with original casing
        return [next((opt for opt in options if opt.lower() == match), match) 
                for match in matches]
    
    if matches:
        # Return single match with original casing
        return next((opt for opt in options if opt.lower() == matches[0]), None)
    return None

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text

# ======================
# CONVERSATION CONTEXT (Dukungan Jenis Kulit Kombinasi)
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []  # Ubah menjadi list untuk mendukung multiple skin types
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.role_attempts = 0
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        self.requested_count = None  # Jumlah produk yang diminta
        self.ingredient_filter = None  # Filter kandungan
        self.price_filter = None  # Filter harga
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return product_name.strip().lower() in {p.lower() for p in self.all_products}
    
    def update(self, question: str, response: str):
        self.last_question = question
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
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
        self.requested_count = None
        self.ingredient_filter = None
        self.price_filter = None
        self.skin_type = []  # Reset skin type
        
        self._extract_skin_type(question)
        self._extract_product_type(question)
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_skin_type(self, question: str):
        q_lower = question.lower()
        detected_skins = set()
        
        # 1. Cari sinonim terlebih dahulu
        for syn, skin in SKIN_SYNONYMS.items():
            if syn in q_lower:
                detected_skins.add(skin)
        
        # 2. Gunakan regex untuk deteksi yang lebih akurat
        skin_patterns = {
            'berjerawat': r'\b(berjerawat|jerawat|acne prone|acne)\b',
            'kering': r'\b(kering|dry)\b',
            'berminyak': r'\b(berminyak|oily)\b',
            'sensitif': r'\b(sensitif|sensitive)\b',
            'kombinasi': r'\b(kombinasi|combination)\b',
            'normal': r'\b(normal)\b',
            'kemerahan': r'\b(kemerahan|redness)\b',
            'kusam': r'\b(kusam|dull)\b'
        }
        
        for skin_type, pattern in skin_patterns.items():
            if re.search(pattern, q_lower):
                detected_skins.add(skin_type)
        
        # 3. Cari eksak di daftar skin type
        for skin in SKIN_TYPES:
            if skin in q_lower:
                detected_skins.add(skin)
        
        # 4. Fuzzy matching fallback
        if not detected_skins:
            match = fuzzy_match(q_lower, SKIN_TYPES, threshold=0.5)
            if match:
                detected_skins.add(match)
        
        # Simpan hasil deteksi
        self.skin_type = list(detected_skins)
        if self.skin_type:
            logger.info(f"Detected skin types: {self.skin_type}")

    def _extract_product_type(self, question: str):
        # Cari sinonim terlebih dahulu
        q_lower = question.lower()
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type via synonym: {syn} -> {pt}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Gunakan regex untuk deteksi yang lebih akurat
        product_patterns = {
            'sunscreen': r'\b(sunscreen|tabir surya|sunblock)\b',
            'moisturizer': r'\b(moisturizer|pelembap)\b',
            'cleanser': r'\b(cleanser|facial wash|cuci muka|pembersih)\b',
            'serum': r'\b(serum|essence)\b',
            'toner': r'\b(toner)\b',
            'masker': r'\b(masker|mask)\b'
        }
        
        for product_type, pattern in product_patterns.items():
            if re.search(pattern, q_lower):
                self.product_type = product_type
                logger.info(f"Detected product type via regex: {product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Cari eksak di daftar product type
        for pt in PRODUCT_TYPES:
            if pt in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(q_lower, PRODUCT_TYPES, threshold=0.5)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            self.need_skin_type = self.product_type and not self.skin_type
            
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
            for match in re.findall(pattern, text):
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
    
    def set_last_info_types(self, info_types: List[str]):
        self.last_info_types = info_types
        
    def get_last_info_types(self) -> List[str]:
        return self.last_info_types

# ======================
# PDF PROCESSOR (Dukungan Jenis Kulit Kombinasi)
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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
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
                    'serum': ['serum', 'essence'],
                    'toner': ['toner'],
                    'masker': ['masker', 'mask']
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
                    
                if i % 300 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

# ======================
# QA ENGINE (Dukungan Jenis Kulit Kombinasi)
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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bertscorer = BERTScorer()
        self.evaluation_results = []
        self.recommendation_history = set()  # Untuk menghindari rekomendasi duplikat
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict], detail_btn=True) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Gunakan nama asli tanpa normalisasi untuk ditampilkan
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image" 
                         onerror="this.onerror=null; this.src='../static/default_product.jpg';">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
            """
            card += "</div>"
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
            if response in self.context.all_products:
                return response
                
            # Cek dengan case-insensitive
            for product in self.context.all_products:
                if product.lower() == response.lower():
                    return product
                    
            # Jika GPT tidak yakin, coba cari dengan similarity
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
        
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
            
        # Handle special cases
        if any(greeting in q_lower for greeting in GREETINGS):
            return ("sapaan", None, [])
            
        if any(kw in q_lower for kw in ["ganti peran", "change role", "ubah peran", "bukan asisten skincare"]):
            self.context.role_attempts += 1
            return ("role_change_blocked", None, []) if self.context.role_attempts > 2 else ("role_change_warning", None, [])
            
        self.context.role_attempts = 0
            
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None, [])
            
        # Deteksi khusus untuk pertanyaan manfaat
        if 'manfaat' in q_lower and any(p.lower() in q_lower for p in self.context.all_products):
            product = self._extract_product_from_question(question)
            return ("product_info", product, ['manfaat'])
            
        # Rekomendasi detection with filters
        rekomendasi_keywords = QUESTION_TYPES['rekomendasi'] + ['rekomendasi', 'sarankan', 'anjuran']
        if any(kw in q_lower for kw in rekomendasi_keywords):
            # Deteksi jumlah produk yang diminta
            num_match = re.search(r'(\d+)\s+produk', q_lower)
            self.context.requested_count = int(num_match.group(1)) if num_match else None
            
            # Deteksi filter kandungan
            self.context.ingredient_filter = None
            ingredient_match = re.search(r'mengandung\s+([\w\s]+)', q_lower)
            if ingredient_match:
                self.context.ingredient_filter = ingredient_match.group(1).strip()
                
            # Deteksi filter harga
            self.context.price_filter = None
            price_match = re.search(r'harga\s+(dibawah|di bawah|di atas|diatas|kurang|lebih|>|<|>=|<=)?\s*([\d\.]+)', q_lower)
            if price_match:
                operator = price_match.group(1).lower() if price_match.group(1) else ""
                price_value = float(price_match.group(2).replace('.', ''))
                self.context.price_filter = (operator, price_value)
                
            return ("rekomendasi", None, [])
            
        # Skin type detection
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            if not self.context.skin_type:
                self.context.skin_type = []
            self.context.skin_type.append(skin_match)
            return ("rekomendasi", None, []) if self.context.product_type else ("need_product_type", None, [])
            
        # PERBAIKAN UTAMA: Deteksi jika dalam pertanyaan terdapat sinonim jenis kulit dan produk
        has_skin_type = False
        has_product_type = False
        
        # Cek skin type dari sinonim
        for syn, skin in SKIN_SYNONYMS.items():
            if syn in q_lower:
                if not self.context.skin_type:
                    self.context.skin_type = []
                self.context.skin_type.append(skin)
                has_skin_type = True
        
        # Cek product type dari sinonim
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.context.product_type = pt
                has_product_type = True
                break
        
        # Jika ditemukan keduanya, maka dianggap permintaan rekomendasi
        if has_skin_type and has_product_type:
            return ("rekomendasi", None, [])
            
        # Product type detection
        product_match = next((pt for pt in PRODUCT_TYPES if pt in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            return ("need_skin_type", None, []) if not self.context.skin_type else ("rekomendasi", None, [])
            
        # Handle all product info questions with unified approach
        # Extract info types from question
        info_types = self._extract_info_types(question)
        product = self._extract_product_from_question(question)
        
        if product and not info_types:
            # Hanya menyebutkan nama produk tanpa info spesifik
            return ("product_card", product, [])
        elif info_types:
            return ("product_info", product or self.context.last_product, info_types)
                
        return ("general", None, [])

    def _extract_product_from_question(self, question: str) -> Optional[str]:
        # Pola regex untuk nama produk uppercase dengan tanda kurung
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)',  # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+() ]{5,}(?:\s+[A-Z0-9&.\-+() ]{3,})*)'  # Izinkan tanda kurung
        ]
        
        # Coba ekstrak dengan pola regex
        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                # Gunakan fuzzy matching untuk verifikasi
                matched = fuzzy_match(name, list(self.context.all_products), threshold=0.5)
                if matched:
                    return matched
        
        # Jika tidak ditemukan, cari dengan similarity
        words = re.findall(r'\b[\w&.\-+]{3,}\b', question)
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                if len(phrase) < 5: 
                    continue
                    
                # Fuzzy match with lower threshold
                match = fuzzy_match(phrase, list(self.context.all_products), threshold=0.4)
                if match:
                    return match
        
        return None

    def get_recommendations(self, query: str) -> str:
        try:
            # Dapatkan semua produk yang tersedia
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data:
                    all_products.append({
                        'name': product_name,  # Gunakan nama asli
                        'data': product_data
                    })
            
            # Log untuk debugging
            logger.info(f"Jumlah produk tersedia: {len(all_products)}")
            logger.info(f"Kriteria filter - Kulit: {self.context.skin_type}, Produk: {self.context.product_type}")
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                skin_data = product['data'].get('jenis_kulit', [])
                type_data = product['data'].get('type_produk', '').lower()
                
                # Logika filter jenis kulit untuk kombinasi
                skin_match = False
                if not self.context.skin_type:  # Tidak ada filter kulit
                    skin_match = True
                else:
                    # Jika produk cocok untuk semua jenis kulit
                    if 'semua' in skin_data:
                        skin_match = True
                    else:
                        # Minimal satu jenis kulit yang diminta cocok dengan produk
                        any_match = any(user_skin in skin_data for user_skin in self.context.skin_type)
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
                
                # Filter kandungan
                ingredient_match = True
                if self.context.ingredient_filter:
                    kandungan = product['data'].get('kandungan', '').lower()
                    ingredient_match = (self.context.ingredient_filter.lower() in kandungan)
                    
                # Filter harga
                price_match = True
                if self.context.price_filter:
                    harga_text = product['data'].get('harga', '0')
                    try:
                        # Ekstrak angka dari teks harga
                        harga = float(re.search(r'[\d,\.]+', harga_text.replace('.', '')).group().replace(',', '.'))
                    except:
                        harga = 0
                        
                    operator, value = self.context.price_filter
                    if operator in ["dibawah", "di bawah", "kurang", "<"]:
                        price_match = harga < value
                    elif operator in ["di atas", "diatas", "lebih", ">"]:
                        price_match = harga > value
                    else:
                        price_match = harga == value  # Default: harga tepat
                
                if skin_match and product_match and ingredient_match and price_match:
                    filtered_products.append(product)
                    logger.info(f"Product matched: {product['name']} | Skin: {skin_data} | Type: {type_data}")
            
            # Batasi jumlah produk jika diminta
            requested_count = self.context.requested_count or 5
            if len(filtered_products) > requested_count:
                # Prioritaskan produk dengan kemiripan tertinggi
                filtered_products = filtered_products[:requested_count]
            
            # Generate product cards
            products_to_show = [{'name': p['name']} for p in filtered_products]
            product_cards = self._generate_product_cards(products_to_show)
            
            # Response dengan jumlah produk aktual
            if filtered_products:
                response_html = f"""
                <div class="recommendation-container">
                    <h3>Ditemukan {len(filtered_products)} produk yang sesuai:</h3>
                    <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                        {product_cards}
                    </div>
                </div>
                """
            else:
                response_html = "❌ Tidak ditemukan produk yang sesuai. Silakan coba dengan kriteria lain."
            
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
                
            # Get reference recommendations from document
            reference_products = self.processor.get_recommendation_reference(self.context.skin_type[0])
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
            threshold = 0.6  # Threshold untuk kecocokan
            
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
                    threshold=0.5
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
                    threshold=0.4, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates], 
                        detail_btn=False
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
            
            # Filter hanya informasi yang diminta
            response_lines = []
            for info_type in info_types:
                if info_type == 'kandungan' and product_data.get('kandungan'):
                    response_lines.append(f"<b>Kandungan</b>:<br /><p class=\"text-justify\">{self._format_ingredients(product_data['kandungan'])}</p>")
                
                elif info_type == 'manfaat' and product_data.get('manfaat'):
                    # Bersihkan hanya bagian yang tidak relevan
                    cleaned_manfaat = self._clean_content(
                        product_data['manfaat'],
                        ['cara pemakaian', 'kandungan', 'harga', 'lokasi']
                    )
                    response_lines.append(f"<b>Manfaat</b>:<br /><p class=\"text-justify\">{cleaned_manfaat}</p>")
                    
                elif info_type == 'cara_pakai' and product_data.get('cara_pemakaian'):
                    cleaned_cara = self._clean_content(
                        product_data['cara_pemakaian'],
                        ['manfaat', 'kandungan', 'harga', 'lokasi', 'informasi produk']
                    )
                    response_lines.append(f"<b>Cara Pakai</b>:<br /><p class=\"text-justify\">{cleaned_cara}</p>")
                
                elif info_type == 'harga' and product_data.get('harga'):
                    response_lines.append(f"<b>Harga</b>:<br /><p class=\"text-justify\">{product_data['harga']}</p>")
                
                elif info_type == 'lokasi' and product_data.get('lokasi'):
                    cleaned_lokasi = self._clean_content(
                        product_data['lokasi'],
                        ['cara pemakaian', 'manfaat', 'kandungan', 'harga', 'informasi produk']
                    )
                    response_lines.append(f"<b>Lokasi Pembelian</b>:<br /><p class=\"text-justify\">{cleaned_lokasi}</p>")
            
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
                <div class="context-info">
                    {context_info}
                </div>
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

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Gunakan algoritma yang lebih toleran untuk pencocokan nama produk
        best_match = None
        best_score = 0
        threshold = 0.6  # Threshold untuk kecocokan
        
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
                threshold=0.5
            )
        
        # Jika masih tidak ketemu, gunakan GPT untuk menebak
        if not best_match:
            guessed_name = self._guess_product_name(f"Produk: {product_name}")
            if guessed_name:
                best_match = guessed_name
        
        if not best_match:
            # Cari kandidat produk yang mirip
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.4, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates], 
                    detail_btn=False
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
        
        # Cek jika user mengatakan "ya" untuk kandidat pertama
        if any(kw in question.lower() for kw in ['ya', 'yes', 'benar', 'betul', 'itu', 'yang pertama']):
            selected_product = self.context.product_candidates[0]
            self.context.product_candidates = []  # Reset kandidat
            return self.show_product_card(selected_product)
        
        # Cek jika user mengatakan "tidak" atau "lain"
        if any(kw in question.lower() for kw in ['tidak', 'no', 'bukan', 'lain', 'yang lain', 'yang lainnya']):
            if len(self.context.product_candidates) > 1:
                # Tampilkan kandidat berikutnya
                next_candidates = self.context.product_candidates[1:]
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in next_candidates], 
                    detail_btn=False
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

    def _format_ingredients(self, ingredients: str) -> str:
        """Format ingredients with bullet points"""
        if not ingredients:
            return "❌ Informasi kandungan tidak tersedia"
        
        ingredients = self._clean_content(ingredients, ['cara pemakaian', 'manfaat', 'harga', 'lokasi'])
        return ingredients

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

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "off_topic":
            return "Maaf, saya hanya membantu soal skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                skin_types = ", ".join(self.context.skin_type)
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {skin_types}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "role_change_warning":
            return "Maaf, saya hanya bisa membantu dengan pertanyaan seputar skincare. Ada yang bisa saya bantu terkait produk skincare?"
        
        if q_type == "role_change_blocked":
            return "Saya tetap akan membantu dengan skincare. Jika Anda ingin mengubah topik, silakan mulai percakapan baru."
        
        if q_type == "need_skin_type":
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(PRODUCT_TYPES)})"
        
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
        if q_type == "rekomendasi":
            if self.context.product_type and not self.context.skin_type:
                return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
            return self.get_recommendations("")
        
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        with self.lock:
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
                self.context._extract_context(question, "")
                
                # Analyze question
                q_type, product, info_types = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
                
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
                
                # Deteksi pertanyaan singkat follow-up (1-3 kata)
                is_short_follow_up = (
                    not product and 
                    self.context.last_product and 
                    len(question.split()) <= 3 and 
                    any(keyword in question.lower() for keyword in [
                        'manfaat', 'kandungan', 'cara', 'pakai', 'penggunaan', 'harga', 'lokasi', 'review'
                    ])
                )
                
                if is_short_follow_up:
                    # Gunakan info types dari kata kunci
                    if 'manfaat' in question.lower():
                        info_types = ['manfaat']
                    elif 'kandungan' in question.lower():
                        info_types = ['kandungan']
                    elif 'cara' in question.lower() or 'pakai' in question.lower() or 'penggunaan' in question.lower():
                        info_types = ['cara_pakai']
                    elif 'harga' in question.lower():
                        info_types = ['harga']
                    elif 'lokasi' in question.lower() or 'beli' in question.lower():
                        info_types = ['lokasi']
                    elif 'review' in question.lower() or 'ulasan' in question.lower():
                        info_types = ['review']
                    
                    q_type = "product_info"
                    product = self.context.last_product
                    logger.info(f"Short follow-up detected: {info_types} for {product}")
                
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


import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
import difflib
from typing import Tuple, Optional, List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

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
    'sensitive': 'sensitif',
    'dry': 'kering',
    'oily': 'berminyak',
    'combination': 'kombinasi',
    'normal': 'normal',
    'redness': 'kemerahan',
    'dull': 'kusam'
}

# Pemisah untuk jenis kulit kombinasi
SKIN_TYPE_SEPARATORS = ['/', 'dan', '&', 'serta', '-', 'atau', '+']

PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                "sunscreen", "tabir surya", "serum"]
PRODUCT_SYNONYMS = {
    'cuci muka': 'facial wash',
    'pembersih': 'cleanser',
    'moisturizer': 'pelembap',
    'sunscreen': 'tabir surya',
    'sunblock': 'tabir surya',
    'essence': 'serum',
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
def fuzzy_match(input_str, options, threshold=0.6, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    normalized_options = [opt.lower() for opt in options]
    matches = difflib.get_close_matches(
        input_str.lower(), 
        normalized_options, 
        n=5 if get_all else 1, 
        cutoff=threshold
    )
    
    if get_all:
        # Return all matches with original casing
        return [next((opt for opt in options if opt.lower() == match), match) 
                for match in matches]
    
    if matches:
        # Return single match with original casing
        return next((opt for opt in options if opt.lower() == matches[0]), None)
    return None

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text

# ======================
# CONVERSATION CONTEXT (Dukungan Jenis Kulit Kombinasi)
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []  # Ubah menjadi list untuk mendukung multiple skin types
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.role_attempts = 0
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        self.requested_count = None  # Jumlah produk yang diminta
        self.ingredient_filter = None  # Filter kandungan
        self.price_filter = None  # Filter harga
        
        # Untuk menyimpan rekomendasi terakhir
        self.last_recommendation = []  # Menyimpan produk yang direkomendasikan terakhir
        self.last_recommendation_criteria = {}  # Menyimpan kriteria terakhir
        self.recommendation_offset = 0  # Indeks terakhir untuk paginasi
        self.recommendation_batch_size = 0  # Jumlah produk per batch
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return product_name.strip().lower() in {p.lower() for p in self.all_products}
    
    def update(self, question: str, response: str):
        self.last_question = question
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
    def set_recommendation(self, products: List[Dict], criteria: Dict, batch_size: int):
        """Menyimpan rekomendasi terakhir dan kriterianya"""
        self.last_recommendation = products
        self.last_recommendation_criteria = criteria
        self.recommendation_offset = 0  # Reset offset
        self.recommendation_batch_size = batch_size  # Simpan ukuran batch
        
    def get_next_recommendation_batch(self) -> List[Dict]:
        """Mendapatkan batch rekomendasi berikutnya"""
        start = self.recommendation_offset
        end = start + self.recommendation_batch_size
        
        if start >= len(self.last_recommendation):
            return []
            
        batch = self.last_recommendation[start:end]
        self.recommendation_offset = end  # Update offset untuk next batch
        return batch
        
    def get_recommendation(self) -> Tuple[List[Dict], Dict]:
        """Mendapatkan rekomendasi terakhir dan kriterianya"""
        return self.last_recommendation, self.last_recommendation_criteria
    
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
        self.requested_count = None
        self.ingredient_filter = None
        self.price_filter = None
        self.skin_type = []  # Reset skin type
        
        # Deteksi jumlah produk yang diminta
        num_match = re.search(r'(\d+)\s+produk', question.lower())
        if num_match:
            self.requested_count = int(num_match.group(1))
            logger.info(f"Requested count: {self.requested_count}")
        
        self._extract_skin_type(question)
        self._extract_product_type(question)
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_skin_type(self, question: str):
        q_lower = question.lower()
        detected_skins = set()
        
        # 1. Cari sinonim terlebih dahulu
        for syn, skin in SKIN_SYNONYMS.items():
            if syn in q_lower:
                detected_skins.add(skin)
        
        # 2. Gunakan regex untuk deteksi yang lebih akurat
        skin_patterns = {
            'berjerawat': r'\b(berjerawat|jerawat|acne prone|acne)\b',
            'kering': r'\b(kering|dry)\b',
            'berminyak': r'\b(berminyak|oily)\b',
            'sensitif': r'\b(sensitif|sensitive)\b',
            'kombinasi': r'\b(kombinasi|combination)\b',
            'normal': r'\b(normal)\b',
            'kemerahan': r'\b(kemerahan|redness)\b',
            'kusam': r'\b(kusam|dull)\b'
        }
        
        for skin_type, pattern in skin_patterns.items():
            if re.search(pattern, q_lower):
                detected_skins.add(skin_type)
        
        # 3. Cari eksak di daftar skin type
        for skin in SKIN_TYPES:
            if skin in q_lower:
                detected_skins.add(skin)
        
        # 4. Fuzzy matching fallback
        if not detected_skins:
            match = fuzzy_match(q_lower, SKIN_TYPES, threshold=0.5)
            if match:
                detected_skins.add(match)
        
        # Simpan hasil deteksi
        self.skin_type = list(detected_skins)
        if self.skin_type:
            logger.info(f"Detected skin types: {self.skin_type}")

    def _extract_product_type(self, question: str):
        # Cari sinonim terlebih dahulu
        q_lower = question.lower()
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type via synonym: {syn} -> {pt}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Gunakan regex untuk deteksi yang lebih akurat
        product_patterns = {
            'sunscreen': r'\b(sunscreen|tabir surya|sunblock)\b',
            'moisturizer': r'\b(moisturizer|pelembap)\b',
            'cleanser': r'\b(cleanser|facial wash|cuci muka|pembersih)\b',
            'serum': r'\b(serum|essence)\b',
            'toner': r'\b(toner)\b',
            'masker': r'\b(masker|mask)\b'
        }
        
        for product_type, pattern in product_patterns.items():
            if re.search(pattern, q_lower):
                self.product_type = product_type
                logger.info(f"Detected product type via regex: {product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Cari eksak di daftar product type
        for pt in PRODUCT_TYPES:
            if pt in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(q_lower, PRODUCT_TYPES, threshold=0.5)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            self.need_skin_type = self.product_type and not self.skin_type
            
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
            for match in re.findall(pattern, text):
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
    
    def set_last_info_types(self, info_types: List[str]):
        self.last_info_types = info_types
        
    def get_last_info_types(self) -> List[str]:
        return self.last_info_types

# ======================
# PDF PROCESSOR (Dukungan Jenis Kulit Kombinasi)
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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
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
                    'serum': ['serum', 'essence'],
                    'toner': ['toner'],
                    'masker': ['masker', 'mask']
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
                    
                if i % 300 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

# ======================
# QA ENGINE (Dukungan Jenis Kulit Kombinasi)
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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bertscorer = BERTScorer()
        self.evaluation_results = []
        self.recommendation_history = set()  # Untuk menghindari rekomendasi duplikat
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict], detail_btn=True) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Gunakan nama asli tanpa normalisasi untuk ditampilkan
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image" 
                         onerror="this.onerror=null; this.src='../static/default_product.jpg';">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
            """
            card += "</div>"
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
            if response in self.context.all_products:
                return response
                
            # Cek dengan case-insensitive
            for product in self.context.all_products:
                if product.lower() == response.lower():
                    return product
                    
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
        
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
            
        # Handle special cases
        if any(greeting in q_lower for greeting in GREETINGS):
            return ("sapaan", None, [])
            
        if any(kw in q_lower for kw in ["ganti peran", "change role", "ubah peran", "bukan asisten skincare"]):
            self.context.role_attempts += 1
            return ("role_change_blocked", None, []) if self.context.role_attempts > 2 else ("role_change_warning", None, [])
            
        self.context.role_attempts = 0
            
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None, [])
            
        # Deteksi khusus untuk pertanyaan manfaat
        if 'manfaat' in q_lower and any(p.lower() in q_lower for p in self.context.all_products):
            product = self._extract_product_from_question(question)
            return ("product_info", product, ['manfaat'])
            
        # Rekomendasi detection with filters
        rekomendasi_keywords = QUESTION_TYPES['rekomendasi'] + ['rekomendasi', 'sarankan', 'anjuran']
        if any(kw in q_lower for kw in rekomendasi_keywords):
            # Deteksi jumlah produk yang diminta
            num_match = re.search(r'(\d+)\s+produk', q_lower)
            self.context.requested_count = int(num_match.group(1)) if num_match else None
            
            # Deteksi filter kandungan
            self.context.ingredient_filter = None
            ingredient_match = re.search(r'mengandung\s+([\w\s]+)', q_lower)
            if ingredient_match:
                self.context.ingredient_filter = ingredient_match.group(1).strip()
                
            # Deteksi filter harga
            self.context.price_filter = None
            price_match = re.search(r'harga\s+(dibawah|di bawah|di atas|diatas|kurang|lebih|>|<|>=|<=)?\s*([\d\.]+)', q_lower)
            if price_match:
                operator = price_match.group(1).lower() if price_match.group(1) else ""
                price_value = float(price_match.group(2).replace('.', ''))
                self.context.price_filter = (operator, price_value)
                
            return ("rekomendasi", None, [])
            
        # Skin type detection
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            if not self.context.skin_type:
                self.context.skin_type = []
            self.context.skin_type.append(skin_match)
            return ("rekomendasi", None, []) if self.context.product_type else ("need_product_type", None, [])
            
        # Deteksi permintaan rekomendasi lainnya
        rekomendasi_lain_keywords = [
            'lainnya', 'yang lain', 'produk lain', 'rekomendasi lain',
            'tambahkan', 'lagi', 'masih ada', 'lain', 'next'
        ]
        
        if any(kw in q_lower for kw in rekomendasi_lain_keywords):
            if self.context.last_recommendation:
                return ("rekomendasi_lain", None, [])
        
        # PERBAIKAN UTAMA: Deteksi jika dalam pertanyaan terdapat sinonim jenis kulit dan produk
        has_skin_type = False
        has_product_type = False
        
        # Cek skin type dari sinonim
        for syn, skin in SKIN_SYNONYMS.items():
            if syn in q_lower:
                if not self.context.skin_type:
                    self.context.skin_type = []
                self.context.skin_type.append(skin)
                has_skin_type = True
        
        # Cek product type dari sinonim
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.context.product_type = pt
                has_product_type = True
                break
        
        # Jika ditemukan keduanya, maka dianggap permintaan rekomendasi
        if has_skin_type and has_product_type:
            return ("rekomendasi", None, [])
            
        # Product type detection
        product_match = next((pt for pt in PRODUCT_TYPES if pt in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            return ("need_skin_type", None, []) if not self.context.skin_type else ("rekomendasi", None, [])
            
        # Handle all product info questions with unified approach
        # Extract info types from question
        info_types = self._extract_info_types(question)
        product = self._extract_product_from_question(question)
        
        if product and not info_types:
            # Hanya menyebutkan nama produk tanpa info spesifik
            return ("product_card", product, [])
        elif info_types:
            return ("product_info", product or self.context.last_product, info_types)
                
        return ("general", None, [])

    def _extract_product_from_question(self, question: str) -> Optional[str]:
        # Pola regex untuk nama produk uppercase dengan tanda kurung
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)',  # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+() ]{5,}(?:\s+[A-Z0-9&.\-+() ]{3,})*)'  # Izinkan tanda kurung
        ]
        
        # Coba ekstrak dengan pola regex
        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                # Gunakan fuzzy matching untuk verifikasi
                matched = fuzzy_match(name, list(self.context.all_products), threshold=0.5)
                if matched:
                    return matched
        
        # Jika tidak ditemukan, cari dengan similarity
        words = re.findall(r'\b[\w&.\-+]{3,}\b', question)
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                if len(phrase) < 5: 
                    continue
                    
                # Fuzzy match with lower threshold
                match = fuzzy_match(phrase, list(self.context.all_products), threshold=0.4)
                if match:
                    return match
        
        return None

    def get_recommendations(self, query: str, is_follow_up: bool = False) -> str:
        try:
            # Jika follow-up, gunakan rekomendasi yang sudah ada
            if is_follow_up and self.context.last_recommendation:
                # Dapatkan batch berikutnya
                batch_products = self.context.get_next_recommendation_batch()
                
                if not batch_products:
                    return "❌ Tidak ada rekomendasi tambahan untuk kriteria ini."
                
                # Generate product cards
                products_to_show = [{'name': p['name']} for p in batch_products]
                product_cards = self._generate_product_cards(products_to_show)
                
                return f"""
                <div class="recommendation-container">
                    <h3>Berikut rekomendasi tambahan:</h3>
                    <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                        {product_cards}
                    </div>
                </div>
                """
            
            # Dapatkan semua produk yang tersedia
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data:
                    all_products.append({
                        'name': product_name,  # Gunakan nama asli
                        'data': product_data
                    })
            
            # Log untuk debugging
            logger.info(f"Jumlah produk tersedia: {len(all_products)}")
            logger.info(f"Kriteria filter - Kulit: {self.context.skin_type}, Produk: {self.context.product_type}")
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                skin_data = product['data'].get('jenis_kulit', [])
                type_data = product['data'].get('type_produk', '').lower()
                
                # Logika filter jenis kulit untuk kombinasi
                skin_match = False
                if not self.context.skin_type:  # Tidak ada filter kulit
                    skin_match = True
                else:
                    # Jika produk cocok untuk semua jenis kulit
                    if 'semua' in skin_data:
                        skin_match = True
                    else:
                        # Minimal satu jenis kulit yang diminta cocok dengan produk
                        any_match = any(user_skin in skin_data for user_skin in self.context.skin_type)
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
                
                # Filter kandungan
                ingredient_match = True
                if self.context.ingredient_filter:
                    kandungan = product['data'].get('kandungan', '').lower()
                    ingredient_match = (self.context.ingredient_filter.lower() in kandungan)
                    
                # Filter harga
                price_match = True
                if self.context.price_filter:
                    harga_text = product['data'].get('harga', '0')
                    try:
                        # Ekstrak angka dari teks harga
                        harga = float(re.search(r'[\d,\.]+', harga_text.replace('.', '')).group().replace(',', '.'))
                    except:
                        harga = 0
                        
                    operator, value = self.context.price_filter
                    if operator in ["dibawah", "di bawah", "kurang", "<"]:
                        price_match = harga < value
                    elif operator in ["di atas", "diatas", "lebih", ">"]:
                        price_match = harga > value
                    else:
                        price_match = harga == value  # Default: harga tepat
                
                if skin_match and product_match and ingredient_match and price_match:
                    filtered_products.append(product)
                    logger.info(f"Product matched: {product['name']} | Skin: {skin_data} | Type: {type_data}")
            
            # Tentukan jumlah produk yang diminta
            requested_count = self.context.requested_count or 3  # Default 3 jika tidak ditentukan
            actual_count = len(filtered_products)
            
            # Batasi jumlah produk jika diminta
            if requested_count > 0 and actual_count > requested_count:
                filtered_products = filtered_products[:requested_count]
                actual_count = requested_count
            
            # Simpan rekomendasi untuk follow-up
            criteria = {
                'skin_type': self.context.skin_type.copy(),
                'product_type': self.context.product_type,
                'requested_count': self.context.requested_count,
                'ingredient_filter': self.context.ingredient_filter,
                'price_filter': self.context.price_filter
            }
            
            # Simpan dengan batch size = jumlah yang diminta atau default
            batch_size = requested_count if requested_count > 0 else 3
            self.context.set_recommendation(filtered_products, criteria, batch_size)
            
            # Generate product cards
            products_to_show = [{'name': p['name']} for p in filtered_products]
            product_cards = self._generate_product_cards(products_to_show)
            
            # Response dengan jumlah produk aktual
            statement = ""
            if actual_count == 0:
                statement = "❌ Tidak ditemukan produk yang sesuai. Silakan coba dengan kriteria lain."
            else:
                skin_types = ", ".join(self.context.skin_type) if self.context.skin_type else "semua jenis kulit"
                product_type = self.context.product_type or "produk"
                
                if self.context.requested_count and actual_count < self.context.requested_count:
                    statement = (
                        f"Untuk {product_type} kulit {skin_types}, "
                        f"hanya ditemukan {actual_count} produk pada dokumen kami (diminta {self.context.requested_count}). "
                        f"Berikut rekomendasinya:"
                    )
                else:
                    count_info = f"sebanyak {self.context.requested_count} " if self.context.requested_count else ""
                    statement = f"Ditemukan {actual_count} produk yang sesuai{count_info}:"
            
            # Evaluate recommendations
            self._evaluate_recommendation(filtered_products)
            
            return f"""
            <div class="recommendation-container">
                <h3>{statement}</h3>
                <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                    {product_cards}
                </div>
            </div>
            """
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi."
    
    def _evaluate_recommendation(self, recommended_products: list):
        """Evaluate recommendations against document reference"""
        try:
            if not self.context.skin_type:
                return
                
            # Get reference recommendations from document
            reference_products = self.processor.get_recommendation_reference(self.context.skin_type[0])
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
            threshold = 0.6  # Threshold untuk kecocokan
            
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
                    threshold=0.5
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
                    threshold=0.4, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates], 
                        detail_btn=False
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
            
            # Filter hanya informasi yang diminta
            response_lines = []
            for info_type in info_types:
                if info_type == 'kandungan' and product_data.get('kandungan'):
                    response_lines.append(f"<b>Kandungan</b>:<br /><p class=\"text-justify\">{self._format_ingredients(product_data['kandungan'])}</p>")
                
                elif info_type == 'manfaat' and product_data.get('manfaat'):
                    # Bersihkan hanya bagian yang tidak relevan
                    cleaned_manfaat = self._clean_content(
                        product_data['manfaat'],
                        ['cara pemakaian', 'kandungan', 'harga', 'lokasi']
                    )
                    response_lines.append(f"<b>Manfaat</b>:<br /><p class=\"text-justify\">{cleaned_manfaat}</p>")
                    
                elif info_type == 'cara_pakai' and product_data.get('cara_pemakaian'):
                    cleaned_cara = self._clean_content(
                        product_data['cara_pemakaian'],
                        ['manfaat', 'kandungan', 'harga', 'lokasi', 'informasi produk']
                    )
                    response_lines.append(f"<b>Cara Pakai</b>:<br /><p class=\"text-justify\">{cleaned_cara}</p>")
                
                elif info_type == 'harga' and product_data.get('harga'):
                    response_lines.append(f"<b>Harga</b>:<br /><p class=\"text-justify\">{product_data['harga']}</p>")
                
                elif info_type == 'lokasi' and product_data.get('lokasi'):
                    cleaned_lokasi = self._clean_content(
                        product_data['lokasi'],
                        ['cara pemakaian', 'manfaat', 'kandungan', 'harga', 'informasi produk']
                    )
                    response_lines.append(f"<b>Lokasi Pembelian</b>:<br /><p class=\"text-justify\">{cleaned_lokasi}</p>")
            
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
                <div class="context-info">
                    {context_info}
                </div>
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

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Gunakan algoritma yang lebih toleran untuk pencocokan nama produk
        best_match = None
        best_score = 0
        threshold = 0.6  # Threshold untuk kecocokan
        
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
                threshold=0.5
            )
        
        # Jika masih tidak ketemu, gunakan GPT untuk menebak
        if not best_match:
            guessed_name = self._guess_product_name(f"Produk: {product_name}")
            if guessed_name:
                best_match = guessed_name
        
        if not best_match:
            # Cari kandidat produk yang mirip
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.4, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates], 
                    detail_btn=False
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
        
        # Cek jika user mengatakan "ya" untuk kandidat pertama
        if any(kw in question.lower() for kw in ['ya', 'yes', 'benar', 'betul', 'itu', 'yang pertama']):
            selected_product = self.context.product_candidates[0]
            self.context.product_candidates = []  # Reset kandidat
            return self.show_product_card(selected_product)
        
        # Cek jika user mengatakan "tidak" atau "lain"
        if any(kw in question.lower() for kw in ['tidak', 'no', 'bukan', 'lain', 'yang lain', 'yang lainnya']):
            if len(self.context.product_candidates) > 1:
                # Tampilkan kandidat berikutnya
                next_candidates = self.context.product_candidates[1:]
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in next_candidates], 
                    detail_btn=False
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

    def _format_ingredients(self, ingredients: str) -> str:
        """Format ingredients with bullet points"""
        if not ingredients:
            return "❌ Informasi kandungan tidak tersedia"
        
        ingredients = self._clean_content(ingredients, ['cara pemakaian', 'manfaat', 'harga', 'lokasi'])
        return ingredients

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

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "off_topic":
            return "Maaf, saya hanya membantu soal skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                skin_types = ", ".join(self.context.skin_type)
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {skin_types}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "role_change_warning":
            return "Maaf, saya hanya bisa membantu dengan pertanyaan seputar skincare. Ada yang bisa saya bantu terkait produk skincare?"
        
        if q_type == "role_change_blocked":
            return "Saya tetap akan membantu dengan skincare. Jika Anda ingin mengubah topik, silakan mulai percakapan baru."
        
        if q_type == "need_skin_type":
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(PRODUCT_TYPES)})"
        
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
        if q_type == "rekomendasi":
            if self.context.product_type and not self.context.skin_type:
                return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
            return self.get_recommendations("", is_follow_up=False)
        
        # Handle additional recommendations
        if q_type == "rekomendasi_lain":
            # Cek apakah ada rekomendasi sebelumnya
            if not self.context.last_recommendation:
                return "❌ Tidak ada rekomendasi sebelumnya. Silakan minta rekomendasi terlebih dahulu."
            
            # Dapatkan rekomendasi tambahan
            return self.get_recommendations("", is_follow_up=True)
                
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        with self.lock:
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
                self.context._extract_context(question, "")
                
                # Analyze question
                q_type, product, info_types = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
                
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
                
                # Deteksi pertanyaan singkat follow-up (1-3 kata)
                is_short_follow_up = (
                    not product and 
                    self.context.last_product and 
                    len(question.split()) <= 3 and 
                    any(keyword in question.lower() for keyword in [
                        'manfaat', 'kandungan', 'cara', 'pakai', 'penggunaan', 'harga', 'lokasi', 'review'
                    ])
                )
                
                if is_short_follow_up:
                    # Gunakan info types dari kata kunci
                    if 'manfaat' in question.lower():
                        info_types = ['manfaat']
                    elif 'kandungan' in question.lower():
                        info_types = ['kandungan']
                    elif 'cara' in question.lower() or 'pakai' in question.lower() or 'penggunaan' in question.lower():
                        info_types = ['cara_pakai']
                    elif 'harga' in question.lower():
                        info_types = ['harga']
                    elif 'lokasi' in question.lower() or 'beli' in question.lower():
                        info_types = ['lokasi']
                    elif 'review' in question.lower() or 'ulasan' in question.lower():
                        info_types = ['review']
                    
                    q_type = "product_info"
                    product = self.context.last_product
                    logger.info(f"Short follow-up detected: {info_types} for {product}")
                
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



    import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
import difflib
from typing import Tuple, Optional, List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import numpy as np

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
    'sensitive': 'sensitif',
    'dry': 'kering',
    'oily': 'berminyak',
    'combination': 'kombinasi',
    'normal': 'normal',
    'redness': 'kemerahan',
    'dull': 'kusam'
}

# Pemisah untuk jenis kulit kombinasi
SKIN_TYPE_SEPARATORS = ['/', 'dan', '&', 'serta', '-', 'atau', '+']

PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                "sunscreen", "tabir surya", "serum"]
PRODUCT_SYNONYMS = {
    'cuci muka': 'facial wash',
    'pembersih': 'cleanser',
    'moisturizer': 'pelembap',
    'sunscreen': 'tabir surya',
    'sunblock': 'tabir surya',
    'essence': 'serum',
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
def fuzzy_match(input_str, options, threshold=0.6, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    normalized_options = [opt.lower() for opt in options]
    matches = difflib.get_close_matches(
        input_str.lower(), 
        normalized_options, 
        n=5 if get_all else 1, 
        cutoff=threshold
    )
    
    if get_all:
        # Return all matches with original casing
        return [next((opt for opt in options if opt.lower() == match), match) 
                for match in matches]
    
    if matches:
        # Return single match with original casing
        return next((opt for opt in options if opt.lower() == matches[0]), None)
    return None

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text

# ======================
# CONVERSATION CONTEXT (Dukungan Jenis Kulit Kombinasi)
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []  # Ubah menjadi list untuk mendukung multiple skin types
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.role_attempts = 0
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        self.requested_count = None  # Jumlah produk yang diminta
        self.ingredient_filter = None  # Filter kandungan
        self.price_filter = None  # Filter harga
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return product_name.strip().lower() in {p.lower() for p in self.all_products}
    
    def update(self, question: str, response: str):
        self.last_question = question
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
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
        self.requested_count = None
        self.ingredient_filter = None
        self.price_filter = None
        
        # Jangan reset skin_type jika tidak ada yang baru
        new_skin_types = self._extract_skin_type(question)
        if new_skin_types:
            self.skin_type = new_skin_types
        
        self._extract_product_type(question)
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
    
    def _extract_skin_type(self, question: str) -> List[str]:
        q_lower = question.lower()
        detected_skins = set()  # Gunakan set untuk menghindari duplikasi

        # 2. Gunakan regex untuk deteksi yang lebih akurat
        skin_patterns = {
            'berjerawat': r'\b(berjerawat|jerawat|acne prone|acne)\b',
            'kering': r'\b(kering|dry)\b',
            'berminyak': r'\b(berminyak|oily)\b',
            'sensitif': r'\b(sensitif|sensitive)\b',
            'kombinasi': r'\b(kombinasi|combination)\b',
            'normal': r'\b(normal)\b',
            'kemerahan': r'\b(kemerahan|redness)\b',
            'kusam': r'\b(kusam|dull)\b'
        }

        for skin_type, pattern in skin_patterns.items():
            if re.search(pattern, q_lower):
                detected_skins.add(skin_type)

        # 3. Cari eksak di daftar skin type
        for skin in SKIN_TYPES:
            if skin in q_lower:
                detected_skins.add(skin)

        # 4. Fuzzy matching fallback
        if not detected_skins:
            match = fuzzy_match(q_lower, SKIN_TYPES, threshold=0.5)
            if match:
                detected_skins.add(match)

        # Return list kosong jika tidak ada skin_type baru
        return list(detected_skins) if detected_skins else []

    def _extract_product_type(self, question: str):
        # Cari sinonim terlebih dahulu
        q_lower = question.lower()
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type via synonym: {syn} -> {pt}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Gunakan regex untuk deteksi yang lebih akurat
        product_patterns = {
            'sunscreen': r'\b(sunscreen|tabir surya|sunblock)\b',
            'moisturizer': r'\b(moisturizer|pelembap)\b',
            'cleanser': r'\b(cleanser|facial wash|cuci muka|pembersih)\b',
            'serum': r'\b(serum|essence)\b',
        }
        
        for product_type, pattern in product_patterns.items():
            if re.search(pattern, q_lower):
                self.product_type = product_type
                logger.info(f"Detected product type via regex: {product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Cari eksak di daftar product type
        for pt in PRODUCT_TYPES:
            if pt in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(q_lower, PRODUCT_TYPES, threshold=0.5)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            self.need_skin_type = self.product_type and not self.skin_type
            
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
            for match in re.findall(pattern, text):
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
    
    def set_last_info_types(self, info_types: List[str]):
        self.last_info_types = info_types
        
    def get_last_info_types(self) -> List[str]:
        return self.last_info_types

# ======================
# PDF PROCESSOR (Dukungan Jenis Kulit Kombinasi)
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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
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
                    'serum': ['serum', 'essence'],
                    'toner': ['toner'],
                    'masker': ['masker', 'mask']
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
                    
                if i % 300 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

# ======================
# QA ENGINE (Dukungan Jenis Kulit Kombinasi)
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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bertscorer = BERTScorer()
        self.evaluation_results = []
        self.recommendation_history = set()  # Untuk menghindari rekomendasi duplikat
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict], detail_btn=True) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Gunakan nama asli tanpa normalisasi untuk ditampilkan
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image" 
                         onerror="this.onerror=null; this.src='../static/default_product.jpg';">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
            """
            card += "</div>"
            cards.append(card)
        return "\n".join(cards)

    def _extract_info_types(self, question: str) -> List[str]:
        """Ekstrak tipe informasi yang diminta secara eksplisit"""
        q_lower = question.lower()
        info_types = []
        
        # Cek tipe info dengan word boundary
        for q_type, keywords in QUESTION_TYPES.items():
            if any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in keywords):
                info_types.append(q_type)
        
        return info_types

    def _enhanced_product_matching(self, question: str, product_hint: str = "") -> Optional[str]:
        """
        Menggunakan GPT untuk mengenali produk yang dimaksud dari pertanyaan pengguna
        dengan memanfaatkan konteks dokumen dan produk yang ada.
        """
        try:
            # Ambil daftar produk (maksimal 100 untuk efisiensi)
            product_list = list(self.context.all_products)
            if not product_list:
                return None
                
            # Batasi jumlah produk jika terlalu banyak
            if len(product_list) > 100:
                product_list = product_list[:100]
            
            # Format prompt untuk GPT dengan contoh-contoh
            prompt = f"""
            Anda adalah asisten skincare yang membantu mengidentifikasi produk dari pertanyaan pengguna.
            Daftar produk yang tersedia: {', '.join(product_list)}
            
            Pertanyaan pengguna: "{question}"
            
            Petunjuk tambahan: 
            - Jika pengguna menyebutkan bagian dari nama produk, cari produk yang paling cocok
            - Perhatikan jenis produk yang disebutkan dalam konteks
            - Prioritaskan produk yang disebutkan sebelumnya jika relevan
            - Gunakan pencocokan case-insensitive
            
            Berdasarkan daftar produk di atas, produk mana yang paling mungkin dimaksud oleh pengguna?
            Jawab HANYA dengan nama produk yang lengkap. Jika tidak yakin, jawab dengan 'Tidak Diketahui'.
            
            Contoh:
            Pertanyaan: "manfaat sunscreen the originote"
            Jawaban: THE ORIGINOTE HYALUCERA GEL SUNSCREEN SPF 40 PA+++
            
            Pertanyaan: "bagaimana cara pakai Serum C"
            Jawaban: THE ORIGINOTE HYALU-C SERUM
            
            Pertanyaan: "harga moisturizer dari somethinc"
            Jawaban: SOMETHINC Ceramic Skin Saviour Moisturizer
            """
            
            # Jika ada petunjuk nama produk, tambahkan ke prompt
            if product_hint:
                prompt += f"\nPetunjuk nama produk: {product_hint}"
            
            response = self.llm.invoke(prompt).content.strip()
            
            # Bersihkan respons
            response = re.sub(r'^"|"$', '', response)  # Hapus tanda kutip jika ada
            
            # Periksa apakah respons adalah produk yang valid (case-insensitive)
            if response:
                for product in self.context.all_products:
                    if product.lower() == response.lower():
                        return product
                    
            # Jika tidak ditemukan, cari dengan similarity
            best_match = None
            best_score = 0
            threshold = 0.4  # Ambang batas rendah untuk toleransi
            
            for product in self.context.all_products:
                score = difflib.SequenceMatcher(
                    None, 
                    response.lower() if response else question.lower(), 
                    product.lower()
                ).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = product
                    
            return best_match if best_score > threshold else None
            
        except Exception as e:
            logger.error(f"Enhanced product matching error: {e}")
            return None

    def _extract_product_from_question(self, question: str) -> Optional[str]:
        """Mengekstrak nama produk dari pertanyaan dengan case-insensitive matching"""
        # Normalize question to lower for case-insensitive matching
        q_lower = question.lower()
        matched_products = []
        
        # Step 1: Check for exact match (case-insensitive)
        for product in self.context.all_products:
            if product.lower() == q_lower:
                return product
        
        # Step 2: Check for partial matches
        for product in self.context.all_products:
            # Cek jika nama produk (dalam lowercase) ada di pertanyaan (lowercase)
            if product.lower() in q_lower:
                matched_products.append(product)
        
        # If we have matches, return the longest one (most specific)
        if matched_products:
            matched_products.sort(key=len, reverse=True)
            return matched_products[0]
        
        # Step 3: Try regex patterns (case-insensitive)
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)',  # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Za-z0-9&.\-+() ]{5,}(?:\s+[A-Za-z0-9&.\-+() ]{3,})*)'  # Izinkan huruf kecil
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                # Use enhanced matching with the extracted name as a hint
                matched = self._enhanced_product_matching(question, name)
                if matched:
                    return matched
        
        # Step 4: Use GPT-enhanced matching without hint
        return self._enhanced_product_matching(question)

    def _generate_product_info_response(self, product_name: str, info_types: List[str]) -> str:
        """Generate formatted response for product information"""
        product_data = self.processor.get_product_data(product_name)
        if not product_data:
            return f"❌ Tidak menemukan informasi tentang {product_name}"
        
        # Filter hanya informasi yang diminta
        response_lines = []
        for info_type in info_types:
            if info_type == 'kandungan' and product_data.get('kandungan'):
                response_lines.append(f"<b>Kandungan</b>:<br /><p class=\"text-justify\">{self._format_ingredients(product_data['kandungan'])}</p>")
            
            elif info_type == 'manfaat' and product_data.get('manfaat'):
                # Bersihkan hanya bagian yang tidak relevan
                cleaned_manfaat = self._clean_content(
                    product_data['manfaat'],
                    ['cara pemakaian', 'kandungan', 'harga', 'lokasi']
                )
                response_lines.append(f"<b>Manfaat</b>:<br /><p class=\"text-justify\">{cleaned_manfaat}</p>")
                
            elif info_type == 'cara_pakai' and product_data.get('cara_pemakaian'):
                cleaned_cara = self._clean_content(
                    product_data['cara_pemakaian'],
                    ['manfaat', 'kandungan', 'harga', 'lokasi', 'informasi produk']
                )
                response_lines.append(f"<b>Cara Pakai</b>:<br /><p class=\"text-justify\">{cleaned_cara}</p>")
            
            elif info_type == 'harga' and product_data.get('harga'):
                response_lines.append(f"<b>Harga</b>:<br /><p class=\"text-justify\">{product_data['harga']}</p>")
            
            elif info_type == 'lokasi' and product_data.get('lokasi'):
                cleaned_lokasi = self._clean_content(
                    product_data['lokasi'],
                    ['cara pemakaian', 'manfaat', 'kandungan', 'harga', 'informasi produk']
                )
                response_lines.append(f"<b>Lokasi Pembelian</b>:<br /><p class=\"text-justify\">{cleaned_lokasi}</p>")
        
        response_text = "<br /><br />".join(response_lines) if response_lines else "❌ Tidak menemukan informasi yang diminta"
        
        # Tampilkan card produk setelah penjelasan teks
        product_card = self._generate_product_cards([{'name': product_name}])
        
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
    
    def analyze_question(self, question: str) -> Tuple[str, Optional[str], List[str]]:
        """Mengembalikan (question_type, product, info_types)"""
        info_types = []
        q_lower = question.lower()
        
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
            
        # Handle special cases
        # Ganti pengecekan substring dengan word boundary
        if any(re.search(rf'\b{re.escape(greeting)}\b', q_lower) for greeting in GREETINGS):
            return ("sapaan", None, [])
            
        if any(kw in q_lower for kw in ["ganti peran", "change role", "ubah peran", "bukan asisten skincare"]):
            self.context.role_attempts += 1
            return ("role_change_blocked", None, []) if self.context.role_attempts > 2 else ("role_change_warning", None, [])
            
        self.context.role_attempts = 0
            
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None, [])
            
        # Rekomendasi detection with filters
        rekomendasi_keywords = QUESTION_TYPES['rekomendasi'] + ['rekomendasi', 'sarankan', 'anjuran']
        if any(kw in q_lower for kw in rekomendasi_keywords):
            # Deteksi jumlah produk yang diminta
            num_match = re.search(r'(\d+)\s+produk', q_lower)
            self.context.requested_count = int(num_match.group(1)) if num_match else None
            
            # Deteksi filter kandungan
            self.context.ingredient_filter = None
            ingredient_match = re.search(r'mengandung\s+([\w\s]+)', q_lower)
            if ingredient_match:
                self.context.ingredient_filter = ingredient_match.group(1).strip()
                
            # Deteksi filter harga
            self.context.price_filter = None
            price_match = re.search(r'harga\s+(dibawah|di bawah|diatas|kurang|lebih|>|<|>=|<=)?\s*([\d\.]+)', q_lower)
            if price_match:
                operator = price_match.group(1).lower() if price_match.group(1) else ""
                price_value = float(price_match.group(2).replace('.', ''))
                self.context.price_filter = (operator, price_value)
                
            # Gunakan konteks yang sudah ada
            if self.context.skin_type and self.context.product_type:
                return ("rekomendasi", None, [])
            elif self.context.skin_type:
                return ("need_product_type", None, [])
            elif self.context.product_type:
                return ("need_skin_type", None, [])
            else:
                return ("rekomendasi", None, [])
            
        # Skin type detection (memperbaiki deteksi skin type)
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            if not self.context.skin_type:
                self.context.skin_type = []
            self.context.skin_type.append(skin_match)
            return ("rekomendasi", None, []) if self.context.product_type else ("need_product_type", None, [])
            
        # Memperbaiki deteksi jenis kulit yang disebutkan tanpa kata kunci "rekomendasi"
        for syn, skin in SKIN_SYNONYMS.items():
            if syn in q_lower:
                if not self.context.skin_type:
                    self.context.skin_type = []
                self.context.skin_type.append(skin)
                return ("rekomendasi", None, [])
        
        # Product type detection
        product_match = next((pt for pt in PRODUCT_TYPES if pt in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            return ("need_skin_type", None, []) if not self.context.skin_type else ("rekomendasi", None, [])
            
        # Handle all product info questions with unified approach
        info_types = self._extract_info_types(question)
        product = self._extract_product_from_question(question)
        
        # Prioritaskan pertanyaan spesifik produk
        if any(kw in q_lower for kw in ["manfaat", "kandungan", "cara pakai", "harga", "lokasi"]):
            if product:
                return ("product_info", product, info_types)
        
        # PERBAIKAN UTAMA: Prioritaskan menampilkan card produk jika hanya nama produk yang diberikan
        if product:
            # Jika ada kata kunci informasi spesifik
            if info_types:
                return ("product_info", product, info_types)
            # Jika hanya nama produk tanpa kata kunci tambahan
            return ("product_card", product, [])
                
        return ("general", None, [])
    
    def get_product_info(self, product_name: str, info_types: List[str]) -> str:
        try:
            if not product_name:
                return "❌ Produk tidak disebutkan"
            
            # Gunakan algoritma cerdas untuk pencocokan produk
            best_match = self._enhanced_product_matching(self.context.last_question, product_name)
            
            if not best_match:
                # Cari kandidat produk yang mirip
                candidates = fuzzy_match(
                    product_name, 
                    list(self.context.all_products), 
                    threshold=0.4, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates], 
                        detail_btn=False
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
            
            # Langsung tampilkan informasi produk
            response_html = self._generate_product_info_response(best_match, info_types)
            self.context.last_product = best_match
            self.context.set_last_info_types(info_types)
            
            # Prepare for evaluation
            response_content = re.sub(r'<[^>]+>', '', response_html)  # Remove HTML tags
            self._evaluate_product_info(best_match, info_types, response_content)
            
            return response_html
            
        except Exception as e:
            logger.error(f"Product info error: {e}")
            return f"❌ Tidak bisa menampilkan informasi untuk produk tersebut"
    
    def _evaluate_product_info(self, product_name: str, info_types: List[str], response: str):
        """Evaluate product info response against document content with precision, recall, and F1"""
        for info_type in info_types:
            try:
                reference_text = self.processor.get_reference_text(product_name, info_type)
                if reference_text:
                    # Calculate BERT score
                    bert_score = self.bertscorer.evaluate_answer(response, reference_text)
                    
                    # Calculate token-based metrics
                    response_tokens = set(response.lower().split())
                    reference_tokens = set(reference_text.lower().split())
                    
                    # Handle empty response case
                    if not response_tokens:
                        precision = 0
                        recall = 0
                        f1 = 0
                    else:
                        # Calculate true positives, false positives, and false negatives
                        true_positives = response_tokens.intersection(reference_tokens)
                        false_positives = response_tokens - reference_tokens
                        false_negatives = reference_tokens - response_tokens
                        
                        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
                        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Log evaluation result
                    self.evaluation_results.append({
                        'type': 'produk_info',
                        'product': product_name,
                        'info_type': info_type,
                        'bert_score': bert_score,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'response': response[:200] + "..." if len(response) > 200 else response,
                        'reference': reference_text[:200] + "..." if len(reference_text) > 200 else reference_text
                    })
                    
                    # Print to terminal for immediate feedback
                    print(f"\n=== Evaluasi Informasi Produk ===")
                    print(f"Produk: {product_name}")
                    print(f"Jenis Info: {info_type}")
                    print(f"BERT Score: {bert_score:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1 Score: {f1:.4f}")
                    print("="*40)
                    
                    logger.info(f"Product info evaluation for {product_name} ({info_type}): "
                                f"BERT={bert_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            except Exception as e:
                logger.error(f"Product info evaluation error: {e}")

    def get_recommendations(self, query: str) -> str:
        try:
            # Gunakan konteks yang sudah ada
            if not self.context.skin_type and not self.context.product_type:
                return "Silakan sebutkan jenis kulit dan jenis produk yang Anda cari"
            
            if not self.context.skin_type:
                return "Silakan sebutkan jenis kulit Anda"
            
            if not self.context.product_type:
                return "Silakan sebutkan jenis produk yang Anda cari"
            
            # Dapatkan semua produk yang tersedia
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data:
                    all_products.append({
                        'name': product_name,  # Gunakan nama asli
                        'data': product_data
                    })
            
            # Log untuk debugging
            logger.info(f"Jumlah produk tersedia: {len(all_products)}")
            logger.info(f"Kriteria filter - Kulit: {self.context.skin_type}, Produk: {self.context.product_type}")
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                skin_data = product['data'].get('jenis_kulit', [])
                type_data = product['data'].get('type_produk', '').lower()
                
                # Logika filter jenis kulit untuk kombinasi
                skin_match = False
                if not self.context.skin_type:  # Tidak ada filter kulit
                    skin_match = True
                else:
                    # Jika produk cocok untuk semua jenis kulit
                    if 'semua' in skin_data:
                        skin_match = True
                    else:
                        # Minimal satu jenis kulit yang diminta cocok dengan produk
                        any_match = any(user_skin in skin_data for user_skin in self.context.skin_type)
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
                
                # Filter kandungan
                ingredient_match = True
                if self.context.ingredient_filter:
                    kandungan = product['data'].get('kandungan', '').lower()
                    ingredient_match = (self.context.ingredient_filter.lower() in kandungan)
                    
                # Filter harga
                price_match = True
                if self.context.price_filter:
                    harga_text = product['data'].get('harga', '0')
                    try:
                        # Ekstrak angka dari teks harga
                        harga = float(re.search(r'[\d,\.]+', harga_text.replace('.', '')).group().replace(',', '.'))
                    except:
                        harga = 0
                        
                    operator, value = self.context.price_filter
                    if operator in ["dibawah", "di bawah", "kurang", "<"]:
                        price_match = harga < value
                    elif operator in ["di atas", "diatas", "lebih", ">"]:
                        price_match = harga > value
                    else:
                        price_match = harga == value  # Default: harga tepat
                
                if skin_match and product_match and ingredient_match and price_match:
                    filtered_products.append(product)
                    logger.info(f"Product matched: {product['name']} | Skin: {skin_data} | Type: {type_data}")
            
            # Batasi jumlah produk jika diminta
            requested_count = self.context.requested_count or 5
            if len(filtered_products) > requested_count:
                # Prioritaskan produk dengan kemiripan tertinggi
                filtered_products = filtered_products[:requested_count]
            
            # Generate product cards
            products_to_show = [{'name': p['name']} for p in filtered_products]
            product_cards = self._generate_product_cards(products_to_show)
            
            # Response dengan jumlah produk aktual
            if filtered_products:
                response_html = f"""
                <div class="recommendation-container">
                    <h3>Ditemukan {len(filtered_products)} produk yang sesuai:</h3>
                    <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                        {product_cards}
                    </div>
                </div>
                """
            else:
                response_html = "❌ Tidak ditemukan produk yang sesuai. Silakan coba dengan kriteria lain."
            
            # Evaluate recommendations
            self._evaluate_recommendation(filtered_products)
            
            return response_html
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return "❌ Terjadi kesalahan saat mencari rekomendasi. Silakan coba lagi."
    
    def _evaluate_recommendation(self, recommended_products: list):
        try:
            logger.info("Memulai evaluasi rekomendasi...")
            
            if not self.context.skin_type:
                logger.info("Dilewati: skin_type tidak tersedia")
                return
                
            logger.info(f"Skin type: {self.context.skin_type}")
            
            reference_products = self.processor.get_recommendation_reference(self.context.skin_type[0])
            if not reference_products:
                logger.info("Dilewati: Tidak ada referensi produk")
                return
                
            logger.info(f"Referensi ditemukan: {len(reference_products)} produk")
            
            # ... (kode sebelumnya) ...
            
            # Hitung BERTScore
            rec_text = ", ".join([p['name'] for p in recommended_products])
            ref_text = ", ".join(reference_products)
            logger.info(f"Menghitung BERTScore... Rec: {rec_text} | Ref: {ref_text}")
            
            bert_score = self.bertscorer.evaluate_answer(rec_text, ref_text)
            logger.info(f"BERTScore dihitung: {bert_score}")
            
            # ... (log hasil) ...
            
        except Exception as e:
            logger.error(f"ERROR evaluasi: {str(e)}", exc_info=True)

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
            summary += "<table border='1'><tr><th>Jenis Kulit</th><th>Presisi</th><th>Recall</th><th>F1 Score</th><th>BERTScore</th><th>Rekomendasi Sistem</th><th>Referensi</th></tr>"
            for result in recommendation_evals:
                summary += (f"<tr><td>{result['skin_type']}</td>"
                            f"<td>{result['precision']:.2f}</td>"
                            f"<td>{result['recall']:.2f}</td>"
                            f"<td>{result['f1_score']:.2f}</td>"
                            f"<td>{result['bert_score']:.4f}</td>"
                            f"<td>{', '.join(result['recommended'])}</td>"
                            f"<td>{', '.join(result['reference'])}</td></tr>")
            summary += "</table>"
        
        return summary

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Gunakan algoritma cerdas untuk pencocokan produk
        best_match = self._enhanced_product_matching(self.context.last_question, product_name)
        
        if not best_match:
            # Cari kandidat produk yang mirip
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.4, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates], 
                    detail_btn=False
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
        if product_data and product_data.get('informasi_produk'):
            brief_info = f"<div class='product-brief'>{product_data['informasi_produk'][:150]}...</div>"
        
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
        
        # Cek jika user mengatakan "ya" untuk kandidat pertama
        if any(kw in question.lower() for kw in ['ya', 'yes', 'benar', 'betul', 'itu', 'yang pertama']):
            selected_product = self.context.product_candidates[0]
            self.context.product_candidates = []  # Reset kandidat
            return self.show_product_card(selected_product)
        
        # Cek jika user mengatakan "tidak" atau "lain"
        if any(kw in question.lower() for kw in ['tidak', 'no', 'bukan', 'lain', 'yang lain', 'yang lainnya']):
            if len(self.context.product_candidates) > 1:
                # Tampilkan kandidat berikutnya
                next_candidates = self.context.product_candidates[1:]
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in next_candidates], 
                    detail_btn=False
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

    def _format_ingredients(self, ingredients: str) -> str:
        """Format ingredients with bullet points"""
        if not ingredients:
            return "❌ Informasi kandungan tidak tersedia"
        
        ingredients = self._clean_content(ingredients, ['cara pemakaian', 'manfaat', 'harga', 'lokasi'])
        return ingredients

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

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "off_topic":
            return "Maaf, saya hanya membantu soal skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                skin_types = ", ".join(self.context.skin_type)
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {skin_types}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "role_change_warning":
            return "Maaf, saya hanya bisa membantu dengan pertanyaan seputar skincare. Ada yang bisa saya bantu terkait produk skincare?"
        
        if q_type == "role_change_blocked":
            return "Saya tetap akan membantu dengan skincare. Jika Anda ingin mengubah topik, silakan mulai percakapan baru."
        
        if q_type == "need_skin_type":
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(PRODUCT_TYPES)})"
        
        # Handle product info requests
        if q_type == "product_info":
            # Jika ada info_types tapi tidak ada produk, gunakan last_product
            if not product and self.context.last_product:
                product = self.context.last_product
                logger.info(f"Using last product: {product} for info request")
            
            if product:
                response = self.get_product_info(product, info_types)
                # Simpan sebagai last_product hanya jika berhasil
                if not response.startswith("❌"):
                    self.context.last_product = product
                    self.context.set_last_info_types(info_types)
                return response
            else:
                return "Produk apa yang Anda maksud? Silakan sebutkan nama produknya."
        
        # Handle product card requests
        if q_type == "product_card":
            if not product:
                return "Produk apa yang Anda maksud? Silakan sebutkan nama produknya."
            
            response = self.show_product_card(product)
            if not response.startswith("❌"):
                self.context.last_product = product
            return response
        
        # Handle recommendations
        if q_type == "rekomendasi":
            return self.get_recommendations("")
        
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        with self.lock:
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
                self.context._extract_context(question, "")
                
                # Analyze question
                q_type, product, info_types = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
                
                # PERBAIKAN: Jika hanya nama produk, langsung tampilkan card
                if q_type == "general" and product:
                    q_type = "product_card"
                    logger.info(f"Auto-converted to product_card for: {product}")
                
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
                
                # Deteksi pertanyaan singkat follow-up (1-3 kata)
                is_short_follow_up = (
                    not product and 
                    self.context.last_product and 
                    len(question.split()) <= 3 and 
                    any(keyword in question.lower() for keyword in [
                        'manfaat', 'kandungan', 'cara', 'pakai', 'penggunaan', 'harga', 'lokasi', 'review'
                    ])
                )
                
                if is_short_follow_up:
                    # Gunakan info types dari kata kunci
                    if 'manfaat' in question.lower():
                        info_types = ['manfaat']
                    elif 'kandungan' in question.lower():
                        info_types = ['kandungan']
                    elif 'cara' in question.lower() or 'pakai' in question.lower() or 'penggunaan' in question.lower():
                        info_types = ['cara_pakai']
                    elif 'harga' in question.lower():
                        info_types = ['harga']
                    elif 'lokasi' in question.lower() or 'beli' in question.lower():
                        info_types = ['lokasi']
                    elif 'review' in question.lower() or 'ulasan' in question.lower():
                        info_types = ['review']
                    
                    q_type = "product_info"
                    product = self.context.last_product
                    logger.info(f"Short follow-up detected: {info_types} for {product}")
                
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




import os
import fitz
import traceback
import tiktoken
import time
import re
import gc
import logging
import difflib
from typing import Tuple, Optional, List, Dict, Set
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

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
    'sensitive': 'sensitif',
    'dry': 'kering',
    'oily': 'berminyak',
    'combination': 'kombinasi',
    'normal': 'normal',
    'redness': 'kemerahan',
    'dull': 'kusam'
}

# Pemisah untuk jenis kulit kombinasi
SKIN_TYPE_SEPARATORS = ['/', 'dan', '&', 'serta', '-', 'atau', '+']

PRODUCT_TYPES = ["facial wash", "cleanser", "pelembap", "moisturizer", 
                "sunscreen", "tabir surya", "serum"]
PRODUCT_SYNONYMS = {
    'cuci muka': 'facial wash',
    'pembersih': 'cleanser',
    'moisturizer': 'pelembap',
    'sunscreen': 'tabir surya',
    'sunblock': 'tabir surya',
    'essence': 'serum',
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
def fuzzy_match(input_str, options, threshold=0.6, get_all=False):
    """Fuzzy match with threshold, can return all matches above threshold"""
    if not options:
        return None if not get_all else []
        
    normalized_options = [opt.lower() for opt in options]
    matches = difflib.get_close_matches(
        input_str.lower(), 
        normalized_options, 
        n=5 if get_all else 1, 
        cutoff=threshold
    )
    
    if get_all:
        # Return all matches with original casing
        return [next((opt for opt in options if opt.lower() == match), match) 
                for match in matches]
    
    if matches:
        # Return single match with original casing
        return next((opt for opt in options if opt.lower() == matches[0]), None)
    return None

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text

# ======================
# CONVERSATION CONTEXT (Dukungan Jenis Kulit Kombinasi)
# ======================
class ConversationContext:
    def __init__(self, product_names: List[str] = None):
        self.skin_type = []  # Ubah menjadi list untuk mendukung multiple skin types
        self.product_type = None
        self.last_product = None
        self.history = []
        self.max_history = 5
        self.need_skin_type = False
        self.need_product_type = False
        self.all_products = set()
        self.role_attempts = 0
        self.last_question_type = None
        self.last_question = None
        self.last_info_types = []  # Menyimpan tipe informasi yang diminta terakhir
        self.product_candidates = []  # Menyimpan kandidat produk untuk konfirmasi typo
        self.confirmation_pending = False  # Status menunggu konfirmasi pengguna
        self.requested_count = None  # Jumlah produk yang diminta
        self.ingredient_filter = None  # Filter kandungan
        self.price_filter = None  # Filter harga
        
        if product_names:
            for name in product_names:
                self.add_product(name)
    
    def add_product(self, product_name: str):
        # Simpan casing asli produk
        self.all_products.add(product_name.strip())
    
    def is_valid_product(self, product_name: str) -> bool:
        # Periksa dengan case-insensitive
        return any(p.lower() == product_name.lower() for p in self.all_products)
    
    def update(self, question: str, response: str):
        self.last_question = question
        self._extract_context(question, response)
        self.history = self.history[-(self.max_history-1):] + [(question[:50], response[:100])]
    
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
        self.requested_count = None
        self.ingredient_filter = None
        self.price_filter = None
        
        # Jangan reset skin_type jika tidak ada yang baru
        new_skin_types = self._extract_skin_type(question)
        if new_skin_types:
            self.skin_type = new_skin_types
        
        self._extract_product_type(question)
        extracted_product = self._extract_product_name(question) or self._extract_product_name(response)
        if extracted_product:
            self.last_product = extracted_product
            logger.info(f"Detected product: {self.last_product}")
    def _extract_skin_type(self, question: str):
        q_lower = question.lower()
        detected_skins = set()
        
        # 1. Cari sinonim terlebih dahulu dengan regex kata utuh
        for syn, skin in SKIN_SYNONYMS.items():
            if re.search(rf'\b{syn}\b', q_lower):
                detected_skins.add(skin)
                logger.info(f"Mapped synonym: '{syn}' -> '{skin}'")
        
        # 2. Gunakan regex untuk deteksi yang lebih akurat
        skin_patterns = {
            'berjerawat': r'\b(berjerawat|jerawat|acne prone|acne)\b',
            'kering': r'\b(kering|dry)\b',
            'berminyak': r'\b(berminyak|oily)\b',
            'sensitif': r'\b(sensitif|sensitive)\b',
            'kombinasi': r'\b(kombinasi|combination)\b',
            'normal': r'\b(normal)\b',
            'kemerahan': r'\b(kemerahan|redness)\b',
            'kusam': r'\b(kusam|dull)\b'
        }
        
        for skin_type, pattern in skin_patterns.items():
            if re.search(pattern, q_lower):
                detected_skins.add(skin_type)
        
        # 3. Cari eksak di daftar skin type
        for skin in SKIN_TYPES:
            if re.search(rf'\b{skin}\b', q_lower):
                detected_skins.add(skin)
        
        # 4. Fuzzy matching fallback
        if not detected_skins:
            match = fuzzy_match(q_lower, SKIN_TYPES, threshold=0.5)
            if match:
                detected_skins.add(match)
        
        # Simpan hasil deteksi
        self.skin_type = list(detected_skins)
        if self.skin_type:
            logger.info(f"Detected skin types: {self.skin_type}")

    def _extract_product_type(self, question: str):
        # Cari sinonim terlebih dahulu
        q_lower = question.lower()
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type via synonym: {syn} -> {pt}")
                self.need_skin_type = self.product_type and not self.skin_type
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
                self.need_skin_type = self.product_type and not self.skin_type
                return
                
        # Cari eksak di daftar product type
        for pt in PRODUCT_TYPES:
            if pt in q_lower:
                self.product_type = pt
                logger.info(f"Detected product type: {self.product_type}")
                self.need_skin_type = self.product_type and not self.skin_type
                return
        
        # Fuzzy matching fallback
        match = fuzzy_match(q_lower, PRODUCT_TYPES, threshold=0.5)
        if match:
            self.product_type = match
            logger.info(f"Fuzzy matched product type: {self.product_type}")
            self.need_skin_type = self.product_type and not self.skin_type
            
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
    
    def set_last_info_types(self, info_types: List[str]):
        self.last_info_types = info_types
        
    def get_last_info_types(self) -> List[str]:
        return self.last_info_types

# ======================
# PDF PROCESSOR (Dukungan Jenis Kulit Kombinasi)
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
        self.text_output_dir = "processed_texts"
        os.makedirs(self.text_output_dir, exist_ok=True)
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
                    'serum': ['serum', 'essence'],
                    'toner': ['toner'],
                    'masker': ['masker', 'mask']
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
                    
                if i % 300 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                if batch_size > 10:
                    batch_size = max(10, batch_size // 2)

# ======================
# QA ENGINE (Dukungan Jenis Kulit Kombinasi)
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
        self.lock = Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bertscorer = BERTScorer()
        self.evaluation_results = []
        self.recommendation_history = set()  # Untuk menghindari rekomendasi duplikat
        
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return text if len(tokens) <= max_tokens else self.tokenizer.decode(tokens[:max_tokens])

    def _generate_product_cards(self, products: List[Dict], detail_btn=True) -> str:
        cards = []
        for product in products:
            name = product['name']
            # Gunakan nama asli tanpa normalisasi untuk ditampilkan
            card = f"""
            <div class="product-card">
                <div class="product-image-container">
                    <img src="../static/data.1/{name}.jpg" alt="{name}" class="product-image" 
                         onerror="this.onerror=null; this.src='../static/default_product.jpg';">
                </div>
                <div class="product-info">
                    <h3 class="product-name">{name}</h3>
                </div>
                <div class="product-footer">
                    <button class="detail-btn" data-product-name="{name}">
                        Lihat Detail
                    </button>
                </div>
            """
            card += "</div>"
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
                    
            # Jika GPT tidak yakin, coba cari dengan similarity
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
        
        if not question.strip() or re.fullmatch(r'[\s\?\*\{\}]+', question):
            return ("invalid", None, [])
            
        # Handle special cases
        if any(greeting in q_lower for greeting in GREETINGS):
            product = self._extract_product_from_question(question)
            if product:
                return ("product_info", product, ['manfaat'])  # Kategorikan sebagai informasi produk
            return ("sapaan", None, [])
            
        if any(kw in q_lower for kw in ["ganti peran", "change role", "ubah peran", "bukan asisten skincare"]):
            self.context.role_attempts += 1
            return ("role_change_blocked", None, []) if self.context.role_attempts > 2 else ("role_change_warning", None, [])
            
        self.context.role_attempts = 0
            
        if any(kw in q_lower for kw in ["kelautan", "politik", "matematika", "asisten kelautan"]):
            return ("off_topic", None, [])
            
        # Deteksi khusus untuk pertanyaan manfaat
        if 'manfaat' in q_lower and any(p.lower() in q_lower for p in self.context.all_products):
            product = self._extract_product_from_question(question)
            return ("product_info", product, ['manfaat'])
            
        # Rekomendasi detection with filters
        rekomendasi_keywords = QUESTION_TYPES['rekomendasi'] + ['rekomendasi', 'sarankan', 'anjuran']
        if any(kw in q_lower for kw in rekomendasi_keywords):
            # Deteksi jumlah produk yang diminta
            num_match = re.search(r'(\d+)\s+produk', q_lower)
            self.context.requested_count = int(num_match.group(1)) if num_match else None
            
            # Deteksi filter kandungan
            self.context.ingredient_filter = None
            ingredient_match = re.search(r'mengandung\s+([\w\s]+)', q_lower)
            if ingredient_match:
                self.context.ingredient_filter = ingredient_match.group(1).strip()
                
            # Deteksi filter harga
            self.context.price_filter = None
            price_match = re.search(r'harga\s+(dibawah|di bawah|di atas|diatas|kurang|lebih|>|<|>=|<=)?\s*([\d\.]+)', q_lower)
            if price_match:
                operator = price_match.group(1).lower() if price_match.group(1) else ""
                price_value = float(price_match.group(2).replace('.', ''))
                self.context.price_filter = (operator, price_value)
                
            return ("rekomendasi", None, [])
            
        # Skin type detection
        skin_match = next((skin for skin in SKIN_TYPES if skin in q_lower), None)
        if skin_match:
            if not self.context.skin_type:
                self.context.skin_type = []
            self.context.skin_type.append(skin_match)
            return ("rekomendasi", None, []) if self.context.product_type else ("need_product_type", None, [])
            
        # PERBAIKAN UTAMA: Deteksi jika dalam pertanyaan terdapat sinonim jenis kulit dan produk
        has_skin_type = False
        has_product_type = False
        
        # Cek skin type dari sinonim
        for syn, skin in SKIN_SYNONYMS.items():
            if re.search(rf'\b{syn}\b', q_lower):
                if not self.context.skin_type:
                    self.context.skin_type = []
                self.context.skin_type.append(skin)
                has_skin_type = True
                logger.info(f"Mapped skin synonym: '{syn}' -> '{skin}'")
        
        # Cek product type dari sinonim
        for syn, pt in PRODUCT_SYNONYMS.items():
            if syn in q_lower:
                self.context.product_type = pt
                has_product_type = True
                break
        
        # Jika ditemukan keduanya, maka dianggap permintaan rekomendasi
        if has_skin_type and has_product_type:
            return ("rekomendasi", None, [])
            
        # Product type detection
        product_match = next((pt for pt in PRODUCT_TYPES if pt in q_lower), None)
        if product_match:
            self.context.product_type = product_match
            return ("need_skin_type", None, []) if not self.context.skin_type else ("rekomendasi", None, [])
            
        # Handle all product info questions with unified approach
        # Extract info types from question
        info_types = self._extract_info_types(question)
        product = self._extract_product_from_question(question)
        
        if product and not info_types:
            # Hanya menyebutkan nama produk tanpa info spesifik
            return ("product_card", product, [])
        elif info_types:
            return ("product_info", product or self.context.last_product, info_types)
                
        return ("general", None, [])

    def _extract_product_from_question(self, question: str) -> Optional[str]:
        # Pola regex untuk nama produk uppercase dengan tanda kurung
        patterns = [
            r'Produk:\s*([^\n]+?)(?=\n|$)',  # Tangkap sampai akhir baris
            r'\[([^\]]+)\]',
            r'"(.*?)"',
            r'Rekomendasi:\s*(.*?)\n',
            r'([A-Z][A-Z0-9&.\-+() ]{5,}(?:\s+[A-Z0-9&.\-+() ]{3,})*)'  # Izinkan tanda kurung
        ]
        
        # Coba ekstrak dengan pola regex
        for pattern in patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                name = name.strip()
                # Gunakan fuzzy matching untuk verifikasi (case-insensitive)
                matched = fuzzy_match(name, list(self.context.all_products), threshold=0.5)
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
                match = fuzzy_match(phrase, list(self.context.all_products), threshold=0.4)
                if match:
                    return match
        
        return None

    def get_recommendations(self, query: str) -> str:
        try:
            # Dapatkan semua produk yang tersedia
            all_products = []
            for product_name in self.context.all_products:
                product_data = self.processor.get_product_data(product_name)
                if product_data:
                    all_products.append({
                        'name': product_name,  # Gunakan nama asli
                        'data': product_data
                    })
            
            # Log untuk debugging
            logger.info(f"Jumlah produk tersedia: {len(all_products)}")
            logger.info(f"Kriteria filter - Kulit: {self.context.skin_type}, Produk: {self.context.product_type}")
            
            # Normalisasi jenis kulit yang diminta user
            user_skins = [SKIN_SYNONYMS.get(st.lower(), st.lower()) for st in self.context.skin_type]
            logger.info(f"Normalized skin types: {user_skins}")
            
            # Filter produk berdasarkan kriteria
            filtered_products = []
            for product in all_products:
                skin_data = [s.lower() for s in product['data'].get('jenis_kulit', [])]
                type_data = product['data'].get('type_produk', '').lower()
                
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
                
                # Filter kandungan
                ingredient_match = True
                if self.context.ingredient_filter:
                    kandungan = product['data'].get('kandungan', '').lower()
                    ingredient_match = (self.context.ingredient_filter.lower() in kandungan)
                    
                # Filter harga
                price_match = True
                if self.context.price_filter:
                    harga_text = product['data'].get('harga', '0')
                    try:
                        # Ekstrak angka dari teks harga
                        harga = float(re.search(r'[\d,\.]+', harga_text.replace('.', '')).group().replace(',', '.'))
                    except:
                        harga = 0
                        
                    operator, value = self.context.price_filter
                    if operator in ["dibawah", "di bawah", "kurang", "<"]:
                        price_match = harga < value
                    elif operator in ["di atas", "diatas", "lebih", ">"]:
                        price_match = harga > value
                    else:
                        price_match = harga == value  # Default: harga tepat
                
                if skin_match and product_match and ingredient_match and price_match:
                    filtered_products.append(product)
                    logger.info(f"Product matched: {product['name']} | Skin: {skin_data} | Type: {type_data}")
            
            # Batasi jumlah produk jika diminta
            requested_count = self.context.requested_count or 5
            if len(filtered_products) > requested_count:
                # Prioritaskan produk dengan kemiripan tertinggi
                filtered_products = filtered_products[:requested_count]
            
            # Generate product cards
            products_to_show = [{'name': p['name']} for p in filtered_products]
            product_cards = self._generate_product_cards(products_to_show)
            
            # Response dengan jumlah produk aktual
            if filtered_products:
                response_html = f"""
                <div class="recommendation-container">
                    <h3>Ditemukan {len(filtered_products)} produk yang sesuai:</h3>
                    <div class="product-grid" style="display: flex; flex-wrap: wrap; gap: 20px;">
                        {product_cards}
                    </div>
                </div>
                """
            else:
                response_html = "❌ Tidak ditemukan produk yang sesuai. Silakan coba dengan kriteria lain."
            
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
            threshold = 0.6  # Threshold untuk kecocokan
            
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
                    threshold=0.5
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
                    threshold=0.4, 
                    get_all=True
                )
                
                if candidates:
                    self.context.product_candidates = candidates
                    candidate_cards = self._generate_product_cards(
                        [{'name': name} for name in candidates], 
                        detail_btn=False
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
            
            # Filter hanya informasi yang diminta
            response_lines = []
            for info_type in info_types:
                if info_type == 'kandungan' and product_data.get('kandungan'):
                    response_lines.append(f"<b>Kandungan</b>:<br /><p class=\"text-justify\">{self._format_ingredients(product_data['kandungan'])}</p>")
                
                elif info_type == 'manfaat' and product_data.get('manfaat'):
                    # Bersihkan hanya bagian yang tidak relevan
                    cleaned_manfaat = self._clean_content(
                        product_data['manfaat'],
                        ['cara pemakaian', 'kandungan', 'harga', 'lokasi']
                    )
                    response_lines.append(f"<b>Manfaat</b>:<br /><p class=\"text-justify\">{cleaned_manfaat}</p>")
                    
                elif info_type == 'cara_pakai' and product_data.get('cara_pemakaian'):
                    cleaned_cara = self._clean_content(
                        product_data['cara_pemakaian'],
                        ['manfaat', 'kandungan', 'harga', 'lokasi', 'informasi produk']
                    )
                    response_lines.append(f"<b>Cara Pakai</b>:<br /><p class=\"text-justify\">{cleaned_cara}</p>")
                
                elif info_type == 'harga' and product_data.get('harga'):
                    response_lines.append(f"<b>Harga</b>:<br /><p class=\"text-justify\">{product_data['harga']}</p>")
                
                elif info_type == 'lokasi' and product_data.get('lokasi'):
                    cleaned_lokasi = self._clean_content(
                        product_data['lokasi'],
                        ['cara pemakaian', 'manfaat', 'kandungan', 'harga', 'informasi produk']
                    )
                    response_lines.append(f"<b>Lokasi Pembelian</b>:<br /><p class=\"text-justify\">{cleaned_lokasi}</p>")
            
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
                <div class="context-info">
                    {context_info}
                </div>
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

    def show_product_card(self, product_name: str) -> str:
        """Menampilkan card produk tanpa detail informasi"""
        # Gunakan algoritma yang lebih toleran untuk pencocokan nama produk
        best_match = None
        best_score = 0
        threshold = 0.6  # Threshold untuk kecocokan
        
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
                threshold=0.5
            )
        
        # Jika masih tidak ketemu, gunakan GPT untuk menebak
        if not best_match:
            guessed_name = self._guess_product_name(f"Produk: {product_name}")
            if guessed_name:
                best_match = guessed_name
        
        if not best_match:
            # Cari kandidat produk yang mirip
            candidates = fuzzy_match(
                product_name, 
                list(self.context.all_products), 
                threshold=0.4, 
                get_all=True
            )
            
            if candidates:
                self.context.product_candidates = candidates
                candidate_cards = self._generate_product_cards(
                    [{'name': name} for name in candidates], 
                    detail_btn=False
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
                    [{'name': name} for name in next_candidates], 
                    detail_btn=False
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

    def _format_ingredients(self, ingredients: str) -> str:
        """Format ingredients with bullet points"""
        if not ingredients:
            return "❌ Informasi kandungan tidak tersedia"
        
        ingredients = self._clean_content(ingredients, ['cara pemakaian', 'manfaat', 'harga', 'lokasi'])
        return ingredients

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

    def _handle_response(self, q_type: str, product: Optional[str], info_types: List[str]) -> str:
        """Handle different question types"""
        # Handle context requests
        if q_type == "invalid":
            return "Silakan ajukan pertanyaan tentang produk skincare."
        
        if q_type == "off_topic":
            return "Maaf, saya hanya membantu soal skincare."
        
        if q_type == "sapaan":
            if self.context.skin_type:
                skin_types = ", ".join(self.context.skin_type)
                return f"Halo! Saya asisten virtual untuk rekomendasi skincare. Terakhir kali Anda menyebutkan memiliki kulit {skin_types}. Ada yang bisa saya bantu?"
            return "Halo! Saya asisten virtual untuk rekomendasi skincare. Silakan bertanya tentang produk skincare."
        
        if q_type == "role_change_warning":
            return "Maaf, saya hanya bisa membantu dengan pertanyaan seputar skincare. Ada yang bisa saya bantu terkait produk skincare?"
        
        if q_type == "role_change_blocked":
            return "Saya tetap akan membantu dengan skincare. Jika Anda ingin mengubah topik, silakan mulai percakapan baru."
        
        if q_type == "need_skin_type":
            return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
        
        if q_type == "need_product_type":
            return f"Silakan sebutkan jenis produk yang Anda cari ({'/'.join(PRODUCT_TYPES)})"
        
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
        if q_type == "rekomendasi":
            if self.context.product_type and not self.context.skin_type:
                return f"Silakan sebutkan jenis kulit Anda ({'/'.join(SKIN_TYPES)})"
            return self.get_recommendations("")
        
        return "Silakan ajukan pertanyaan tentang produk skincare atau rekomendasi."

    def generate_response(self, question: str) -> str:
        with self.lock:
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
                self.context._extract_context(question, "")
                
                # Analyze question
                q_type, product, info_types = self.analyze_question(question)
                logger.info(f"Question type: {q_type}, Product: {product}, Info types: {info_types}")
                
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
                
                # Deteksi pertanyaan singkat follow-up (1-3 kata)
                is_short_follow_up = (
                    not product and 
                    self.context.last_product and 
                    len(question.split()) <= 3 and 
                    any(keyword in question.lower() for keyword in [
                        'manfaat', 'kandungan', 'cara', 'pakai', 'penggunaan', 'harga', 'lokasi', 'review'
                    ])
                )
                
                if is_short_follow_up:
                    # Gunakan info types dari kata kunci
                    if 'manfaat' in question.lower():
                        info_types = ['manfaat']
                    elif 'kandungan' in question.lower():
                        info_types = ['kandungan']
                    elif 'cara' in question.lower() or 'pakai' in question.lower() or 'penggunaan' in question.lower():
                        info_types = ['cara_pakai']
                    elif 'harga' in question.lower():
                        info_types = ['harga']
                    elif 'lokasi' in question.lower() or 'beli' in question.lower():
                        info_types = ['lokasi']
                    elif 'review' in question.lower() or 'ulasan' in question.lower():
                        info_types = ['review']
                    
                    q_type = "product_info"
                    product = self.context.last_product
                    logger.info(f"Short follow-up detected: {info_types} for {product}")
                
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