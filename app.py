"""
Livestock Helper Chatbot (RAG) - Livestock & Dairy Development Department, Punjab
==============================================================================
Farmers ask questions in English or Urdu and get answers in Urdu and English,
based on the department's FAQ documents (PDF).

How it works (RAG pipeline):
 1. Load PDFs (from the GitHub repo folder, plus optional admin uploads)
 2. Extract text page by page and clean it
 3. Split text into token chunks (tiktoken) with overlap
 4. Create multilingual embeddings (FastEmbed, free and open source, runs on CPU)
 5. Store vectors in a FAISS index
 6. For each question: translate to an English search query if needed (Groq),
    search FAISS, send the best chunks to a Groq open-source model, and show
    a bilingual answer with sources.

Run locally:  streamlit run app.py
Secret:       GROQ_API_KEY (Streamlit Cloud -> App settings -> Secrets)
"""

import glob
import hashlib
import io
import os
import re
from datetime import datetime

import numpy as np
import streamlit as st

# =============================================================== CONFIG
APP_TITLE = "Livestock Helper | لائیو سٹاک معاون"
DEPARTMENT = "Livestock & Dairy Development Department, Punjab"
HELPLINE = "0800-09211"
WEBSITE = "https://livestock.punjab.gov.pk"

GROQ_MODELS = {
    "GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (faster)": "openai/gpt-oss-20b",
}
TRANSLATE_MODEL = "openai/gpt-oss-20b"        # small, fast model for query translation
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # supports Urdu + English

DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_TOP_K = 5
LOW_MATCH_SCORE = 0.30          # below this, the documents probably do not cover the question
MAX_UPLOAD_MB = 25

TECH_LEVELS = {
    "Simple (for farmers)": (
        "Use very simple everyday words that a village livestock farmer understands. "
        "Avoid scientific terms; if you must use one, explain it in brackets. "
        "Give practical, step-by-step actions."
    ),
    "Moderate (educated farmer / trainee)": (
        "Use clear language with some technical terms, each briefly explained. "
        "Give practical steps and the reason behind them."
    ),
    "Technical (veterinary staff)": (
        "Use correct veterinary and animal-science terms. Include causes, signs, "
        "diagnosis points, prevention and management protocols where relevant."
    ),
}

RESPONSE_SIZES = {
    "Short": ("Keep each language section to 2-4 short sentences or up to 4 bullet points.", 1500),
    "Medium": ("Keep each language section to about 80-150 words, using bullet points where helpful.", 2500),
    "Detailed": ("Give a complete answer in each language (about 200-350 words each) with headings "
                 "or bullet points: overview, what to do, prevention, and when to call a vet.", 4000),
}

ANSWER_LANGUAGES = ["Urdu + English", "Urdu only", "English only"]

SAMPLE_QUESTIONS = [
    "How can I detect heat (estrus) in my buffalo?",
    "What is the CM Livestock Card and how can I apply?",
    "میری گائے کا دودھ کم ہو گیا ہے، کیا کروں؟",
    "مویشیوں کو ویکسین کب لگوانی چاہیے؟",
    "How much does artificial insemination cost?",
    "Livestock Assistant Diploma میں داخلہ کیسے ہوتا ہے؟",
]

SYSTEM_PROMPT = f"""You are "Livestock Helper", the official AI assistant of the {DEPARTMENT}, Pakistan.
You help livestock farmers of Punjab with questions about animal health, breeding, artificial
insemination, feeding and fodder, dairy, poultry, small ruminants (goats/sheep), government programmes
(e.g. CM Livestock Card, asset transfer schemes), training institutes and department services.

LOCAL CONTEXT (use when relevant):
- Farmers are mostly small-holders with 2-10 animals, often women manage the animals at home.
- Important local breeds: Sahiwal and Cholistani cows, Nili-Ravi buffalo, Beetal / Teddy goats,
  Lohi / Kajli sheep.
- Common issues: heat stress in summer, cold in winter, floods, low milk yield, repeat breeding,
  mastitis, Foot and Mouth Disease, Haemorrhagic Septicaemia (gal ghotu), Lumpy Skin Disease,
  ticks and worms, poor-quality fodder in lean seasons, and access to veterinary services.
- Department help: nearest Civil Veterinary Hospital / Dispensary, AI technician, Mobile
  Veterinary Dispensary, and the Livestock Helpline {HELPLINE} (24/7). Website: {WEBSITE}.

RULES:
1. Answer FIRST from the CONTEXT taken from the department documents. Facts in the context
   (fees, dates, eligibility, procedures, contact details) must be reported exactly.
2. If the context does not fully cover the question, you may add general good livestock practice
   for Punjab, but say clearly that it is general advice, and recommend the helpline or nearest
   veterinary hospital for confirmation.
3. Never invent fees, dates, phone numbers, names or programme rules.
4. Do NOT give exact medicine doses unless they are written in the context. For medicines, say
   the dose must be decided by a veterinary doctor.
5. For emergencies (animal not eating, high fever, bleeding, difficult birth, sudden deaths,
   suspected outbreak) tell the farmer to contact the nearest veterinary hospital or the
   helpline {HELPLINE} immediately.
6. Be respectful and kind. Use simple Urdu that rural farmers understand (common words, not
   difficult literary Urdu). Use Pakistani units and terms (maund, kanal, Rs).
7. If the question is not about livestock or the department, politely say you can only help with
   livestock matters.
8. Do not write source page numbers in the answer text; sources are shown separately.
"""


# =============================================================== PAGE SETUP + STYLE
st.set_page_config(page_title="Livestock Helper", page_icon="🐄", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
    .urdu, .urdu * {
        direction: rtl; text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
        line-height: 2.3; font-size: 1.05rem;
    }
    .lang-tag {font-weight: 600; color: #2e7d32; margin: 0.4rem 0 0.2rem 0;}
    .lang-tag-ur {font-weight: 600; color: #2e7d32; margin: 0.4rem 0 0.2rem 0;
                  direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', serif;}
    .helpline {background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 0.6rem 0.9rem;
               border-radius: 6px; margin-bottom: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================== HELPERS
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def has_urdu(text):
    return bool(re.search(r"[؀-ۿ]", text or ""))


def _is_broken_char(ch):
    """Characters produced when a PDF stores Urdu as font glyphs (cannot be read as text)."""
    code = ord(ch)
    return 0x1100 <= code <= 0x1FFF or 0xE000 <= code <= 0xF8FF or code == 0xFFFD


def clean_page_text(text):
    """Clean extracted text. Lines with broken (non-readable) Urdu glyphs are removed,
    because they would confuse the search. Properly encoded Urdu is kept."""
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\(cid:\d+\)", " ", text)
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        broken = sum(_is_broken_char(c) for c in line)
        if broken and broken / max(1, len(line)) > 0.03:
            # Mostly broken Urdu: keep only the readable English words, if any
            english = " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9&()/.,:'%-]*", line))
            if len(english.split()) >= 6:
                clean_lines.append(english)
            continue
        line = "".join(c for c in line if not _is_broken_char(c))
        line = re.sub(r"[ \t]+", " ", line)
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


# =============================================================== CACHED RESOURCES
@st.cache_resource(show_spinner=False)
def load_tokenizer():
    """tiktoken tokenizer; returns None if unavailable (a word-based fallback is used)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading the multilingual embedding model (first time ~1 minute)...")
def load_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_texts(texts, embedder):
    vectors = np.array(list(embedder.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms                     # cosine similarity via inner product


# =============================================================== RAG: BUILD KNOWLEDGE BASE
def extract_pages(file_bytes, file_name):
    """STEP 1-2: read PDF pages and clean them."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        broken = sum(_is_broken_char(c) for c in raw)
        if raw and broken / len(raw) > 0.05:
            continue        # page is mostly non-readable Urdu glyphs: skip to avoid noise
        text = clean_page_text(raw)
        if len(text.split()) >= 5:
            pages.append({"source": file_name, "page": number, "text": text})
    return pages


def make_chunks(pages, chunk_tokens, overlap_tokens, tokenizer):
    """STEP 3: join pages of each document and split into overlapping token chunks.
    Each chunk remembers the page(s) it came from."""
    chunks = []
    step = max(1, chunk_tokens - overlap_tokens)
    by_source = {}
    for p in pages:
        by_source.setdefault(p["source"], []).append(p)

    for source, doc_pages in by_source.items():
        units, unit_pages = [], []            # tokens (or words) with page numbers
        for p in doc_pages:
            if tokenizer is not None:
                toks = tokenizer.encode(p["text"] + "\n")
            else:
                toks = (p["text"] + "\n").split(" ")
            units.extend(toks)
            unit_pages.extend([p["page"]] * len(toks))

        if tokenizer is None:                 # ~1.3 tokens per word
            size, stride = max(1, int(chunk_tokens / 1.3)), max(1, int(step / 1.3))
        else:
            size, stride = chunk_tokens, step

        for start in range(0, len(units), stride):
            piece = units[start:start + size]
            if not piece:
                break
            text = tokenizer.decode(piece) if tokenizer is not None else " ".join(piece)
            first, last = unit_pages[start], unit_pages[start + len(piece) - 1]
            chunks.append({
                "source": source,
                "page": first if first == last else f"{first}-{last}",
                "text": text.strip(),
                "tokens": len(piece) if tokenizer is not None else int(len(piece) * 1.3),
            })
            if start + size >= len(units):
                break
    return [c for c in chunks if c["text"]]


@st.cache_resource(show_spinner=False)
def build_knowledge_base(files, chunk_tokens, overlap_tokens):
    """STEPS 1-5. `files` is a tuple of (name, bytes). Cached, so it is built only once
    per server and shared by all users until the documents or settings change."""
    import faiss

    pages, skipped = [], []
    for name, data in files:
        try:
            got = extract_pages(data, name)
            if got:
                pages.extend(got)
            else:
                skipped.append(f"{name}: no readable text (scanned PDF?)")
        except Exception as error:
            skipped.append(f"{name}: {error}")
    if not pages:
        raise ValueError("No readable text found in the documents. " + "; ".join(skipped))

    tokenizer = load_tokenizer()
    chunks = make_chunks(pages, chunk_tokens, overlap_tokens, tokenizer)
    embedder = load_embedder()
    vectors = embed_texts([c["text"] for c in chunks], embedder)

    index = faiss.IndexFlatIP(vectors.shape[1])        # STEP 5: FAISS vector store
    index.add(vectors)

    stats = {
        "documents": len({p["source"] for p in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "tokens": sum(c["tokens"] for c in chunks),
        "dimension": int(vectors.shape[1]),
        "tokenizer": "tiktoken cl100k_base" if tokenizer else "word estimate",
        "skipped": skipped,
        "built_at": datetime.now().strftime("%d %b %Y %H:%M"),
    }
    return {"index": index, "chunks": chunks, "stats": stats}


def find_repo_pdfs():
    """PDFs stored in the GitHub repo (same folder as app.py, or a 'data' folder)."""
    folder = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(set(glob.glob(os.path.join(folder, "*.pdf")) +
                       glob.glob(os.path.join(folder, "data", "*.pdf"))))
    files = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                files.append((os.path.basename(path), f.read()))
        except Exception:
            pass
    return files


# =============================================================== RAG: ANSWER
def groq_chat(client, model, messages, max_tokens, temperature=0.2):
    """Call Groq. Uses low reasoning effort for GPT-OSS models (faster); retries without it."""
    params = dict(model=model, messages=messages, temperature=temperature,
                  max_completion_tokens=max_tokens)
    try:
        response = client.chat.completions.create(reasoning_effort="low", **params)
    except Exception:
        response = client.chat.completions.create(**params)
    return (response.choices[0].message.content or "").strip()


def to_english_query(client, question, history):
    """Turn an Urdu / mixed / follow-up question into a short English search query."""
    last_user = [m["content"] for m in history if m["role"] == "user"][-2:]
    prompt = (
        "Rewrite the farmer's latest question as ONE short English search query for a livestock "
        "FAQ database. Translate Urdu/Punjabi words to English (e.g. 'gal ghotu' -> "
        "'Haemorrhagic Septicaemia', 'bachra' -> 'calf', 'garmi pe aana' -> 'heat estrus'). "
        "Use the previous questions only to resolve words like 'it' or 'this'. "
        "Return only the query, nothing else.\n\n"
        f"Previous questions: {last_user}\nLatest question: {question}"
    )
    try:
        query = groq_chat(client, TRANSLATE_MODEL, [{"role": "user", "content": prompt}], 300, 0)
        query = query.strip().strip('"').splitlines()[0] if query else ""
        return query[:300]
    except Exception:
        return ""


def retrieve(queries, kb, embedder, top_k):
    """STEP 6: search FAISS with each query and merge the best results."""
    best = {}
    vectors = embed_texts([q for q in queries if q], embedder)
    scores, ids = kb["index"].search(vectors, min(top_k, len(kb["chunks"])))
    for row_scores, row_ids in zip(scores, ids):
        for score, i in zip(row_scores, row_ids):
            if i == -1:
                continue
            if i not in best or score > best[i]:
                best[i] = float(score)
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**kb["chunks"][i], "score": s} for i, s in ranked]


def build_answer_instructions(tech_level, size, language):
    size_text = RESPONSE_SIZES[size][0]
    if language == "Urdu + English":
        fmt = ("Write the answer in BOTH languages in exactly this format:\n"
               "[[EN]]\n<English answer>\n[[UR]]\n<the same answer in simple Urdu script>\n"
               "Both parts must contain the same information.")
    elif language == "Urdu only":
        fmt = "Write the answer only in simple Urdu script, in this format:\n[[UR]]\n<Urdu answer>"
    else:
        fmt = "Write the answer only in English, in this format:\n[[EN]]\n<English answer>"
    return (f"TECHNICAL LEVEL: {TECH_LEVELS[tech_level]}\n"
            f"RESPONSE SIZE: {size_text}\n"
            f"FORMAT: {fmt}\n"
            "Use markdown bullet points where useful. Do not use tables.")


def generate_answer(client, model, question, context_chunks, history,
                    tech_level, size, language):
    if context_chunks:
        context = "\n\n".join(
            f"[Document: {c['source']} | page {c['page']} | match {c['score']:.2f}]\n{c['text']}"
            for c in context_chunks)
    else:
        context = "(No matching text found in the documents.)"

    weak = (not context_chunks) or context_chunks[0]["score"] < LOW_MATCH_SCORE
    note = ("NOTE: The documents do not seem to cover this question well. Follow rule 2."
            if weak else "")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"][:2000]})
    messages.append({
        "role": "user",
        "content": (f"CONTEXT FROM DEPARTMENT DOCUMENTS:\n{context}\n\n{note}\n\n"
                    f"{build_answer_instructions(tech_level, size, language)}\n\n"
                    f"FARMER'S QUESTION: {question}"),
    })
    return groq_chat(client, model, messages, RESPONSE_SIZES[size][1])


def split_languages(answer):
    """Split the model output into English and Urdu parts."""
    en, ur = "", ""
    if "[[EN]]" in answer or "[[UR]]" in answer:
        en_match = re.search(r"\[\[EN\]\](.*?)(?=\[\[UR\]\]|\Z)", answer, re.S)
        ur_match = re.search(r"\[\[UR\]\](.*?)(?=\[\[EN\]\]|\Z)", answer, re.S)
        en = en_match.group(1).strip() if en_match else ""
        ur = ur_match.group(1).strip() if ur_match else ""
    elif has_urdu(answer) and not re.search(r"[A-Za-z]{4,}.*[A-Za-z]{4,}", answer[:200]):
        ur = answer.strip()
    else:
        en = answer.strip()
    return en, ur


# =============================================================== DISPLAY HELPERS
def show_answer(answer_text, layout):
    en, ur = split_languages(answer_text)
    if en and ur and layout == "Side by side":
        col_en, col_ur = st.columns(2)
        with col_en:
            st.markdown('<div class="lang-tag">🇬🇧 English</div>', unsafe_allow_html=True)
            st.markdown(en)
        with col_ur:
            st.markdown('<div class="lang-tag-ur">اردو 🇵🇰</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="urdu">\n\n{ur}\n\n</div>', unsafe_allow_html=True)
        return
    if en:
        if ur:
            st.markdown('<div class="lang-tag">🇬🇧 English</div>', unsafe_allow_html=True)
        st.markdown(en)
    if ur:
        if en:
            st.markdown('<div class="lang-tag-ur">اردو 🇵🇰</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="urdu">\n\n{ur}\n\n</div>', unsafe_allow_html=True)


def show_sources(sources, search_query=""):
    if not sources:
        return
    with st.expander(f"📎 Sources from documents ({len(sources)})"):
        if search_query:
            st.caption(f"🔎 Search query used: {search_query}")
        for s in sources:
            level = "🟢" if s["score"] >= 0.55 else ("🟡" if s["score"] >= LOW_MATCH_SCORE else "🔴")
            st.markdown(f"{level} **{s['source']} – page {s['page']}** · match {s['score']:.2f}")
            st.caption(s["text"][:450] + ("..." if len(s["text"]) > 450 else ""))


def chat_transcript(messages):
    lines = [f"{APP_TITLE}", DEPARTMENT, f"Helpline: {HELPLINE}",
             f"Date: {datetime.now().strftime('%d %B %Y %H:%M')}", "=" * 50]
    for m in messages:
        who = "Farmer" if m["role"] == "user" else "Livestock Helper"
        text = m["content"].replace("[[EN]]", "\n[English]\n").replace("[[UR]]", "\n[Urdu]\n")
        lines += ["", f"{who}:", text.strip()]
    return "\n".join(lines)


# =============================================================== SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_files" not in st.session_state:
    st.session_state.admin_files = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =============================================================== SIDEBAR
with st.sidebar:
    st.markdown(f"### 🐄 {APP_TITLE}")
    st.caption(DEPARTMENT)
    st.markdown(
        f'<div class="helpline">📞 <b>Livestock Helpline: {HELPLINE}</b> (24/7)<br>'
        f'<span class="urdu" style="line-height:1.9">ہنگامی صورت میں فوراً ہیلپ لائن پر کال کریں</span></div>',
        unsafe_allow_html=True)

    st.subheader("🎛️ Answer settings")
    tech_level = st.radio("Technical level", list(TECH_LEVELS.keys()), index=0,
                          help="Simple for farmers, Technical for veterinary staff")
    response_size = st.select_slider("Response size", options=list(RESPONSE_SIZES.keys()), value="Medium")
    answer_language = st.selectbox("Answer language", ANSWER_LANGUAGES, index=0)
    layout = st.radio("Show Urdu & English", ["One below the other", "Side by side"], index=0,
                      horizontal=True)
    model_label = st.selectbox("AI model (Groq, open source)", list(GROQ_MODELS.keys()))
    model = GROQ_MODELS[model_label]

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help="Free key: console.groq.com/keys")

    with st.expander("🛠️ Admin: documents & search settings"):
        chunk_tokens = st.slider("Chunk size (tokens)", 200, 800, DEFAULT_CHUNK_TOKENS, 50)
        overlap_tokens = st.slider("Chunk overlap (tokens)", 0, 150, DEFAULT_OVERLAP_TOKENS, 10)
        top_k = st.slider("Chunks used per answer", 2, 10, DEFAULT_TOP_K)
        show_src = st.checkbox("Show sources under answers", value=True)
        extra = st.file_uploader("Add more PDF documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Add to knowledge base", disabled=not extra):
            added = []
            for f in extra:
                if f.size <= MAX_UPLOAD_MB * 1024 * 1024:
                    st.session_state.admin_files.append((f.name, f.getvalue()))
                    added.append(f.name)
            st.success("Added: " + ", ".join(added) if added else "No file added (too large).")
        if st.session_state.admin_files and st.button("Remove uploaded documents"):
            st.session_state.admin_files = []
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        c2.download_button("📥 Save chat", chat_transcript(st.session_state.messages),
                           file_name="livestock_chat.txt", use_container_width=True)
    st.caption("⚠️ This assistant gives guidance only. For sick animals, always consult a "
               "veterinary doctor.")


# =============================================================== LOAD KNOWLEDGE BASE
st.title("🐄 Livestock Helper")
st.markdown('<div class="urdu" style="font-size:1.25rem">لائیو سٹاک معاون — اپنا سوال اردو یا انگریزی میں پوچھیں</div>',
            unsafe_allow_html=True)
st.caption(f"{DEPARTMENT} · Ask in English or Urdu · Answers in Urdu and English")

all_files = tuple(find_repo_pdfs() + st.session_state.admin_files)
kb = None
if not all_files:
    st.warning("No documents found. Upload the FAQ PDF to the GitHub repo (next to app.py) "
               "or add it in the sidebar under **Admin**.")
else:
    try:
        with st.spinner("Preparing the knowledge base (only the first time)..."):
            kb = build_knowledge_base(all_files, chunk_tokens, overlap_tokens)
    except Exception as error:
        st.error(f"Could not build the knowledge base: {error}")

if kb:
    s = kb["stats"]
    with st.sidebar.expander("📚 Knowledge base"):
        st.write(f"Documents: **{s['documents']}** · Pages: **{s['pages']}**")
        st.write(f"Chunks: **{s['chunks']}** · Tokens: **{s['tokens']:,}**")
        st.write(f"Vector size: **{s['dimension']}** · Store: **FAISS**")
        st.caption(f"Tokenizer: {s['tokenizer']} · Built: {s['built_at']}")
        for item in s["skipped"]:
            st.warning("Skipped " + item)


# =============================================================== SAMPLE QUESTIONS
if not st.session_state.messages:
    st.markdown("##### 💡 Try a question / کوئی سوال آزمائیں")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


# =============================================================== CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🐄"):
        if msg["role"] == "user":
            if has_urdu(msg["content"]):
                st.markdown(f'<div class="urdu">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
        else:
            show_answer(msg["content"], layout)
            if show_src:
                show_sources(msg.get("sources", []), msg.get("query", ""))


# =============================================================== NEW QUESTION
typed = st.chat_input("Type your question in Urdu or English… / اپنا سوال لکھیں…", disabled=kb is None)
question = typed or st.session_state.pending_question
st.session_state.pending_question = None

if question and kb:
    question = question.strip()[:1000]
    if not api_key:
        st.error("Groq API key is missing. Add it in Streamlit secrets or the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        if has_urdu(question):
            st.markdown(f'<div class="urdu">{question}</div>', unsafe_allow_html=True)
        else:
            st.markdown(question)

    with st.chat_message("assistant", avatar="🐄"):
        sources, search_query = [], ""
        with st.spinner("Searching documents… / جواب تیار ہو رہا ہے…"):
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                embedder = load_embedder()
                queries = [question]
                if has_urdu(question) or history:          # translate / resolve follow-ups
                    search_query = to_english_query(client, question, history)
                    if search_query and search_query.lower() != question.lower():
                        queries.append(search_query)
                sources = retrieve(queries, kb, embedder, top_k)
                answer = generate_answer(client, model, question, sources, history,
                                         tech_level, response_size, answer_language)
                if not answer:
                    answer = ("[[EN]]\nSorry, I could not prepare an answer. Please try again or call "
                              f"the helpline {HELPLINE}.\n[[UR]]\nمعذرت، جواب تیار نہیں ہو سکا۔ "
                              f"دوبارہ کوشش کریں یا ہیلپ لائن {HELPLINE} پر رابطہ کریں۔")
            except Exception as error:
                answer = (f"[[EN]]\n⚠️ Something went wrong: {error}\n\nPlease try again, or call the "
                          f"Livestock Helpline {HELPLINE}.\n[[UR]]\n⚠️ کوئی مسئلہ پیش آ گیا۔ "
                          f"دوبارہ کوشش کریں یا ہیلپ لائن {HELPLINE} پر کال کریں۔")
        show_answer(answer, layout)
        if show_src:
            show_sources(sources, search_query)

    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources, "query": search_query})
