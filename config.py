"""
Central configuration for the Nutrition Agent system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai credentials ────────────────────────────────────────────────
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "oEemYECa1DXFEQtUq0EbMoPUHJ1OW6tqemLsddrI3RsQ")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "40e09d61-b236-4796-8222-2f212105aeb0")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# ── Model identifiers ─────────────────────────────────────────────────────────
PRIMARY_MODEL   = "meta-llama/llama-3-3-70b-instruct"
EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr"

# ── LLM generation parameters ─────────────────────────────────────────────────
GENERATION_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 50,
    "temperature":    0.4,
    "top_p":          0.9,
    "repetition_penalty": 1.1,
}

# ── Vector store settings ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./nutrition_agent/data/chroma_db"
COLLECTION_NAME    = "nutrition_knowledge"

# ── Daily Reference Values (DRVs) – general adult ────────────────────────────
DAILY_REFERENCE_VALUES = {
    "calories":      2000,
    "protein_g":       50,
    "carbohydrates_g": 275,
    "fat_g":            78,
    "fiber_g":          28,
    "sugar_g":          50,
    "sodium_mg":      2300,
    "calcium_mg":     1000,
    "iron_mg":          18,
    "vitamin_c_mg":     90,
    "vitamin_d_iu":    600,
    "potassium_mg":   4700,
    "magnesium_mg":    420,
}

# ── Chronic condition diet profiles ───────────────────────────────────────────
HEALTH_CONDITION_PROFILES = {
    "diabetes":           {"carbohydrates_g": 130, "sugar_g": 25, "fiber_g": 35},
    "hypertension":       {"sodium_mg": 1500},
    "heart_disease":      {"fat_g": 55, "sodium_mg": 1500},
    "kidney_disease":     {"protein_g": 40, "potassium_mg": 2000, "sodium_mg": 1500},
    "obesity":            {"calories": 1600, "fat_g": 55, "sugar_g": 25},
    "celiac_disease":     {"gluten_free": True},
    "lactose_intolerance":{"dairy_free": True},
}
