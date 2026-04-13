import json

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# ===================== LOAD DATA =====================

with open("mudra_meanings.json", "r", encoding="utf-8") as f:
    MUDRA_MEANINGS = json.load(f)

with open("synonyms.json", "r", encoding="utf-8") as f:
    SYNONYMS = json.load(f)

with open("b_chapter_1.json", "r", encoding="utf-8") as f:
    VERSES = json.load(f)


# ===================== LOAD SBERT MODEL =====================

MODEL_NAME = "all-MiniLM-L6-v2"
model = None
model_load_attempted = False


def get_embedding_model():
    """Load the embedding model lazily so offline/cache issues do not crash imports."""
    global model, model_load_attempted

    if model is not None:
        return model

    if model_load_attempted:
        return None

    if SentenceTransformer is None:
        return None

    model_load_attempted = True

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception:
        model = None

    return model


# ===================== LLM SUPPORT =====================

try:
    from llm_handler import generate_with_llm, is_llm_available
except Exception:
    generate_with_llm = None

    def is_llm_available():
        return False


def generate_with_llm_wrapper(mudra_sequence, verse):
    if generate_with_llm is None or not is_llm_available():
        return None
    return generate_with_llm(MUDRA_MEANINGS, mudra_sequence, verse)


# ===================== TEMPLATE FALLBACK =====================

def generate_template_story(mudra_sequence, verse):
    mudra_desc = []
    for m in mudra_sequence:
        meanings = ", ".join(MUDRA_MEANINGS.get(m, [])[:4])
        mudra_desc.append(f"'{m}' ({meanings})")

    mudra_text = " → ".join(mudra_desc)

    all_meanings = extract_meanings(mudra_sequence)
    top_meanings = list(all_meanings)[:5]
    meaning_phrase = ", ".join(top_meanings)

    return (
        f"The dancer expresses {meaning_phrase}. "
        f"Through {mudra_text}, the performance reflects "
        f"the essence of {verse['theme'].lower()}."
    )


# ===================== STEP 1: EXTRACT MEANINGS =====================

def extract_meanings(mudra_sequence):
    meanings = set()
    for mudra in mudra_sequence:
        meanings.update(MUDRA_MEANINGS.get(mudra, []))
    return meanings


# ===================== STEP 2: EXPAND SYNONYMS =====================

def expand_synonyms(meanings):
    expanded = set(m.lower() for m in meanings)

    for meaning in meanings:
        for syn in SYNONYMS.get(meaning, []):
            expanded.add(syn.lower())

    return expanded


# ===================== STEP 3: PREPARE VERSE EMBEDDINGS =====================

def prepare_verses(verses):
    embedding_model = get_embedding_model()

    for verse in verses:
        verse.pop("embedding", None)
        if embedding_model is not None:
            text = " ".join(verse.get("keywords", [])) + " " + verse.get("translation", "")
            verse["embedding"] = embedding_model.encode(text)
    return verses


VERSES = prepare_verses(VERSES)


# ===================== STEP 4: MATCH VERSES =====================

def match_verses(expanded_meanings, verses):
    embedding_model = get_embedding_model()
    query_embedding = None

    if embedding_model is not None:
        query_text = " ".join(expanded_meanings)
        query_embedding = embedding_model.encode(query_text)

    matched = []

    for verse in verses:
        verse_keywords = set(k.lower() for k in verse.get("keywords", []))
        jaccard_score = len(verse_keywords & expanded_meanings) / max(1, len(verse_keywords | expanded_meanings))
        sbert_score = 0.0

        if query_embedding is not None and util is not None and "embedding" in verse:
            sbert_score = util.cos_sim(query_embedding, verse["embedding"]).item()
            final_score = 0.8 * sbert_score + 0.2 * jaccard_score
        else:
            final_score = jaccard_score

        matched.append((final_score, verse))

    matched.sort(reverse=True, key=lambda x: x[0])
    return matched


# ===================== STEP 5: STORY GENERATION =====================

def generate_storyline(mudra_sequence, verse):
    """
    Generate story with BOTH AI-generated and Template-based versions.
    Returns a dictionary with both interpretations.
    """
    mudra_desc = []
    for m in mudra_sequence:
        meanings = ", ".join(MUDRA_MEANINGS.get(m, []))
        mudra_desc.append(f"'{m}' ({meanings})")

    mudra_text = " → ".join(mudra_desc)

    # Generate BOTH versions
    llm_story = generate_with_llm_wrapper(mudra_sequence, verse)
    template_story = generate_template_story(mudra_sequence, verse)

    # Base story structure (common for both)
    base_story = f"""
🩰 MUDRA-BASED STORY INTERPRETATION

🔹 Mudra Sequence:
{mudra_text}

🔹 Matched Verse:
Source : {verse['source']}
Speaker: {verse['speaker']}
Id: {verse['id']}

📜 Sanskrit:
{verse['text_sanskrit']}

🔤 Transliteration:
{verse['transliteration']}

📘 Translation:
{verse['translation']}

🧠 Commentary Summary:
{verse['commentary_summary']}

🎭 Theme:
{verse['theme']}

✨ Interpretation ("""

    # AI-generated version (if available)
    ai_version = base_story + f"AI-generated):\n{llm_story}\n" if llm_story else None

    # Template version (always available as fallback)
    template_version = base_story + f"Template-based):\n{template_story}\n"

    return {
        "ai_generated": ai_version,
        "template_based": template_version,
        "preferred": ai_version if ai_version else template_version
    }


# ===================== FULL PIPELINE FUNCTION =====================

def run_story_engine(mudra_sequence, version="all"):
    """
    Main entry point. Generates story in specified version.
    
    Args:
        mudra_sequence: List of mudra names
        version: "all" (both versions), "ai" (AI only, fallback to template), 
                "template" (template only), "preferred" (AI if available else template)
    
    Returns:
        Dictionary with story versions or single story string based on version param
    """
    meanings = extract_meanings(mudra_sequence)
    expanded = expand_synonyms(meanings)
    matched = match_verses(expanded, VERSES)

    if not matched:
        return "❌ No matching verse found."

    best_verse = matched[0][1]
    stories = generate_storyline(mudra_sequence, best_verse)
    
    # Return based on requested version
    if version == "all":
        return stories
    elif version == "ai":
        return stories["ai_generated"] if stories["ai_generated"] else stories["template_based"]
    elif version == "template":
        return stories["template_based"]
    else:  # "preferred" or default
        return stories["preferred"]
