import json

# ===================== LOAD DATA =====================

with open("mudra_meanings.json", "r", encoding="utf-8") as f:
    MUDRA_MEANINGS = json.load(f)

with open("synonyms.json", "r", encoding="utf-8") as f:
    SYNONYMS = json.load(f)

with open("b_chapter_1.json", "r", encoding="utf-8") as f:
    VERSES = json.load(f)


# ===================== STEP 1: EXTRACT MEANINGS =====================

def extract_meanings(mudra_sequence):
    """
    Convert mudra sequence into a set of symbolic meanings
    """
    meanings = set()
    for mudra in mudra_sequence:
        meanings.update(MUDRA_MEANINGS.get(mudra, []))
    return meanings


# ===================== STEP 2: EXPAND SYNONYMS =====================

def expand_synonyms(meanings):
    """
    Expand meanings using synonym dictionary
    """
    expanded = set(m.lower() for m in meanings)

    for meaning in meanings:
        for syn in SYNONYMS.get(meaning, []):
            expanded.add(syn.lower())

    return expanded


# ===================== STEP 3: MATCH VERSES =====================

def match_verses(expanded_meanings, verses):
    """
    Match verses using keyword intersection scoring
    """
    matched = []

    for verse in verses:
        verse_keywords = set(k.lower() for k in verse.get("keywords", []))
        score = len(verse_keywords & expanded_meanings)

        if score > 0:
            matched.append((score, verse))

    matched.sort(reverse=True, key=lambda x: x[0])
    return matched


# ===================== STEP 4: STORY GENERATION =====================

def generate_storyline(mudra_sequence, verse):
    """
    Generate a narrative explanation using verse metadata
    """

    mudra_desc = []
    for m in mudra_sequence:
        meanings = ", ".join(MUDRA_MEANINGS.get(m, []))
        mudra_desc.append(f"'{m}' ({meanings})")

    mudra_text = " → ".join(mudra_desc)

    story = f"""
🩰 MUDRA-BASED STORY INTERPRETATION

🔹 Mudra Sequence:
{mudra_text}

🔹 Matched Verse:
Source : {verse['source']}
Speaker: {verse['speaker']}

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

✨ Interpretation:
The sequence of mudras symbolically conveys the emotions and narrative expressed
in the above verse, translating physical gestures into classical textual meaning.
"""

    return story


# ===================== FULL PIPELINE =====================

def run_story_engine(mudra_sequence):
    meanings = extract_meanings(mudra_sequence)
    expanded = expand_synonyms(meanings)
    matched = match_verses(expanded, VERSES)

    if not matched:
        return "❌ No matching verse found."

    best_verse = matched[0][1]
    return generate_storyline(mudra_sequence, best_verse)


# ===================== EXAMPLE RUN =====================

if __name__ == "__main__":
    # Example mudra sequence from real-time system
    mudra_sequence = ["Alapadmam", "Anjali"]

    output = run_story_engine(mudra_sequence)
    print(output)
