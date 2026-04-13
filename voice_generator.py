"""
VOICE GENERATION MODULE
-----------------------
This module converts story text into speech using Google Text-to-Speech.
It includes preprocessing for better narration.
The generated audio is stored inside the static folder.
"""

from gtts import gTTS
import os
import datetime
import re
from num2words import num2words

# ============================================================
# CONFIGURATION SETTINGS
# ============================================================

DEFAULT_LANGUAGE = "en"
DEFAULT_FILENAME = "story_voice.mp3"
STATIC_FOLDER = "static"

# Abbreviation dictionary
ABBREVIATIONS = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "St.": "Saint",
    "vs.": "versus"
}

# Sanskrit transliteration replacements
SANSKRIT_MAP = {
    # Characters from Ramayana
    "Rāma": "Rama",
    "Sītā": "Sita",
    "Lakṣmaṇa": "Lakshmana",
    "Hanumān": "Hanuman",
    "Rāvan": "Ravan",
    "Kaikeyī": "Kaikeyi",
    "Sugrīva": "Sugriva",
    "Vibhīṣaṇa": "Vibhishana",
    "Indrajit": "Indrajit",
    "Bharata": "Bharata",
    "Jātāyu": "Jatayu",
    "Marīca": "Maricha",
    
    # Places from Ramayana
    "Ayodhyā": "Ayodhya",
    "Lankā": "Lanka",
    "Dandaka": "Dandaka",
    "Chitrakūṭa": "Chitrakut",
    "Panchavati": "Panchavati",
    
    # Terms & common Sanskrit words
    "Dharma": "Dharma",
    "Karma": "Karma",
    "Bhagavān": "Bhagavan",
    "Ātman": "Atman",
    "Mokṣa": "Moksha",
    
    # Characters from Bhagavad Gita / Mahabharata
    "Kṛṣṇa": "Krishna",
    "Arjuna": "Arjuna",
    "Bhīṣma": "Bhishma",
    "Duryodhana": "Duryodhana",
    "Yudhiṣṭhira": "Yudhishthira",
    "Nakula": "Nakula",
    "Sahadeva": "Sahadeva",
    "Draupadī": "Draupadi",
    "Gāndhārī": "Gandhari",
    "Kunti": "Kunti",
    "Vidura": "Vidura",
    "Sanatkumāra": "Sanatkumara"
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_static_directory():
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)


def clean_text(text):
    """
    Remove extra spaces, unwanted symbols, normalize punctuation
    """
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[!@#$%^&*()_+=<>?{}\\[\]|]", "", text)  # Remove special symbols
    text = re.sub(r"\.{2,}", ".", text)  # Normalize multiple periods
    text = text.strip()
    return text


def expand_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text


def convert_numbers(text):
    """
    Convert digits to words using num2words
    """
    def replace_number(match):
        return num2words(int(match.group(0)))
    text = re.sub(r"\b\d+\b", replace_number, text)
    return text


def simplify_sanskrit(text):
    for original, replacement in SANSKRIT_MAP.items():
        text = text.replace(original, replacement)
    return text


def split_sentences(text):
    """
    Split long paragraphs into sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences)


def add_dramatic_pauses(text):
    """
    Add ellipsis for dramatic effect
    """
    text = re.sub(r"\b(suddenly|abruptly|quickly)\b", r"... \1 ...", text, flags=re.IGNORECASE)
    return text


def preprocess_text(text):
    """
    Full preprocessing pipeline
    """
    if not text:
        return ""
    text = clean_text(text)
    text = expand_abbreviations(text)
    text = convert_numbers(text)
    text = simplify_sanskrit(text)
    text = split_sentences(text)
    text = add_dramatic_pauses(text)
    return text


def build_output_path(filename):
    ensure_static_directory()
    return os.path.join(STATIC_FOLDER, filename)


# ============================================================
# CORE VOICE GENERATION FUNCTION
# ============================================================

def generate_voice(text, filename=DEFAULT_FILENAME):
    try:
        processed_text = preprocess_text(text)
        if processed_text == "":
            print("⚠ No text provided for voice generation.")
            return None

        tts_engine = gTTS(
            text=processed_text,
            lang=DEFAULT_LANGUAGE,
            slow=False
        )

        filepath = build_output_path(filename)
        tts_engine.save(filepath)

        print("Voice narration generated successfully.")
        print("Audio file saved at:", filepath)
        print("Timestamp:", datetime.datetime.now())

        return filepath

    except Exception as e:
        print("Voice generation error occurred.")
        print("Error details:", str(e))
        return None