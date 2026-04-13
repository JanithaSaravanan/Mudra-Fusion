"""
Handles all LLM-related operations for Mudra-Fusion.
Separates Gemini API calls from the core story engine logic.
"""

import os
import google.generativeai as genai

# ===================== CONFIGURATION =====================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "api key yours")
USE_LLM = GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE'

if USE_LLM:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured successfully")
    except Exception as e:
        print(f"⚠️  Failed to configure Gemini API: {e}")
        USE_LLM = False

# ===================== LLM STORY GENERATION =====================

def generate_with_llm(mudra_meanings_dict, mudra_sequence, verse):
    """
    Use Google Gemini Flash to write a domain-accurate Bharatanatyam story interpretation.
    
    Args:
        mudra_meanings_dict: Dictionary mapping mudra names to their meanings
        mudra_sequence: List of mudra names in the sequence performed
        verse: Dictionary containing verse data (text, translation, commentary, etc.)
    
    Returns:
        Generated story string, or None if LLM is unavailable or call fails
    """
    if not USE_LLM:
        return None

    try:
        # --- Build per-mudra detail block ---
        mudra_lines = []
        for i, m in enumerate(mudra_sequence, 1):
            meanings = mudra_meanings_dict.get(m, [])
            mudra_lines.append(
                f"  {i}. {m}\n"
                f"     Symbolic meanings: {', '.join(meanings)}\n"
                f"     Position in sequence: {'opening gesture' if i == 1 else 'closing gesture' if i == len(mudra_sequence) else f'gesture {i} of {len(mudra_sequence)}'}"
            )
        mudra_block = "\n".join(mudra_lines)

        verse_emotions = verse.get("emotions", [])

        # --- Combined symbolic meanings across all mudras ---
        all_meanings = set()
        for mudra in mudra_sequence:
            all_meanings.update(mudra_meanings_dict.get(mudra, []))
        meanings_summary = ", ".join(sorted(all_meanings))

        # --- Speaker context ---
        speaker = verse.get("speaker", "Unknown")
        source = verse.get("source", "")
        chapter = verse.get("Chapter", "")

        # --- Build the prompt ---
        prompt = f"""You are a friendly guide at a Bharatanatyam dance performance, explaining to a general audience 
what the dancer on stage is communicating through their hand gestures.

=== BACKGROUND ===
In Bharatanatyam, hand gestures are a precise sign language. Each gesture carries a symbolic meaning, 
and a sequence of gestures tells a story — like sentences building into a paragraph.
The dancer is narrating a moment from Indian scripture through their body.
=== WHAT THE DANCER JUST DID ===
The dancer performed these gestures in order:
{mudra_block}

=== THE STORY BEHIND THE GESTURES ===
These gestures relate to this moment from {source}:
Speaker in the verse: {speaker}
What is happening: {verse.get('translation', '')}
The deeper meaning: {verse.get('commentary_full', verse.get('commentary_summary', ''))}

=== CRITICAL — READ BEFORE WRITING ===
Before writing a single sentence, find the ONE central story that connects ALL the gestures together.
Every gesture must feel like a natural step in that same story — not isolated descriptions stitched together.
Think of it like a short film: the gestures are scenes, and your explanation is the narrator's voice 
that makes the audience feel they are watching one continuous, meaningful moment — not a slideshow.
Ask yourself: what is the dancer saying from beginning to end, as one complete thought?

=== YOUR TASK ===
Write a warm, connected explanation as if you are a knowledgeable friend sitting next to someone 
in the audience, quietly telling them what they are watching. Structure it like this:

1. Opening (2-3 sentences): Paint the scene. What story or emotion is unfolding? 
   Who is being portrayed, and what moment in their journey is this?
   Make the audience feel the context before the gestures are explained.

2. Gesture by gesture (1-2 sentences per gesture): Walk through each gesture in sequence order.
   Each sentence must do two things: explain what the gesture represents AND show how it 
   moves the story forward from the previous gesture. Use connecting words like 
   "then", "this leads to", "building on that", "as the story deepens" — so it reads 
   as one flowing narrative, not separate descriptions.

3. Closing (2-3 sentences): Bring it all together. What is the complete message the dancer 
   has just delivered? What should the audience feel sitting there — what emotion, 
   what understanding, what takeaway stays with them?

=== STRICT RULES ===
- Use plain, warm English — no Sanskrit terms, no technical dance jargon
- Do NOT use the words "mudra", "abhinaya", "navarasa", "hasta"
- Every single gesture in the sequence must appear in the explanation — skip none
- The story must flow as ONE connected narrative — if any sentence feels disconnected, rewrite it
- The explanation must be grounded in the verse's actual events and characters — do not invent
- No bullet points, no headings, no bold text, no markdown — natural flowing paragraphs only
- Someone who knows nothing about classical dance or scripture must finish reading and think: 
  "I understood that, and I felt something." That is the standard to meet.
  - Maximum 6-8 sentences. Keep it concise and clear."""

  
  

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"⚠️  LLM call failed: {e}. Using template fallback.")
        return None


# ===================== STATUS CHECK =====================

def is_llm_available():
    """Check if LLM is available and configured."""
    return USE_LLM
