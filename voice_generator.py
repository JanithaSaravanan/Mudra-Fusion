from gtts import gTTS
import os

def generate_voice(text, filename="story_voice.mp3"):
    """
    Generate AI voice from story text
    """

    try:
        # Convert text to speech
        tts = gTTS(
            text=text,
            lang="en",      # language
            slow=False
        )

        # Save audio
        filepath = os.path.join("static", filename)
        tts.save(filepath)

        return filepath

    except Exception as e:
        print("Voice generation error:", e)
        return None