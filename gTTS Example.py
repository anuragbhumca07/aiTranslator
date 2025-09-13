from gtts import gTTS

def text_to_speech(text, lang="en", filename="output.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename
# Example usage
if __name__ == "__main__":
    text = "Hello, how are you?"
    output_file = text_to_speech(text, lang="en", filename="hello.mp3")
    print(f"Audio content saved to {output_file}")