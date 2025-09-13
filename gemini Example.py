import google.generativeai as genai

genai.configure(api_key="AIzaSyA0JYaSkJlUzt8FVSuAXddPLcO0A4monaE")


model = genai.GenerativeModel("gemini-1.5-flash")

def translate_text(text, target_language):
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text
if __name__ == "__main__":   
    text_to_translate = "Hello, how are you?"
    target_language = "Spanish"
    translated_text = translate_text(text_to_translate, target_language)
    print(f"Translated Text: {translated_text}")