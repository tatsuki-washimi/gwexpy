import os
import sys
import time
import polib
import google.generativeai as genai

def translate_text(model, text):
    if not text.strip():
        return text
    
    prompt = f"""
Translate the following technical documentation string for a Python library 'gwexpy' from English to Japanese.
Keep technical terms (like class names, function names, parameters) in English unless there's a standard Japanese equivalent.
Return ONLY the translated text.

Source English:
{text}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error translating: {e}")
        return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    po_dir = "docs/locales/ja/LC_MESSAGES"
    if not os.path.exists(po_dir):
        print(f"Directory {po_dir} not found")
        return

    for filename in os.listdir(po_dir):
        if filename.endswith(".po"):
            filepath = os.path.join(po_dir, filename)
            print(f"Processing {filepath}...")
            po = polib.pofile(filepath)
            
            modified = False
            for entry in po:
                if not entry.msgstr and entry.msgid:
                    translated = translate_text(model, entry.msgid)
                    if translated:
                        entry.msgstr = translated
                        modified = True
                        print(f"  Translated: {entry.msgid[:30]}... -> {translated[:30]}...")
                        time.sleep(1) # Rate limit protection

            if modified:
                po.save()
                print(f"  Saved improvements to {filename}")

if __name__ == "__main__":
    main()
