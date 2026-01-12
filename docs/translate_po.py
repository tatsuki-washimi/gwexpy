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
    # Using gemini-1.5-flash as it is the most standard for this task
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
                # Translate if msgstr is empty OR if the entry is marked as 'fuzzy'
                should_translate = not entry.msgstr or 'fuzzy' in entry.flags
                
                if should_translate and entry.msgid:
                    translated = translate_text(model, entry.msgid)
                    if translated:
                        entry.msgstr = translated
                        if 'fuzzy' in entry.flags:
                            entry.flags.remove('fuzzy') # Clear fuzzy flag after translation
                        modified = True
                        print(f"  Translated: {entry.msgid[:30]}... -> {translated[:30]}...")
                        time.sleep(1) # Rate limit protection

            if modified:
                po.save()
                print(f"  Saved improvements to {filename}")

if __name__ == "__main__":
    main()
