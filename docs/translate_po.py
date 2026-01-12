import os
import sys
import time
import polib
import google.generativeai as genai

def translate_text(model, text):
    if not text.strip() or len(text) < 2:
        return text
    
    # Check if it's just a placeholder or technical string
    if text.startswith(':') and text.endswith(':'):
        return text

    prompt = f"""
Translate the following technical documentation string from English to Japanese.
CRITICAL RULES:
1. Maintain all technical syntax: keep backticks (``), asterisks (*), and curly braces {{}} EXACTLY as they are.
2. Do not translate code-like strings or cross-references like :doc:`...` or :ref:`...`.
3. Use professional, technical Japanese (Desu/Masu style).
4. If the string contains a signature or path, do not translate it.
5. Return ONLY the translated Japanese text.

Source:
{text}
"""
    try:
        response = model.generate_content(prompt)
        translated = response.text.strip()
        
        # Basic validation: check backtick count consistency
        if text.count('`') != translated.count('`'):
            print(f"  Warning: Backtick count mismatch. Reverting to English for this entry.")
            return text
            
        return translated
    except Exception as e:
        print(f"Error translating: {e}")
        return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found")
        sys.exit(1)

    genai.configure(api_key=api_key)
    
    # Try alternate model names if the first one fails
    model_name = 'gemini-1.5-flash'
    try:
        model = genai.GenerativeModel(model_name)
        # Test call
        model.generate_content("Hi")
    except Exception:
        model_name = 'models/gemini-1.5-flash'
        model = genai.GenerativeModel(model_name)

    print(f"Using model: {model_name}")

    po_dir = "docs/locales/ja/LC_MESSAGES"
    if not os.path.exists(po_dir):
        print(f"Directory {po_dir} not found")
        return

    # Walk through all .po files in subdirectories
    for root, dirs, files in os.walk(po_dir):
        for filename in files:
            if filename.endswith(".po"):
                filepath = os.path.join(root, filename)
                print(f"Processing {filepath}...")
                po = polib.pofile(filepath)
                
                modified = False
                for entry in po:
                    # Translate if msgstr is empty OR fuzzy
                    if (not entry.msgstr or 'fuzzy' in entry.flags) and entry.msgid:
                        # Skip very short strings or those that look like code
                        if len(entry.msgid) < 3 or entry.msgid.isnumeric():
                            continue
                            
                        translated = translate_text(model, entry.msgid)
                        if translated and translated != entry.msgid:
                            entry.msgstr = translated
                            if 'fuzzy' in entry.flags:
                                entry.flags.remove('fuzzy')
                            modified = True
                            print(f"  OK: {entry.msgid[:30]}...")
                            time.sleep(1) # Respect rate limits

                if modified:
                    po.save()
                    print(f"  Saved improvements to {filename}")

if __name__ == "__main__":
    main()
