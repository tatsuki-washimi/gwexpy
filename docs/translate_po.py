import os
import sys
import time
import polib
import google.generativeai as genai

def translate_text(model, text):
    if not text.strip() or len(text) < 2:
        return text
    
    # Do not translate literal path or refs
    if any(x in text for x in [":doc:", ":ref:", "http://", "https://"]):
        if len(text.split()) < 3: # If it's just a ref, skip
            return text

    prompt = f"""
Translate the following technical documentation string from English to Japanese.
CRITICAL RULES:
1. Maintain all technical syntax: keep backticks (``), asterisks (*), and curly braces {{}} EXACTLY as they are.
2. DO NOT translate strings inside backticks if they look like code.
3. Use professional Japanese (Desu/Masu).
4. Return ONLY the translated Japanese text.

Source English:
{text}
"""
    try:
        response = model.generate_content(prompt)
        translated = response.text.strip()
        
        # Validation: check backtick count consistency
        # If backticks are broken, it ruins the Sphinx build.
        if text.count('`') != translated.count('`'):
            return text
            
        return translated
    except Exception as e:
        print(f"  Error during API call: {e}", flush=True)
        return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found", flush=True)
        sys.exit(1)

    genai.configure(api_key=api_key)
    
    # Model selection. 'gemini-1.5-flash' is the stable target.
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Using model: gemini-1.5-flash", flush=True)

    po_dir = "docs/locales/ja/LC_MESSAGES"
    if not os.path.exists(po_dir):
        # Fallback if path is different in GHA
        po_dir = "locales/ja/LC_MESSAGES"
        if not os.path.exists(po_dir):
            print(f"Directory {po_dir} not found. Current dir: {os.getcwd()}", flush=True)
            return

    for root, dirs, files in os.walk(po_dir):
        for filename in files:
            if filename.endswith(".po"):
                filepath = os.path.join(root, filename)
                print(f"Processing {filepath}...", flush=True)
                po = polib.pofile(filepath)
                
                modified = False
                total = len(po)
                count = 0
                for entry in po:
                    # Translate if msgstr is empty OR fuzzy
                    if (not entry.msgstr or 'fuzzy' in entry.flags) and entry.msgid:
                        # Skip numeric/very short
                        if entry.msgid.isnumeric() or len(entry.msgid) < 3:
                            continue
                            
                        translated = translate_text(model, entry.msgid)
                        if translated and translated != entry.msgid:
                            entry.msgstr = translated
                            if 'fuzzy' in entry.flags:
                                entry.flags.remove('fuzzy')
                            modified = True
                            count += 1
                            if count % 5 == 0:
                                print(f"  Progress: {count} strings translated...", flush=True)
                            time.sleep(1) # Rate limit

                if modified:
                    po.save()
                    print(f"  Saved {count} improvements to {filename}", flush=True)

if __name__ == "__main__":
    main()
