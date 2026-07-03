"""
02_wiki_xml_parser.py

NogaiLLM Data Engineering Pipeline - Phase 2
Parses MediaWiki XML dumps and normalizes Kipchak/Turkic character variations 
into standard Nogai Cyrillic digraphs to prevent token fragmentation during BPE training.
"""

import re
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standardizes cross-lingual Turkic characters into native Nogai Digraphs
KIPCHAK_CONVERSIONS = {
    'Қ': 'Къ', 'Ғ': 'Гъ', 'Ң': 'Нъ', 'Ү': 'Уь', 'Ө': 'Оь', 'Ә': 'Аь', 'Ұ': 'У', 
    'І': 'И', 'Ї': 'И', 'Ў': 'У', 'Ҡ': 'Къ',
    'қ': 'къ', 'ғ': 'гъ', 'ң': 'нъ', 'ү': 'уь', 'ө': 'оь', 'ә': 'аь', 'ұ': 'у', 
    'і': 'и', 'ї': 'и', 'ў': 'у', 'ҡ': 'къ',
    'ӏ': 'I', 'Ӏ': 'I', 'ӑ': 'а', 'ӗ': 'е'
}

def clean_mediawiki_syntax(text: str) -> str:
    """Recursively strips MediaWiki formatting, templates, and link structures."""
    if not text: return ""
    
    text = re.sub(r'^#\w+\s*\[\[.*?\]\]', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Recursively remove nested templates {{...}}
    while True:
        new_text = re.sub(r'\{\{[^{}]*\}\}', '', text, flags=re.DOTALL)
        if new_text == text: break
        text = new_text

    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL) # Remove Tables
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text) # External links
    text = re.sub(r'\[https?://\S+\]', '', text) 
    text = text.replace("'''", "").replace("''", "") # Bold/Italics
    text = re.sub(r'==+\s*(.*?)\s*==+', r'\1', text) # Headers
    
    return re.sub(r'\n\s*\n+', '\n\n', text).strip()

def normalize_turkic_orthography(text: str) -> str:
    """Maps isolated Kipchak characters to unified Nogai representations."""
    for char, replacement in KIPCHAK_CONVERSIONS.items():
        text = text.replace(char, replacement)
        
    # Purge isolated unicode blocks (Arabic, CJK, Latin)
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', '', text)
    text = re.sub(r'[\u4E00-\u9FFF\u3400-\u4DBF]', '', text)
    text = re.sub(r'[a-zA-Z\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    
    return re.sub(r' +', ' ', text).strip()

def parse_xml_dump(xml_path: Path) -> Dict[str, str]:
    """Extracts raw text blocks from a MediaWiki XML namespace."""
    articles = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        
        for page in root.findall(f".//{ns}page"):
            title = page.find(f"{ns}title")
            revision = page.find(f"{ns}revision")
            if title is not None and revision is not None:
                text_elem = revision.find(f"{ns}text")
                if text_elem is not None and text_elem.text:
                    if not any(x in title.text.lower() for x in ["/doc", "/header"]):
                        articles[title.text] = text_elem.text
    except Exception as e:
        logger.error(f"XML Parsing failed: {e}")
        
    return articles

def main(xml_file: str, output_file: str):
    xml_path = Path(xml_file)
    if not xml_path.exists():
        logger.error(f"File {xml_path} not found.")
        return

    logger.info(f"Parsing XML database: {xml_path.name}")
    articles = parse_xml_dump(xml_path)
    
    logger.info(f"Normalizing {len(articles)} documents...")
    with open(output_file, "a", encoding="utf-8") as f:
        for title, raw_text in articles.items():
            clean_text = clean_mediawiki_syntax(raw_text)
            final_text = normalize_turkic_orthography(clean_text)
            if final_text:
                f.write(final_text + "\n\n")
                
    logger.info(f"Wiki processing complete. Appended to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wiki XML Parser for NogaiLLM")
    parser.add_argument("--xml_file", type=str, required=True, help="Input MediaWiki XML Dump")
    parser.add_argument("--output_file", type=str, default="combined.txt", help="Output text file")
    args = parser.parse_args()
    
    main(args.xml_file, args.output_file)