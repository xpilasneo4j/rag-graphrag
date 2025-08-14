import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from PIL import Image
import pandas as pd
import re
from typing import List
from pathlib import Path
import logging
import io
import os
import configparser
import sys

if len(sys.argv) != 2:
    sys.exit("Please provide as an argument a env file built on the template.env provided")
else:
    ENV_FILE = sys.argv[1]

# Config loading
config = configparser.ConfigParser()
config.read(ENV_FILE)

pytesseract.pytesseract.tesseract_cmd = config.get('Conf','tesseract_path')
FILES_PATH = config.get('Conf','files_path')
OUTPUT_PATH = config.get('Conf','output_path')

class PDFTextExtractor:
    """PDF text extractor with OCR and table processing"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # OCR configuration
        self.ocr_config = '--oem 3 --psm 6'

    def extract_complete_text(self) -> str:
        """Extract all text from PDF including OCR and tables"""
        try:
            self.doc = fitz.open(self.pdf_path)
            complete_text = []

            for page_num in range(len(self.doc)):
                self.logger.info(f"Processing page {page_num + 1}/{len(self.doc)}")

                # Add page header
                complete_text.append(f"\n{'='*80}")
                complete_text.append(f"PAGE {page_num + 1}")
                complete_text.append(f"{'='*80}\n")

                # Extract text content
                text_content = self._extract_text_with_layout(page_num)

                # Extract tables
                tables = self._extract_tables(page_num)

                # Extract text from images
                image_text = self._extract_images_with_ocr(page_num)

                # Combine all content
                page_content = self._combine_page_content(text_content, tables, image_text)

                # Clean the text
                cleaned_content = self._clean_text(page_content)

                complete_text.append(cleaned_content)
                complete_text.append(f"\n{'-'*80}\n")

            return '\n'.join(complete_text)

        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""

    def _extract_text_with_layout(self, page_num: int) -> str:
        """Extract text preserving layout structure"""
        try:
            # Try pdfplumber first for better layout
            with pdfplumber.open(self.pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text(layout=True)

                    if not text or text.strip() == "":
                        # Fallback to PyMuPDF
                        page_pymupdf = self.doc.load_page(page_num)
                        text = page_pymupdf.get_text()

                        # If still no text, try OCR
                        if not text or text.strip() == "":
                            text = self._ocr_page(page_num)

                    return text or ""

        except Exception as e:
            self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            # Final fallback
            page = self.doc.load_page(page_num)
            return page.get_text()

    def _extract_tables(self, page_num: int) -> List[str]:
        """Extract tables as formatted text"""
        table_texts = []

        try:
            # Try camelot first
            try:
                camelot_tables = camelot.read_pdf(
                    self.pdf_path,
                    pages=str(page_num + 1),
                    flavor='lattice'
                )

                if len(camelot_tables) == 0:
                    camelot_tables = camelot.read_pdf(
                        self.pdf_path,
                        pages=str(page_num + 1),
                        flavor='stream'
                    )

                for i, table in enumerate(camelot_tables):
                    if table.accuracy > 50:
                        table_text = f"\n--- TABLE {i+1} (Accuracy: {table.accuracy:.1f}%) ---\n"
                        table_text += table.df.to_string(index=False)
                        table_text += f"\n--- END TABLE {i+1} ---\n"
                        table_texts.append(table_text)

            except Exception as e:
                self.logger.debug(f"Camelot failed for page {page_num + 1}: {e}")

            # Try pdfplumber as backup
            if len(table_texts) == 0:
                try:
                    with pdfplumber.open(self.pdf_path) as pdf:
                        if page_num < len(pdf.pages):
                            page = pdf.pages[page_num]
                            page_tables = page.extract_tables()

                            for i, table in enumerate(page_tables):
                                if table and len(table) > 1:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    table_text = f"\n--- TABLE {i+1} ---\n"
                                    table_text += df.to_string(index=False)
                                    table_text += f"\n--- END TABLE {i+1} ---\n"
                                    table_texts.append(table_text)

                except Exception as e:
                    self.logger.debug(f"Pdfplumber table extraction failed for page {page_num + 1}: {e}")

        except Exception as e:
            self.logger.warning(f"Table extraction failed for page {page_num + 1}: {e}")

        return table_texts

    def _extract_images_with_ocr(self, page_num: int) -> List[str]:
        """Extract text from images using OCR"""
        image_texts = []

        try:
            page = self.doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(self.doc, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")

                        # Convert to PIL Image for OCR
                        pil_image = Image.open(io.BytesIO(img_data))

                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(
                            pil_image,
                            config=self.ocr_config
                        ).strip()

                        if ocr_text and len(ocr_text) > 10:  # Only meaningful text
                            image_text = f"\n--- IMAGE {img_index+1} TEXT ---\n"
                            image_text += ocr_text
                            image_text += f"\n--- END IMAGE {img_index+1} ---\n"
                            image_texts.append(image_text)

                    pix = None  # Free memory

                except Exception as e:
                    self.logger.debug(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Image extraction failed for page {page_num + 1}: {e}")

        return image_texts

    def _ocr_page(self, page_num: int) -> str:
        """Perform OCR on entire page"""
        try:
            page = self.doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(img_data))

            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                pil_image,
                config=self.ocr_config
            )

            return ocr_text.strip()

        except Exception as e:
            self.logger.warning(f"OCR failed for page {page_num + 1}: {e}")
            return ""

    def _combine_page_content(self, text: str, tables: List[str], images: List[str]) -> str:
        """Combine all content from the page"""
        combined = []

        # Add main text
        if text and text.strip():
            combined.append("MAIN TEXT:")
            combined.append(text)

        # Add tables
        if tables:
            combined.append("\nTABLES FOUND ON THIS PAGE:")
            combined.extend(tables)

        # Add image text
        if images:
            combined.append("\nTEXT FROM IMAGES:")
            combined.extend(images)

        return '\n'.join(combined)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR/PDF issues
        replacements = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            '–': '-', '—': '-', ''': "'", ''': "'", '"': '"', '"': '"',
            '…': '...', 'â€™': "'", 'â€œ': '"', 'â€': '"', 'Â': ' ',
            '\u00a0': ' ', '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-', '\u2026': '...'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove page headers/footers
        #text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        #text = re.sub(r'AMP.*2023 Annual report', '', text, flags=re.IGNORECASE)

        # Clean up spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def save_text(self, doc: str, output_path: str = "extracted_text.txt"):
        """Extract and save complete text to file"""
        self.logger.info("Starting text extraction...")
        complete_text = self.extract_complete_text()

        # Add document header
        header = f"""{doc} - COMPLETE TEXT EXTRACTION
{'='*80}
Document: {doc}
Extraction Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Pages: {len(self.doc) if self.doc else 'Unknown'}
{'='*80}

"""

        final_text = header + complete_text

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        self.logger.info(f"Text extraction completed. Saved to: {output_path}")
        self.logger.info(f"Total characters extracted: {len(final_text):,}")

        return final_text

    def save_structured_text(self, output_dir: str):
        """Save text in structured format with separate sections"""
        Path(output_dir).mkdir(exist_ok=True)

        complete_text = self.extract_complete_text()

        # Save complete text
        with open(f"{output_dir}" + os.sep + "complete_text.txt", 'w', encoding='utf-8') as f:
            f.write(complete_text)

        # Split by pages and save separately
        pages = complete_text.split('='*80)

        for i, page_content in enumerate(pages[1:], 1):  # Skip header
            if page_content.strip():
                with open(f"{output_dir}" + os.sep + "page_{i:03d}.txt", 'w', encoding='utf-8') as f:
                    f.write(f"PAGE {i}\n{'='*40}\n{page_content.strip()}")

        # Generate summary
        summary = self._generate_summary(complete_text)
        with open(f"{output_dir}" + os.sep + "extraction_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)

        self.logger.info(f"Structured extraction completed. Files saved in: {output_dir}/")

        return complete_text

    def _generate_summary(self, text: str) -> str:
        """Generate extraction summary"""
        lines = text.split('\n')
        total_lines = len(lines)
        total_chars = len(text)
        total_words = len(text.split())

        # Count tables and images
        table_count = text.count('--- TABLE')
        image_count = text.count('--- IMAGE')
        page_count = text.count('PAGE ')

        summary = f"""EXTRACTION SUMMARY
{'='*50}
Total Pages Processed: {page_count}
Total Lines: {total_lines:,}
Total Characters: {total_chars:,}
Total Words: {total_words:,}
Tables Found: {table_count}
Images with Text: {image_count}

EXTRACTION METHODS USED:
- Text Extraction: pdfplumber + PyMuPDF
- Table Extraction: Camelot + pdfplumber
- Image Text: Tesseract OCR
- Layout Preservation: Yes
- Character Normalization: Yes

FILE STRUCTURE:
- complete_text.txt: All extracted content
- page_XXX.txt: Individual page content
- extraction_summary.txt: This summary
"""
        return summary

def extract_pdf_text(pdf_path: str, cpt: int, output_path: str):
    """Simple function to extract text from PDF"""
    extractor = PDFTextExtractor(pdf_path)
    return extractor.save_text(pdf_path, str(cpt) + "-" + output_path)

def extract_pdf_structured(pdf_path: str, cpt: int, output_dir: str):
    """Extract PDF text in structured format"""
    extractor = PDFTextExtractor(pdf_path)
    return extractor.save_structured_text(output_dir + "-" + str(cpt))

def print_requirements():
    """Print installation requirements"""
    requirements = """
INSTALLATION REQUIREMENTS:
=========================

Python packages:
pip install PyMuPDF pdfplumber camelot-py[cv] pytesseract pandas pillow

System dependencies:
- Ubuntu/Debian: sudo apt-get install tesseract-ocr ghostscript
- macOS: brew install tesseract ghostscript  
- Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
           Download Ghostscript from https://www.ghostscript.com/download/

USAGE:
======
# Simple text extraction
extract_pdf_text("document.pdf", "output.txt")

# Structured extraction (separate files)
extract_pdf_structured("document.pdf", cpt, "output_folder")
"""
    print(requirements)

def list_files_by_type_os(directory, file_extension):
    """
    Lists files of a specific type in a given directory using os.listdir().

    Args:
        directory (str): The path to the directory.
        file_extension (str): The desired file extension (e.g., '.txt', '.py').

    Returns:
        list: A list of filenames matching the specified extension.
    """
    matching_files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(file_extension):
            matching_files.append(filename)
    return matching_files

# Main execution
if __name__ == "__main__":
    print_requirements()
    cpt = 1
    # Example usage

    files = list_files_by_type_os(FILES_PATH, ".pdf")
    for pdf_path in files:
        try:
            # Simple extraction
            print("\n" + "="*60)
            print("SIMPLE TEXT EXTRACTION")
            print("="*60)
            text = extract_pdf_text(FILES_PATH+pdf_path, cpt, OUTPUT_PATH+os.sep+"extracted_content")
            print(f"✓ Extraction completed for file {cpt} {pdf_path}")

            # Structured extraction
            print("\n" + "="*60)
            print("STRUCTURED TEXT EXTRACTION")
            print("="*60)
            extract_pdf_structured(FILES_PATH+pdf_path, cpt, OUTPUT_PATH+os.sep+"extracted_content")
            print(f"✓ Structured extraction completed: {cpt} {pdf_path}")
            cpt = cpt + 1
        except FileNotFoundError:
            print(f"Error: PDF file '{FILES_PATH+pdf_path}' not found.")
            print("Please ensure the PDF file exists in the current directory.")
        except Exception as e:
            print(f"Error during extraction: {e}")
            print("Please ensure all dependencies are installed correctly.")