from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .xlsx_parser import XLSXParser
from .csv_parser import CSVParser
from .pptx_parser import PPTXParser
from .html_parser import HTMLParser
from .txt_parser import TXTParser

__all__ = [
    "PDFParser",
    "DOCXParser",
    "XLSXParser",
    "CSVParser",
    "PPTXParser",
    "HTMLParser",
    "TXTParser",
]