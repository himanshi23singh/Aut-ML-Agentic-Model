"""
Input Handler Module
Handles file uploads and raw text input with validation and metadata logging.
"""
import os
import io
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib


class InputHandler:
    """
    Handles input processing for text, files (TXT, PDF, DOCX).
    Provides glass-box transparency through detailed metadata logging.
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self):
        self.last_input_metadata = None
    
    async def process_input(
        self, 
        text: Optional[str] = None,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process input from text or file.
        
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        metadata = {
            "processed_at": datetime.utcnow().isoformat(),
            "input_type": None,
            "original_length": 0,
            "file_info": None,
            "validation_passed": False,
            "processing_steps": []
        }
        
        try:
            if text:
                # Direct text input
                extracted_text = text
                metadata["input_type"] = "raw_text"
                metadata["original_length"] = len(text)
                metadata["processing_steps"].append("Received raw text input")
                
            elif file_content and filename:
                # File upload
                ext = os.path.splitext(filename)[1].lower()
                
                if ext not in self.SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported file type: {ext}")
                
                if len(file_content) > self.MAX_FILE_SIZE:
                    raise ValueError(f"File too large. Max size: {self.MAX_FILE_SIZE // (1024*1024)}MB")
                
                metadata["file_info"] = {
                    "filename": filename,
                    "extension": ext,
                    "size_bytes": len(file_content),
                    "hash": hashlib.md5(file_content).hexdigest()[:8]
                }
                
                # Extract text based on file type
                if ext == '.txt':
                    extracted_text = await self._process_txt(file_content, metadata)
                elif ext == '.pdf':
                    extracted_text = await self._process_pdf(file_content, metadata)
                elif ext in ['.docx', '.doc']:
                    extracted_text = await self._process_docx(file_content, metadata)
                else:
                    raise ValueError(f"Cannot process file type: {ext}")
                
                metadata["input_type"] = ext[1:]  # Remove the dot
                
            else:
                raise ValueError("No input provided. Supply either text or file.")
            
            # Validate extracted text
            if not extracted_text or not extracted_text.strip():
                raise ValueError("Extracted text is empty")
            
            metadata["original_length"] = len(extracted_text)
            metadata["validation_passed"] = True
            metadata["processing_steps"].append("Input validation passed")
            
            self.last_input_metadata = metadata
            return extracted_text, metadata
            
        except Exception as e:
            metadata["error"] = str(e)
            metadata["validation_passed"] = False
            self.last_input_metadata = metadata
            raise
    
    async def _process_txt(self, content: bytes, metadata: Dict) -> str:
        """Process plain text file."""
        metadata["processing_steps"].append("Decoding TXT file (UTF-8)")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                text = content.decode(encoding)
                metadata["processing_steps"].append(f"Successfully decoded with {encoding}")
                metadata["file_info"]["encoding"] = encoding
                return text
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode text file with supported encodings")
    
    async def _process_pdf(self, content: bytes, metadata: Dict) -> str:
        """Process PDF file using PyMuPDF."""
        metadata["processing_steps"].append("Processing PDF with PyMuPDF")
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(stream=content, filetype="pdf")
            text_parts = []
            
            metadata["file_info"]["page_count"] = len(doc)
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text_parts.append(page_text)
                metadata["processing_steps"].append(f"Extracted page {page_num + 1}/{len(doc)}")
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            metadata["processing_steps"].append(f"PDF extraction complete: {len(full_text)} chars")
            
            return full_text
            
        except ImportError:
            metadata["processing_steps"].append("PyMuPDF not available, using fallback")
            raise ValueError("PDF processing requires PyMuPDF. Install with: pip install pymupdf")
    
    async def _process_docx(self, content: bytes, metadata: Dict) -> str:
        """Process DOCX file using python-docx."""
        metadata["processing_steps"].append("Processing DOCX with python-docx")
        
        try:
            from docx import Document
            
            doc = Document(io.BytesIO(content))
            text_parts = []
            
            paragraph_count = 0
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
                    paragraph_count += 1
            
            metadata["file_info"]["paragraph_count"] = paragraph_count
            
            full_text = "\n\n".join(text_parts)
            metadata["processing_steps"].append(f"DOCX extraction complete: {len(full_text)} chars")
            
            return full_text
            
        except ImportError:
            metadata["processing_steps"].append("python-docx not available")
            raise ValueError("DOCX processing requires python-docx. Install with: pip install python-docx")
    
    def get_input_summary(self) -> Dict[str, Any]:
        """Get a summary of the last processed input for dashboard display."""
        if not self.last_input_metadata:
            return {"status": "No input processed yet"}
        
        meta = self.last_input_metadata
        return {
            "type": meta.get("input_type"),
            "length": meta.get("original_length"),
            "file": meta.get("file_info", {}).get("filename"),
            "valid": meta.get("validation_passed"),
            "steps": len(meta.get("processing_steps", []))
        }
