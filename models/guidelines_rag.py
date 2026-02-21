"""
Guidelines RAG System - Retrieval-Augmented Generation for clinical guidelines
Uses FAISS for vector similarity search on chunked guideline PDFs.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# Paths
GUIDELINES_DIR = Path(__file__).parent.parent / "guidelines"
INDEX_DIR = GUIDELINES_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# Chunking parameters
CHUNK_SIZE = 500  # tokens (approximate)
CHUNK_OVERLAP = 50  # tokens overlap between chunks


class GuidelinesRAG:
    """
    RAG system for clinical guidelines.
    Extracts text from PDFs, chunks it, creates embeddings, and provides search.
    """

    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedder = None
        self.loaded = False

    def _load_embedder(self):
        """Load sentence transformer model for embeddings"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file"""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            # Fallback to PyPDF2
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'\n\d+\s*\n', '\n', text)
        # Fix broken words from line breaks
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        return text.strip()

    def _extract_pdf_with_pages(self, pdf_path: Path) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        pages.append((page_text, i))
            return pages
        except ImportError:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    pages.append((text, i))
            return pages

    def _chunk_text(self, text: str, source: str, page_num: int = 0) -> List[Dict]:
        """
        Chunk text into overlapping segments.
        Returns list of dicts with 'text', 'source', 'chunk_id', 'page'.
        """
        # Approximate tokens by words (rough estimate: 1 token ≈ 0.75 words)
        words = text.split()
        chunk_words = int(CHUNK_SIZE * 0.75)
        overlap_words = int(CHUNK_OVERLAP * 0.75)

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + chunk_words
            chunk_text = ' '.join(words[start:end])

            # Try to end at sentence boundary
            if end < len(words):
                last_period = chunk_text.rfind('.')
                if last_period > len(chunk_text) * 0.7:
                    chunk_text = chunk_text[:last_period + 1]

            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': chunk_id,
                'page': page_num
            })

            start = end - overlap_words
            chunk_id += 1

        return chunks

    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build FAISS index from guideline PDFs.
        Returns True if index was built, False if loaded from cache.
        """
        # Check if index already exists
        if not force_rebuild and FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
            return self.load_index()

        print("Building guidelines index...")
        self._load_embedder()

        # Create index directory
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # Extract and chunk all PDFs with page tracking
        all_chunks = []
        pdf_files = list(GUIDELINES_DIR.glob("*.pdf"))

        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")
            pages = self._extract_pdf_with_pages(pdf_path)
            pdf_chunks = 0
            for page_text, page_num in pages:
                cleaned = self._clean_text(page_text)
                chunks = self._chunk_text(cleaned, pdf_path.name, page_num)
                all_chunks.extend(chunks)
                pdf_chunks += len(chunks)
            print(f"    -> {pdf_chunks} chunks from {len(pages)} pages")

        if not all_chunks:
            print("No chunks extracted from PDFs!")
            return False

        self.chunks = all_chunks
        print(f"Total chunks: {len(self.chunks)}")

        # Generate embeddings
        print("Generating embeddings...")
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Build FAISS index
        import faiss
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Save index and chunks
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, 'w') as f:
            json.dump(self.chunks, f)

        print(f"Index saved to {INDEX_DIR}")
        self.loaded = True
        return True

    def load_index(self) -> bool:
        """Load persisted FAISS index and chunks"""
        if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            return False

        import faiss
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNKS_PATH, 'r') as f:
            self.chunks = json.load(f)

        self._load_embedder()
        self.loaded = True
        return True

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant guideline chunks.
        Returns list of chunks with similarity scores.
        """
        if not self.loaded:
            if not self.load_index():
                self.build_index()

        import faiss

        # Encode query
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)

        return results

    def get_management_context(self, diagnosis: str, features: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Get formatted context from guidelines for management recommendations.
        Returns tuple of (context_string, references_list).
        References can be used for citation hyperlinks.
        """
        # Build search query
        query = f"{diagnosis} management treatment recommendations"
        if features:
            query += f" {features}"

        chunks = self.search(query, k=5)

        if not chunks:
            return "No relevant guidelines found.", []

        # Build context and collect references
        context_parts = []
        references = []

        # Unicode superscript digits
        superscripts = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

        for i, chunk in enumerate(chunks, 1):
            source = chunk['source'].replace('.pdf', '')
            page = chunk.get('page', 0)
            ref_id = f"ref{i}"
            superscript = superscripts[i-1] if i <= len(superscripts) else f"[{i}]"

            # Add reference marker with superscript
            context_parts.append(f"[Source {superscript}] {chunk['text']}")

            # Collect reference info
            references.append({
                'id': ref_id,
                'source': source,
                'page': page,
                'file': chunk['source'],
                'score': chunk.get('score', 0)
            })

        context = "\n\n".join(context_parts)
        return context, references

    def format_references_for_prompt(self, references: List[Dict]) -> str:
        """Format references for inclusion in LLM prompt"""
        if not references:
            return ""

        lines = ["\n**References:**"]
        for ref in references:
            lines.append(f"[{ref['id']}] {ref['source']}, p.{ref['page']}")
        return "\n".join(lines)

    def format_references_for_display(self, references: List[Dict]) -> str:
        """
        Format references with markers that frontend can parse into hyperlinks.
        Uses format: [REF:id:source:page:file:superscript]
        """
        if not references:
            return ""

        # Unicode superscript digits
        superscripts = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

        lines = ["\n[REFERENCES]"]
        for i, ref in enumerate(references, 1):
            superscript = superscripts[i-1] if i <= len(superscripts) else f"[{i}]"
            # Format: [REF:ref1:Melanoma Guidelines:5:melanoma.pdf:¹]
            lines.append(f"[REF:{ref['id']}:{ref['source']}:{ref['page']}:{ref['file']}:{superscript}]")
        lines.append("[/REFERENCES]")
        return "\n".join(lines)


# Singleton instance
_rag_instance = None


def get_guidelines_rag() -> GuidelinesRAG:
    """Get or create RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GuidelinesRAG()
    return _rag_instance


if __name__ == "__main__":
    print("=" * 60)
    print("  Guidelines RAG System - Index Builder")
    print("=" * 60)

    rag = GuidelinesRAG()

    # Build or rebuild index
    import sys
    force = "--force" in sys.argv
    rag.build_index(force_rebuild=force)

    # Test search
    print("\n" + "=" * 60)
    print("  Testing Search")
    print("=" * 60)

    test_queries = [
        "melanoma management",
        "actinic keratosis treatment",
        "surgical excision margins"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag.search(query, k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:100]}...")
