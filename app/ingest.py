"""Document ingestion pipeline for HTML, PDF, and CSV."""
import os
import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.utils.embeddings import generate_fake_embeddings
from app.utils.ocr import extract_pdf_text

load_dotenv()


class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the chunk."""
        content = f"{self.text}_{self.metadata.get('source', '')}_{self.metadata.get('chunk_index', 0)}"
        return hashlib.md5(content.encode()).hexdigest()


class DocumentIngestion:
    """Handles document ingestion and processing."""
    
    def __init__(self):
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIM", "1536"))
        self.use_fake_embeddings = os.getenv("USE_FAKE_EMBEDDINGS", "true").lower() != "false"

        self.openai_client = None if self.use_fake_embeddings else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_ocr_model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
        self.mistral_ocr_endpoint = os.getenv("MISTRAL_OCR_ENDPOINT", "https://api.mistral.ai/v1/ocr")
        self.use_fake_ocr = os.getenv("USE_FAKE_OCR", "true").lower() == "true"
        
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self.chunks: List[DocumentChunk] = []
    
    def parse_html(self, file_path: str) -> List[DocumentChunk]:
        """Parse HTML file and create chunks based on sections."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        chunks = []
        chunk_index = 0
        
        # Extract title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else "Q2 Research Letter"
        
        # Process main content sections
        h1_tags = soup.find_all('h1')
        h2_tags = soup.find_all('h2')
        
        # Chunk by h2 sections (main sections)
        for h2 in h2_tags:
            section_title = h2.get_text(strip=True)
            section_content = [section_title]
            
            # Collect content until next h2
            for sibling in h2.find_next_siblings():
                if sibling.name == 'h2':
                    break
                
                text = sibling.get_text(strip=True)
                if text:
                    section_content.append(text)
            
            chunk_text = "\n".join(section_content)
            
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                metadata = {
                    "source": Path(file_path).name,
                    "doc_type": "html",
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                    "document_title": title_text
                }
                chunks.append(DocumentChunk(chunk_text, metadata))
                chunk_index += 1
        
        # Also create chunks for tables
        tables = soup.find_all('table')
        for idx, table in enumerate(tables):
            # Find the preceding h2 for context
            prev_h2 = None
            for prev in table.find_all_previous(['h2']):
                prev_h2 = prev
                break
            
            section_title = prev_h2.get_text(strip=True) if prev_h2 else "Table"
            
            # Convert table to text
            table_text = self._table_to_text(table)
            
            if len(table_text.strip()) > 20:
                metadata = {
                    "source": Path(file_path).name,
                    "doc_type": "html",
                    "section_title": f"{section_title} - Table {idx + 1}",
                    "chunk_index": chunk_index,
                    "document_title": title_text
                }
                chunks.append(DocumentChunk(table_text, metadata))
                chunk_index += 1
        
        print(f"Parsed HTML: {len(chunks)} chunks from {file_path}")
        return chunks
    
    def _table_to_text(self, table) -> str:
        """Convert HTML table to readable text."""
        rows = []
        
        # Header
        headers = []
        thead = table.find('thead')
        if thead:
            for th in thead.find_all('th'):
                headers.append(th.get_text(strip=True))
            rows.append(" | ".join(headers))
        
        # Body
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                rows.append(" | ".join(cells))
        
        # Footer
        tfoot = table.find('tfoot')
        if tfoot:
            for tr in tfoot.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                rows.append(" | ".join(cells))
        
        return "\n".join(rows)
    
    def parse_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Parse PDF file using Mistral OCR and create chunks."""
        chunks: List[DocumentChunk] = []
        chunk_index = 0

        try:
            full_text = extract_pdf_text(
                file_path=file_path,
                api_key=self.mistral_api_key,
                model=self.mistral_ocr_model,
                endpoint=self.mistral_ocr_endpoint,
                use_fallback=self.use_fake_ocr
            )
        except Exception as exc:
            print(f"Warning: OCR failed for {file_path}: {exc}")
            full_text = ""

        if not full_text.strip():
            return chunks

        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]

        current_chunk: List[str] = []
        current_length = 0
        target_chunk_size = 500

        for paragraph in paragraphs:
            para_length = len(paragraph)

            if current_length + para_length > target_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                metadata = {
                    "source": Path(file_path).name,
                    "doc_type": "pdf",
                    "chunk_index": chunk_index,
                }
                chunks.append(DocumentChunk(chunk_text, metadata))
                chunk_index += 1
                current_chunk = [paragraph]
                current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            metadata = {
                "source": Path(file_path).name,
                "doc_type": "pdf",
                "chunk_index": chunk_index,
            }
            chunks.append(DocumentChunk(chunk_text, metadata))

        print(f"Parsed PDF: {len(chunks)} chunks from {file_path}")
        return chunks
    
    def parse_csv(self, file_path: str) -> List[DocumentChunk]:
        """Parse CSV chat log and create chunks (one per message)."""
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                timestamp = row.get('timestamp', '')
                author = row.get('author', '')
                text = row.get('text', '')
                
                # Create chunk text with context
                chunk_text = f"[{timestamp}] {author}: {text}"
                
                metadata = {
                    "source": Path(file_path).name,
                    "doc_type": "csv",
                    "timestamp": timestamp,
                    "author": author,
                    "chunk_index": idx
                }
                chunks.append(DocumentChunk(chunk_text, metadata))
        
        print(f"Parsed CSV: {len(chunks)} chunks from {file_path}")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.use_fake_embeddings or self.openai_client is None:
            return generate_fake_embeddings(texts, dim=self.embedding_dimensions)

        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]
    
    def setup_qdrant_collection(self):
        """Create or recreate Qdrant collection."""
        # Delete existing collection if it exists
        collections = self.qdrant_client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            self.qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        
        # Create new collection
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dimensions, distance=Distance.COSINE)
        )
        print(f"Created collection: {self.collection_name}")
    
    def store_chunks(self, chunks: List[DocumentChunk]):
        """Store chunks in Qdrant."""
        if not chunks:
            return
        
        # Generate embeddings in batches
        batch_size = 100
        all_points = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch_chunks]
            embeddings = self.generate_embeddings(texts)
            
            for chunk, embedding in zip(batch_chunks, embeddings):
                point = PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "metadata": chunk.metadata
                    }
                )
                all_points.append(point)
            
            print(f"Generated embeddings for batch {i // batch_size + 1}")
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=all_points
        )
        print(f"Stored {len(all_points)} chunks in Qdrant")
    
    def ingest_all_documents(self, data_dir: str = "data"):
        """Ingest all documents from the data directory."""
        data_path = Path(data_dir)
        
        # Parse HTML
        html_files = list((data_path / "fund_letters").glob("*.html"))
        for html_file in html_files:
            chunks = self.parse_html(str(html_file))
            self.chunks.extend(chunks)
        
        # Parse PDF
        pdf_files = list((data_path / "fund_letters").glob("*.pdf"))
        for pdf_file in pdf_files:
            chunks = self.parse_pdf(str(pdf_file))
            self.chunks.extend(chunks)
        
        # Parse CSV
        csv_files = list((data_path / "chat_logs").glob("*.csv"))
        for csv_file in csv_files:
            chunks = self.parse_csv(str(csv_file))
            self.chunks.extend(chunks)
        
        print(f"\nTotal chunks created: {len(self.chunks)}")
        
        # Setup Qdrant and store chunks
        self.setup_qdrant_collection()
        self.store_chunks(self.chunks)
        
        print("\n✓ Ingestion complete!")


def main():
    """Main ingestion entry point."""
    print("Starting document ingestion...")
    ingestion = DocumentIngestion()
    ingestion.ingest_all_documents()


if __name__ == "__main__":
    main()
