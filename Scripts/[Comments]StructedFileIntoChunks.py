# --- Imports ---
from itext2kg.documents_distiller import DocumentsDistiller, Article
import asyncio
from itext2kg import iText2KG_Star
from itext2kg.logging_config import setup_logging, get_logger
from itext2kg import itext2kg_star
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PDFMinerLoader, UnstructuredPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
import time
import gc
import json
import os
from typing import List, Optional
from datetime import datetime

# Optional: PyMuPDF (fitz) for more accurate PDF extraction
try:
    import fitz  # PyMuPDF for better PDF handling
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not available. Install with: pip install PyMuPDF")
    PYMUPDF_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# --- Data models for structured extraction ---
class ContentSource(BaseModel):
    """Represents a single source (author/speaker) in the document"""
    name: str = Field(description="Name of the author/speaker/presenter")
    role: Optional[str] = Field(description="Role or position")
    affiliation: Optional[str] = Field(description="Organization or company affiliation")


class Content(BaseModel):
    """Schema for extracting structured information from research/business/technical PDFs"""
    title: str = Field(description="Title of the content (document/paper)")
    sources: List[ContentSource] = Field(description="Authors involved")
    summary: str = Field(description="Brief summary of the content")
    key_concepts: List[str] = Field(description="Main concepts or topics covered")
    insights: str = Field(description="Key insights and findings")
    challenges: str = Field(description="Challenges or limitations discussed")
    solutions: str = Field(description="Proposed solutions or approaches")
    practical_applications: str = Field(description="Practical applications or implementations mentioned")
    methodology: str = Field(description="Research methodology or approach used")
    conclusions: str = Field(description="Main conclusions and recommendations")


# --- PDF extraction with PyMuPDF ---
def extract_pdf_with_pymupdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Fallback is handled if PyMuPDF is unavailable.
    """
    if not PYMUPDF_AVAILABLE:
        print(f"PyMuPDF not available, cannot extract {pdf_path}")
        return ""

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF with PyMuPDF: {str(e)}")
        return ""


# --- Document loader with multiple strategies ---
def load_pdf_documents(directory_path: str, loader_type: str = "pypdf") -> List:
    """
    Load PDF documents using different loaders (PyMuPDF, PyPDF, Unstructured, PDFMiner).
    Provides fallbacks if some libraries are missing.
    """
    documents = []

    # Validate directory
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        return documents

    # Find all PDF files
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {directory_path}")
        return documents

    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    # --- PyMuPDF loader ---
    if loader_type == "pymupdf":
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF not available, falling back to PyPDF")
            loader_type = "pypdf"
        else:
            # Custom document wrapper to mimic LangChain Document API
            for pdf_file in pdf_files:
                pdf_path = os.path.join(directory_path, pdf_file)
                text = extract_pdf_with_pymupdf(pdf_path)
                if text:
                    class Document:
                        def __init__(self, content, metadata):
                            self.page_content = content
                            self.metadata = metadata
                    documents.append(Document(text, {"source": pdf_file}))
                    print(f"Loaded {pdf_file}: {len(text)} characters")
                else:
                    print(f"Failed to extract text from {pdf_file}")
            return documents

    # --- Fallback loaders (PyPDF, Unstructured, PDFMiner) ---
    try:
        if loader_type == "pypdf":
            loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents using PyPDF")

        elif loader_type == "unstructured":
            try:
                loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
                documents = loader.load()
                print(f"Loaded {len(documents)} documents using Unstructured")
            except ImportError:
                print("Unstructured not available, falling back to PyPDF")
                loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()

        elif loader_type == "pdfminer":
            try:
                loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PDFMinerLoader)
                documents = loader.load()
                print(f"Loaded {len(documents)} documents using PDFMiner")
            except ImportError:
                print("PDFMiner not available, falling back to PyPDF")
                loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()

    except Exception as e:
        print(f"Error loading PDFs with {loader_type}: {str(e)}")
        print("Trying PyPDF as fallback...")
        try:
            loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents using PyPDF fallback")
        except Exception as fallback_error:
            print(f"All PDF loaders failed: {str(fallback_error)}")

    return documents


# --- PDF text cleaning ---
def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text (remove whitespace, headers, footers, page numbers).
    """
    import re
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Collapse excessive whitespace
    text = re.sub(r' +', ' ', text)          # Collapse multiple spaces
    text = re.sub(r'\n\d+\n', '\n', text)    # Remove isolated page numbers

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 3:  # Skip very short artifacts
            continue
        if line.isdigit():  # Skip numbers (page numbers)
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


# --- Batch processor ---
async def process_batches(pages, batch_size, max_retries=2, timeout_seconds=300, query_type="research"):
    """
    Process PDF pages in batches, extract structured content using LLM.
    Handles retries, timeouts, and saves results to JSON.
    """
    distilled_texts = []
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Process started at: {start_timestamp}")

    total_pages = len(pages)
    print(f"Processing {total_pages} pages in batches of {batch_size}")

    # Create LLM instance once
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    document_distiller = DocumentsDistiller(llm_model=llm)

    # Different IE queries depending on content type
    queries = {
        "research": '''# DIRECTIVES : 
                      - Act like an experienced research analyst.
                      - Extract concepts, methodologies, findings, insights, and conclusions.''',

        "business": '''# DIRECTIVES : 
                      - Act like an experienced business analyst.
                      - Extract strategies, market insights, and recommendations.''',

        "technical": '''# DIRECTIVES : 
                       - Act like an experienced technical analyst.
                       - Extract specifications, procedures, and best practices.'''
    }

    selected_query = queries.get(query_type, queries["research"])

    # --- Process each batch ---
    for i in range(0, total_pages, batch_size):
        batch_num = (i // batch_size) + 1
        end_idx = min(i + batch_size, total_pages)
        print(f"Processing batch {batch_num} (pages {i + 1}-{end_idx})...")

        current_batch = pages[i:end_idx]
        batch_success = False
        retry_count = 0

        # Retry loop
        while not batch_success and retry_count < max_retries:
            try:
                if retry_count > 0:
                    print(f"  Retry attempt {retry_count} for batch {batch_num}")
                    await asyncio.sleep(5)

                # Clean text for each page
                documents = []
                for page in current_batch:
                    cleaned_text = clean_pdf_text(page.page_content)
                    cleaned_text = cleaned_text.replace("{", '[').replace("}", "]")
                    documents.append(cleaned_text)

                # Run extraction with timeout
                distilled_text = await asyncio.wait_for(
                    document_distiller.distill(
                        documents=documents,
                        IE_query=selected_query,
                        output_data_structure=Content
                    ),
                    timeout=timeout_seconds
                )

                distilled_texts.append(distilled_text)
                batch_success = True

                # Save results to JSON
                output_dir = "distilled_results"
                os.makedirs(output_dir, exist_ok=True)
                json_data = distilled_text.model_dump_json()
                batch_filename = os.path.join(output_dir, f"batch_{batch_num}.json")
                with open(batch_filename, "w", encoding="utf-8") as f:
                    f.write(json_data)

                print(f"Batch {batch_num} completed successfully")

            except asyncio.TimeoutError:
                print(f"Batch {batch_num} timed out after {timeout_seconds} seconds")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Batch {batch_num} failed after {max_retries} attempts - skipping")
                    distilled_texts.append(None)

            except Exception as e:
                print(f"Error in batch {batch_num}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Batch {batch_num} failed after {max_retries} attempts - skipping")
                    distilled_texts.append(None)

        elapsed_time = time.time() - start_time
        print(f"Elapsed: {elapsed_time:.2f} sec ({elapsed_time / 60:.2f} min)")

        gc.collect()  # Free memory
        await asyncio.sleep(3)  # Prevent overload

    print(f"\nProcess finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total batches processed: {len(distilled_texts)}")
    return distilled_texts


# --- Debugging a single PDF file ---
async def process_single_pdf_debug(pdf_path: str, query_type: str = "research"):
    """
    Debug function: process a single PDF file and show preview of extracted text.
    """
    print(f"Debug processing PDF: {pdf_path}")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    document_distiller = DocumentsDistiller(llm_model=llm)

    try:
        text = extract_pdf_with_pymupdf(pdf_path)
        cleaned_text = clean_pdf_text(text).replace("{", '[').replace("}", "]")

        print(f"PDF contains {len(cleaned_text)} characters")
        print(f"Preview: {cleaned_text[:200]}...")

        queries = {
            "research": '''# DIRECTIVES : Extract research methodology, findings, and conclusions.''',
            "business": '''# DIRECTIVES : Extract business strategies, insights, and recommendations.''',
            "technical": '''# DIRECTIVES : Extract technical procedures and best practices.'''
        }
        selected_query = queries.get(query_type, queries["research"])

        start_time = time.time()
        distilled_text = await asyncio.wait_for(
            document_distiller.distill(
                documents=[cleaned_text],
                IE_query=selected_query,
                output_data_structure=Content
            ),
            timeout=300
        )
        print(f"PDF processed in {time.time() - start_time:.2f} sec")
        return distilled_text

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None


# --- Merge multiple batch results ---
def merge_batch_results(output_dir: str = "novco_results") -> dict:
    """
    Merge all batch JSON files into a single summary JSON file.
    Tracks successes and failures.
    """
    merged_results = {
        "processing_summary": {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "extracted_content": []
    }

    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return merged_results

    batch_files = [f for f in os.listdir(output_dir) if f.startswith("batch_") and f.endswith(".json")]
    batch_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for batch_file in batch_files:
        try:
            with open(os.path.join(output_dir, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                merged_results["extracted_content"].append(batch_data)
                merged_results["processing_summary"]["successful_batches"] += 1
        except Exception as e:
            print(f"Error reading {batch_file}: {str(e)}")
            merged_results["processing_summary"]["failed_batches"] += 1
        merged_results["processing_summary"]["total_batches"] += 1

    merged_filename = os.path.join(output_dir, "merged_results.json")
    with open(merged_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"Merged results saved to {merged_filename}")
    return merged_results


# --- Main script ---
if __name__ == "__main__":
    PDF_DIRECTORY = "../data/pdfs/"   # Input directory
    BATCH_SIZE = 5                    # Batch size for processing
    LOADER_TYPE = "pypdf"             # Loader option: pymupdf, pypdf, unstructured, pdfminer
    QUERY_TYPE = "research"           # Context type
    CHUNK_SIZE = 2000                 # Split size for large docs
    CHUNK_OVERLAP = 400               # Overlap to preserve context

    print(f"Loading PDFs from: {PDF_DIRECTORY}")
    documents = load_pdf_documents(PDF_DIRECTORY, LOADER_TYPE)
    if not documents:
        print("No PDFs found, exiting")
        exit(1)

    # Split large docs into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]  # PDF-aware splitting
    )
    pages = text_splitter.split_documents(documents)
    print(f"Split into {len(pages)} chunks")

    # Run batch extraction
    distilled_results = asyncio.run(process_batches(pages, BATCH_SIZE, query_type=QUERY_TYPE))

    # Print processing summary
    successful_batches = sum(1 for r in distilled_results if r is not None)
    failed_batches = len(distilled_results) - successful_batches
    print(f"\nSummary: {successful_batches} successful, {failed_batches} failed")

    # Merge results into single file
    merged_results = merge_batch_results("novco_results")
    print(f"Merged {merged_results['processing_summary']['successful_batches']} successful batches")
