# --- Imports ---
from itext2kg.documents_distiller import DocumentsDistiller, Article
import asyncio
from itext2kg import iText2KG_Star
from itext2kg.logging_config import setup_logging, get_logger
from itext2kg import itext2kg_star
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
import time
import gc
import json
import os
from typing import List, Optional
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator


# --- Data models for structured output ---
class ContentSource(BaseModel):
    """Represents a single source (author/speaker) in the content"""
    name: str = Field(description="Name of the author/speaker/presenter")
    role: Optional[str] = Field(description="Role or position")
    affiliation: Optional[str] = Field(description="Organization or company affiliation")


class Content(BaseModel):
    """Schema for extracting structured information from transcripts/articles"""
    title: str = Field(description="Title of the content (article/video)")
    sources: List[ContentSource] = Field(description="Authors/speakers involved")
    summary: str = Field(description="Brief summary of the content")
    key_concepts: List[str] = Field(description="Main concepts or topics covered")
    insights: str = Field(description="Key insights and findings")
    challenges: str = Field(description="Challenges or limitations discussed")
    solutions: str = Field(description="Proposed solutions or approaches")
    practical_applications: str = Field(description="Practical applications or implementations mentioned")


# --- Main batch processor ---
async def process_batches(pages, batch_size, max_retries=1, timeout_seconds=300):
    """
    Process transcript pages in batches using DocumentsDistiller.
    Each batch is passed to an LLM for information extraction (IE)
    and results are saved as JSON files.
    """

    distilled_texts = []  # Store results from all batches
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Process started at: {start_timestamp}")

    total_pages = len(pages)
    print(f"Processing {total_pages} pages in batches of {batch_size}")

    # Create the LLM instance once to reuse across batches
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    document_distiller = DocumentsDistiller(llm_model=llm)

    # Iterate through batches of pages
    for i in range(0, total_pages, batch_size):
        batch_num = (i // batch_size) + 1
        end_idx = min(i + batch_size, total_pages)

        print(f"Processing batch {batch_num} (pages {i + 1}-{end_idx})...")

        current_batch = pages[i:end_idx]
        batch_success = False
        retry_count = 0

        # Retry loop in case of timeout or error
        while not batch_success and retry_count < max_retries:
            try:
                if retry_count > 0:
                    print(f"  Retry attempt {retry_count} for batch {batch_num}")
                    await asyncio.sleep(5)  # wait before retry

                # Prepare documents (sanitizing curly braces)
                documents = [page.page_content.replace("{", '[').replace("}", "]") for page in current_batch]

                # Run distillation with a timeout
                distilled_text = await asyncio.wait_for(
                    document_distiller.distill(
                        documents=documents,
                        IE_query='''# DIRECTIVES : 
                                    - Act like an experienced information extractor.
                                    - You have a YouTube transcript about Scrum/Agile software development.
                                    - Extract key concepts, methodologies, best practices, and insights discussed.
                                    - If you do not find the right information, keep its place empty.
                                    - Focus on practical advice and real-world applications.''',
                        output_data_structure=Content
                    ),
                    timeout=timeout_seconds
                )

                # Save result
                distilled_texts.append(distilled_text)
                batch_success = True
                output_dir = "distilled_results"
                os.makedirs(output_dir, exist_ok=True)
                json_data = distilled_text.model_dump_json()
                batch_filename = os.path.join(output_dir, f"batch_{batch_num}.json")

                with open(batch_filename, "w", encoding="utf-8") as f:
                    f.write(json_data)

                print(f"Batch {batch_num} completed successfully")

            except asyncio.TimeoutError:
                # Handle timeout case
                print(f"Batch {batch_num} timed out after {timeout_seconds} seconds (attempt {retry_count + 1})")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Batch {batch_num} failed after {max_retries} attempts - skipping")
                    distilled_texts.append(None)

            except Exception as e:
                # Handle general errors
                print(f"Error in batch {batch_num} (attempt {retry_count + 1}): {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Batch {batch_num} failed after {max_retries} attempts - skipping")
                    distilled_texts.append(None)

        # Track elapsed time
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

        # Free memory between batches
        gc.collect()

        # Pause to avoid overloading system/LLM
        await asyncio.sleep(5)

    # Final summary
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed = time.time() - start_time
    print(f"\nProcess finished at: {end_timestamp}")
    print(f"Time elapsed: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print(f"Total batches processed: {len(distilled_texts)}")

    return distilled_texts


# --- Debug function for problematic batches ---
async def process_single_batch_debug(pages, batch_num, start_idx, end_idx):
    """
    Debug a single batch by printing content info and trying distillation once.
    Useful when one batch keeps failing.
    """
    print(f"Debug processing batch {batch_num} (pages {start_idx + 1}-{end_idx})...")

    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    document_distiller = DocumentsDistiller(llm_model=llm)

    current_batch = pages[start_idx:end_idx]

    try:
        # Prepare documents and show preview
        documents = [page.page_content.replace("{", '[').replace("}", "]") for page in current_batch]
        print(f"Batch contains {len(documents)} documents")
        for i, doc in enumerate(documents):
            print(f"  Document {i + 1}: {len(doc)} characters")
            print(f"  Preview: {doc[:100]}...")

        # Run distillation with a timeout
        start_time = time.time()
        distilled_text = await asyncio.wait_for(
            document_distiller.distill(
                documents=documents,
                IE_query='''# DIRECTIVES : 
                            - Act like an experienced information extractor.
                            - You have a YouTube transcript about Waterfall software development.
                            - Extract key concepts, methodologies, best practices, and insights discussed.
                            - If you do not find the right information, keep its place empty.
                            - Focus on practical advice and real-world applications.''',
                output_data_structure=Content
            ),
            timeout=300
        )

        processing_time = time.time() - start_time
        print(f"Batch processed successfully in {processing_time:.2f} seconds")
        return distilled_text

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return None


# --- Script entry point ---
if __name__ == "__main__":
    # Load documents from directory (text files)
    loader = DirectoryLoader(
        "../data/test/",            # Directory with transcripts
        glob="*.txt",               # Match only text files
        loader_cls=TextLoader,      # Use text loader
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    # Split documents into smaller chunks for LLM processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    pages = text_splitter.split_documents(documents)

    batch_size = 3  # Smaller batch size = more stable

    # Run the main batch processing
    distilled_results = asyncio.run(process_batches(pages, batch_size, timeout_seconds=120))

    # Debug a specific batch (optional)
    # debug_result = asyncio.run(process_single_batch_debug(pages, 4, 15, 20))

    # Print summary of successes/failures
    successful_batches = 0
    failed_batches = 0
    for i, result in enumerate(distilled_results):
        if result:
            print(f"Batch {i + 1}: Success")
            successful_batches += 1
        else:
            print(f"Batch {i + 1}: Failed")
            failed_batches += 1

    print(f"\nSummary: {successful_batches} successful, {failed_batches} failed batches")
