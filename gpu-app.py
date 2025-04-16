import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import fitz  # PyMuPDF
import os
import time
import io
import pathlib # For easier path handling

# --- Configuration ---
PDF_PATH = r"D:\RAG\Data-to-feed\accounting_books\leac101.pdf" # <--- !!! SET THE PATH TO YOUR PDF FILE HERE !!!
OUTPUT_DIR = "output"
MODEL_NAME = "ds4sd/SmolDocling-256M-preview"
# Set DPI for image conversion. Higher DPI means better quality but slower processing and more memory.
IMAGE_DPI = 150 # You might need to adjust this (e.g., 150, 200, 300)
# --- End Configuration ---

def setup_device():
    """Sets up the computation device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU.")
        # Optional: Print CUDA device details
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        # device = "cpu"
        device = "eager"
        print("CUDA not available. Using CPU.")
    return device

def load_model_and_processor(model_name, device):
    """Loads the model and processor."""
    print(f"Loading processor: {model_name}...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        print(f"Loading model: {model_name}...")
        # Determine attention implementation based on device
        attn_implementation = "eager"
        print(f"Using attention implementation: {attn_implementation}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # bfloat16 only on CUDA usually
             _attn_implementation=attn_implementation,
        ).to(device)
        print("Model and processor loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading model/processor: {e}")
        print("Please ensure the model name is correct and you have an internet connection.")
        # Instead of exiting, raise the exception
        raise e # This will propagate the error and stop execution

def convert_page_to_image(page, dpi=150):
    """Converts a PyMuPDF page object to a PIL Image."""
    print(f"Converting page to image (DPI: {dpi})...")
    start_conv_time = time.time()
    try:
        # Render page to a pixmap (image representation)
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")  # Convert pixmap to PNG bytes

        # Create PIL image from bytes
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        conv_time = time.time() - start_conv_time
        print(f"Page converted to image in {conv_time:.2f} seconds.")
        return image
    except Exception as e:
        print(f"Error converting page to image: {e}")
        return None

def process_image_to_markdown(image, processor, model, device):
    """Processes a single image using the model and returns markdown."""
    print("Processing image with model...")
    start_process_time = time.time()
    try:
        # Prepare model inputs
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        # Use context manager for potential memory benefits on GPU
        with torch.no_grad(): # Disable gradient calculations for inference
             inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

             # Generate output IDs
             print("Generating token IDs...")
             generated_ids = model.generate(**inputs, max_new_tokens=8192) # Adjust max_new_tokens if needed

        # Decode and process output
        print("Decoding generated IDs...")
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        # Ensure skip_special_tokens is appropriate - sometimes False is needed by docling
        doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
        # print(f"Raw Doctags (first 100 chars): {doctags[:100]}...") # Optional debug print

        print("Creating Docling documents...")
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="PageContent") # Use a generic name
        doc.load_from_doctags(doctags_doc)
        markdown_content = doc.export_to_markdown()

        process_time = time.time() - start_process_time
        print(f"Image processed and markdown generated in {process_time:.2f} seconds.")
        return markdown_content

    except Exception as e:
        process_time = time.time() - start_process_time
        print(f"Error during model processing (took {process_time:.2f}s): {e}")
        # Optionally include traceback for debugging
        # import traceback
        # traceback.print_exc()
        return f"\n\n--- ERROR PROCESSING PAGE: {e} ---\n\n"


def main():
    """Main function to orchestrate the PDF to Markdown conversion."""
    print("--- Starting PDF to Markdown Conversion ---")
    start_time_total = time.time()

    # --- Input Validation ---
    pdf_path_obj = pathlib.Path(PDF_PATH)
    if not pdf_path_obj.is_file() or pdf_path_obj.suffix.lower() != ".pdf":
        print(f"Error: Invalid PDF path '{PDF_PATH}'. Please check the file exists and is a .pdf file.")
        return

    print(f"Input PDF: {pdf_path_obj.name}")

    # --- Output Setup ---
    output_dir_path = pathlib.Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True) # Create output dir if needed
    output_md_filename = pdf_path_obj.stem + ".md" # e.g., mydocument.md
    output_md_path = output_dir_path / output_md_filename
    print(f"Output Markdown will be saved to: {output_md_path}")

    # --- Setup Device and Model ---
    device = setup_device()
    processor, model = load_model_and_processor(MODEL_NAME, device)

    # --- PDF Processing ---
    all_markdown_content = []
    total_pages = 0
    processed_pages = 0

    try:
        print(f"Opening PDF file: {pdf_path_obj}...")
        pdf_document = fitz.open(pdf_path_obj)
        total_pages = len(pdf_document)
        print(f"PDF has {total_pages} pages.")

        for page_num in range(total_pages):
            page_start_time = time.time()
            print(f"\n--- Processing Page {page_num + 1} of {total_pages} ---")

            page = pdf_document.load_page(page_num)

            # 1. Convert page to image
            page_image = convert_page_to_image(page, dpi=IMAGE_DPI)

            if page_image:
                # 2. Process image with model
                markdown_output = process_image_to_markdown(page_image, processor, model, device)
                all_markdown_content.append(markdown_output)
                processed_pages += 1
            else:
                print(f"Skipping page {page_num + 1} due to image conversion error.")
                all_markdown_content.append(f"\n\n--- SKIPPED PAGE {page_num + 1} (Image Conversion Error) ---\n\n")

            page_end_time = time.time()
            page_duration = page_end_time - page_start_time
            print(f"--- Page {page_num + 1} finished in {page_duration:.2f} seconds ---")

        print("\nClosing PDF file.")
        pdf_document.close()

    except Exception as e:
        print(f"\nAn error occurred during PDF processing: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed error trace

    # --- Final Output ---
    print(f"\nProcessed {processed_pages} out of {total_pages} pages.")
    print(f"Writing combined markdown content to {output_md_path}...")

    # Add page separators for clarity in the final markdown
    final_markdown = "\n\n---\n\n".join(all_markdown_content)

    try:
        with open(output_md_path, "w", encoding='utf-8') as f:
            f.write(final_markdown)
        print("Markdown file saved successfully.")
    except Exception as e:
        print(f"Error writing final markdown file: {e}")

    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"\n--- Conversion Complete ---")
    print(f"Total time taken: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Average time per processed page: {total_duration / processed_pages:.2f} seconds" if processed_pages > 0 else "N/A")
    print(f"Output saved in: {output_md_path.parent}")
    print("---------------------------")

    

if __name__ == "__main__":
    main()