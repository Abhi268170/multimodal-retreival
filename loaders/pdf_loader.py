import fitz  # PyMuPDF
import os

def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list:
    """
    Converts each page of a PDF into an image and saves them.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Folder where images will be saved.

    Returns:
        list: Paths to the generated image files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    pdf = fitz.open(pdf_path)  # Open the PDF
    image_paths = []

    for i in range(len(pdf)):
        page = pdf[i]                       # Get the i-th page
        pix = page.get_pixmap()             # Render the page to an image
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(img_path)                  # Save the image as a PNG
        image_paths.append(img_path)        # Store path to return later

    pdf.close()  # Cleanup
    return image_paths
