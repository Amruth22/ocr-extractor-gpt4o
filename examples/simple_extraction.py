import os
import sys

# Add the parent directory to the path so we can import the ocr_extractor module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_extractor import OCRExtractor

def main():
    """
    Simple example of using the OCRExtractor to extract text from an image.
    """
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python simple_extraction.py <path_to_image>")
        return
    
    image_path = sys.argv[1]
    
    # Create an instance of the OCRExtractor
    try:
        extractor = OCRExtractor()
        
        # Extract text from the image
        print(f"Extracting text from {image_path}...")
        extracted_text = extractor.extract_text_from_image(image_path)
        
        print("\nExtracted Text:")
        print("-" * 50)
        print(extracted_text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()