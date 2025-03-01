import os
import sys
import json

# Add the parent directory to the path so we can import the ocr_extractor module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_extractor import OCRExtractor

def main():
    """
    Example of extracting tabular data from an image.
    """
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python extract_table.py <path_to_image>")
        return
    
    image_path = sys.argv[1]
    
    # Create an instance of the OCRExtractor
    try:
        extractor = OCRExtractor()
        
        # Extract table data from the image
        print(f"Extracting table data from {image_path}...")
        
        # Custom prompt for table extraction
        prompt = """
        Extract the table data from this image. 
        Format the output as a JSON array of objects, where each object represents a row in the table.
        Each object should have keys corresponding to the column headers.
        """
        
        extracted_data = extractor.extract_text_from_image(image_path, prompt)
        
        print("\nExtracted Table Data:")
        print("-" * 50)
        print(extracted_data)
        print("-" * 50)
        
        # Note: In a real application, you might want to parse the JSON response
        # and format it nicely or save it to a CSV file
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()