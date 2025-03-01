#!/usr/bin/env python3
import argparse
import os
import sys
import json
from typing import List, Optional
import logging
from tqdm import tqdm

from ocr_extractor import OCRExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_image(extractor: OCRExtractor, image_path: str, prompt: Optional[str] = None, 
                         output_file: Optional[str] = None, structured: bool = False) -> None:
    """
    Process a single image and optionally save the result to a file.
    
    Args:
        extractor: OCRExtractor instance
        image_path: Path to the image file
        prompt: Custom prompt to use
        output_file: Path to save the output
        structured: Whether to extract structured data
    """
    try:
        if structured:
            result = extractor.extract_structured_data(image_path)
            extracted_text = json.dumps(result, indent=2)
        else:
            extracted_text = extractor.extract_text_from_image(image_path, prompt or "Extract all text from this image.")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"Results saved to {output_file}")
        else:
            print("\nExtracted Text:")
            print("-" * 50)
            print(extracted_text)
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        print(f"Error: {e}")

def process_batch(extractor: OCRExtractor, image_paths: List[str], prompt: Optional[str] = None,
                 output_dir: Optional[str] = None, structured: bool = False) -> None:
    """
    Process multiple images in batch.
    
    Args:
        extractor: OCRExtractor instance
        image_paths: List of paths to image files
        prompt: Custom prompt to use for all images
        output_dir: Directory to save the outputs
        structured: Whether to extract structured data
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            if structured:
                result = extractor.extract_structured_data(image_path)
                extracted_text = json.dumps(result, indent=2)
            else:
                extracted_text = extractor.extract_text_from_image(image_path, prompt or "Extract all text from this image.")
            
            if output_dir:
                base_name = os.path.basename(image_path)
                file_name = os.path.splitext(base_name)[0] + ".txt"
                output_file = os.path.join(output_dir, file_name)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                logger.info(f"Saved results for {image_path} to {output_file}")
            else:
                print(f"\nResults for {image_path}:")
                print("-" * 50)
                print(extracted_text)
                print("-" * 50)
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract text from images using GPT-4o Vision")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single image file")
    input_group.add_argument("--dir", help="Path to a directory containing images")
    input_group.add_argument("--list", help="Path to a text file with image paths (one per line)")
    
    # Processing options
    parser.add_argument("--prompt", help="Custom prompt for the model")
    parser.add_argument("--structured", action="store_true", help="Extract structured data (like tables)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    
    # Output options
    parser.add_argument("--output", help="Output file for single image or directory for batch processing")
    parser.add_argument("--format", choices=["txt", "json"], default="txt", help="Output format (default: txt)")
    
    # API options
    parser.add_argument("--api-key", help="OpenAI API key (if not set in environment variable)")
    
    args = parser.parse_args()
    
    # Initialize the extractor
    try:
        extractor = OCRExtractor(api_key=args.api_key, model=args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Process single image
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
            
        process_single_image(
            extractor=extractor,
            image_path=args.image,
            prompt=args.prompt,
            output_file=args.output,
            structured=args.structured
        )
    
    # Process directory of images
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        image_paths = []
        
        for file in os.listdir(args.dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.dir, file))
        
        if not image_paths:
            print(f"No image files found in directory: {args.dir}")
            return
            
        print(f"Found {len(image_paths)} images to process")
        process_batch(
            extractor=extractor,
            image_paths=image_paths,
            prompt=args.prompt,
            output_dir=args.output,
            structured=args.structured
        )
    
    # Process list of image paths
    elif args.list:
        if not os.path.isfile(args.list):
            print(f"Error: List file not found: {args.list}")
            return
            
        with open(args.list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # Validate paths
        valid_paths = []
        for path in image_paths:
            if os.path.isfile(path):
                valid_paths.append(path)
            else:
                print(f"Warning: File not found, skipping: {path}")
        
        if not valid_paths:
            print("No valid image paths found in the list")
            return
            
        print(f"Processing {len(valid_paths)} images from list")
        process_batch(
            extractor=extractor,
            image_paths=valid_paths,
            prompt=args.prompt,
            output_dir=args.output,
            structured=args.structured
        )

if __name__ == "__main__":
    main()