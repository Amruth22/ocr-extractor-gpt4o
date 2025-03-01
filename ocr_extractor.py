import os
import base64
from typing import Optional, Dict, Any, List, Union
import requests
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRExtractor:
    """
    A class for extracting text from images using OpenAI's GPT-4o Vision model.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the OCR extractor.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
            model: The OpenAI model to use. Default is "gpt-4o".
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as an argument or set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"OCRExtractor initialized with model: {model}")
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded image string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def _prepare_image_payload(self, image_data: str) -> Dict[str, Any]:
        """
        Prepare the image payload for the API request.
        
        Args:
            image_data: Base64 encoded image data.
            
        Returns:
            Dictionary containing the image payload.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": "high"
            }
        }
    
    def extract_text_from_image(self, image_path: str, prompt: str = "Extract all text from this image.") -> str:
        """
        Extract text from an image using GPT-4o Vision.
        
        Args:
            image_path: Path to the image file.
            prompt: The prompt to send to the model. Default is "Extract all text from this image."
            
        Returns:
            Extracted text from the image.
        """
        try:
            # Encode the image
            base64_image = self._encode_image(image_path)
            
            # Prepare the payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            self._prepare_image_payload(base64_image)
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            # Make the API request
            logger.info(f"Sending request to OpenAI API for image: {image_path}")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Extract and return the text
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            logger.info(f"Successfully extracted text from image: {image_path}")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            raise
    
    def extract_structured_data(self, image_path: str, data_format: str = "table") -> Dict[str, Any]:
        """
        Extract structured data from an image.
        
        Args:
            image_path: Path to the image file.
            data_format: The format of data to extract (table, form, etc.). Default is "table".
            
        Returns:
            Dictionary containing the structured data.
        """
        prompt = f"Extract the {data_format} data from this image and format it as JSON."
        
        try:
            text_result = self.extract_text_from_image(image_path, prompt)
            # Note: In a real implementation, you would parse the JSON here
            # For simplicity, we're just returning the text
            return {"result": text_result}
        except Exception as e:
            logger.error(f"Error extracting structured data from image: {e}")
            raise
    
    def batch_process(self, image_paths: List[str], prompt: Optional[str] = None) -> Dict[str, str]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to image files.
            prompt: Optional custom prompt to use for all images.
            
        Returns:
            Dictionary mapping image paths to extracted text.
        """
        results = {}
        for image_path in image_paths:
            try:
                if prompt:
                    results[image_path] = self.extract_text_from_image(image_path, prompt)
                else:
                    results[image_path] = self.extract_text_from_image(image_path)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results[image_path] = f"Error: {str(e)}"
        
        return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from images using GPT-4o Vision")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--prompt", default="Extract all text from this image.", help="Custom prompt for the model")
    
    args = parser.parse_args()
    
    try:
        extractor = OCRExtractor()
        result = extractor.extract_text_from_image(args.image_path, args.prompt)
        print(result)
    except Exception as e:
        print(f"Error: {e}")