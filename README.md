# OCR Extractor using GPT-4o Vision

This repository contains a tool for extracting text from images using OpenAI's GPT-4o Vision Language Model. The tool can process various types of images and documents to extract text content with high accuracy.

## Features

- Extract text from images using GPT-4o Vision capabilities
- Support for various image formats (PNG, JPG, JPEG, etc.)
- Handling of complex document layouts and formatting
- Option to extract structured data from forms and tables
- Command-line interface for batch processing

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4o
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Amruth22/ocr-extractor-gpt4o.git
   cd ocr-extractor-gpt4o
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

Basic usage:
```python
from ocr_extractor import OCRExtractor

extractor = OCRExtractor()
text = extractor.extract_text_from_image("path/to/your/image.jpg")
print(text)
```

For more examples and advanced usage, see the examples directory.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.