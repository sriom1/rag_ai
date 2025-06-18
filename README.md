# AI Knowledge Assistant

An intelligent question-answering system that combines vector store search and Wikipedia lookups using LangGraph pipeline and Streamlit interface.

## Features

- Smart routing between vector store and Wikipedia based on question content
- Vector store search for AI-specific topics (agents, prompt engineering, adversarial attacks)
- Wikipedia search for general knowledge questions
- Interactive web interface built with Streamlit
- Uses Groq LLM and HuggingFace embeddings

## Requirements

- Python 3.8+
- Groq API key
- HuggingFace API key
- Cassandra/Astra DB setup

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_ai
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
   - Create a `.streamlit/secrets.toml` file
   - Add your API keys:
```toml
GROQ_API_KEY = "your-groq-api-key"
HF_API_KEY = "your-huggingface-api-key"
```

## Usage

Run the Streamlit app:
```bash
streamlit run pipeline.py
```

The web interface will be available at `http://localhost:8501`

## Project Workflow

1. **Question Input**: User enters a question through the Streamlit interface

2. **Question Routing**:
   - System analyzes the question using Groq LLM
   - Routes to vector store for AI-specific topics
   - Routes to Wikipedia for general knowledge

3. **Document Retrieval**:
   - Vector store: Retrieves relevant chunks from AI-related documents
   - Wikipedia: Fetches relevant article content

4. **Answer Display**:
   - Shows retrieved content in the Streamlit interface
   - Formats results with proper spacing and markdown

## Technical Details

- **Vector Store**: Uses Cassandra/Astra DB with HuggingFace embeddings
- **LLM**: Groq's meta-llama/llama-4-scout-17b-16e-instruct model
- **Embeddings**: all-MiniLM-L6-v2 from HuggingFace
- **Document Sources**: 
  - AI Agents (Lilian Weng's blog)
  - Prompt Engineering
  - Adversarial Attacks on LLMs

## Environment Variables

Required environment variables in `.streamlit/secrets.toml`:
- `GROQ_API_KEY`: Your Groq API key
- `HF_API_KEY`: Your HuggingFace API key

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.