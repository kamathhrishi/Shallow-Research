# Shallow Research Podcast Generator 😅

(Credits to Claude 3.5 Sonnet for this README haha) 

A open source AI Agent that conducts research and generates both research reports and engaging podcast conversations between two virtual hosts (Alex and Sarah). 

While tech giants are working on "deep" research capabilities, we're keeping it light and breezy with:
- Quick Google searches via SERP API
- Conversational AI using Groq LLM
- Text-to-speech magic with Kokoro
- Real-time updates to keep you entertained while we do our "research" 😉


## Requirements

Create a `requirements.txt` file with the following dependencies:

```
fastapi
uvicorn
openai
redis-py
soundfile
numpy
pydantic
python-multipart
httpx
kokoro-onnx
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Kokoro Models
Download the required Kokoro model files and place them in your project directory:
- [kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx)
- [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)

### 3. Configure API Keys
Replace the following API keys in `app.py`:
```python
serp_api_key = 'your_serp_api_key'
groq_api_key = 'your_groq_api_key'
```

### 4. Start Redis Server
Start a Redis instance on port 6379. If you have Docker installed, you can run:
```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

### 5. Run Backend Server
Start the FastAPI backend server:
```bash
python3 app.py
```
The server will run on `http://localhost:8001`

### 6. Launch Frontend
Open `index.html` in a web browser to access the frontend interface.

## Features

- Real-time LLM thought process visibility:
  * See the AI's research strategy development
  * Watch it evaluate information and decide next steps
  * Observe how it determines research completeness
  * Follow its thought process for converting research to dialogue
- Generates comprehensive research reports with:
  * Academic-style citations using [n] notation
  * Full bibliography with numbered sources
  * Access dates and URLs for each citation
  * Proper markdown formatting for readability
- Intelligent conversion of cited research into natural podcast dialogue
- High-quality voice synthesis using Kokoro
- Real-time progress updates via WebSocket (because waiting is boring)
- Redis-based task status tracking
- Downloadable podcast audio and scripts

## System Requirements

- Python 3.8+
- Redis server
- Modern web browser
- Internet connection for API access
- A sense of humor 😄

## Disclaimer

This project demonstrates automated research and content generation capabilities. The system first conducts research using multiple sources, generating a properly formatted academic-style report with numbered citations and a complete bibliography. This research is then intelligently converted into an engaging podcast conversation, maintaining the factual accuracy while making the content accessible and entertaining.
