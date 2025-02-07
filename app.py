# Standard library imports
import asyncio
import http.client
import json
import logging
import os
import re
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
import uuid

# Third-party imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
import redis.asyncio as redis
import soundfile as sf

# Local imports
from kokoro_onnx import Kokoro


# Ensure audio directory exists
os.makedirs("static/audio", exist_ok=True)

logger = logging.getLogger(__name__)


model_name = "llama-3.3-70b-versatile"
serp_api_key = "<YOUR SERP API KEY>"
groq_api_key="<YOUR GROQ API KEY>"

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_update(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending update to client {client_id}: {e}")
                self.disconnect(client_id)


class RedisManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
    
    async def connect(self):
        self.redis = redis.from_url(self.redis_url)
        await self.redis.ping()
        logger.info("Connected to Redis")
    
    async def disconnect(self):
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")
    
    async def set_task_status(self, task_id: str, status: dict, expire: int = 3600):
        await self.redis.set(
            f"task:{task_id}",
            json.dumps(status),
            ex=expire
        )
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        status = await self.redis.get(f"task:{task_id}")
        return json.loads(status) if status else None


class ResearchAgent:
    def __init__(self, serper_api_key: str, openai_api_key: str, base_url: str = "https://api.groq.com/openai/v1", thought_callback: Optional[Callable[[str], None]] = None):
        self.serper_api_key = serper_api_key
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.research_history = []
        self.research_data = []
        self.sources = []
        self.thought_callback = thought_callback
        self._loop = asyncio.get_event_loop()

    def _send_thought(self, thought: str):
        """Synchronous wrapper for thought callback"""
        if self.thought_callback:
            future = asyncio.run_coroutine_threadsafe(
                self.thought_callback(thought),
                self._loop
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error sending thought: {e}")
    
    def add_source(self, url: str, title: str, snippet: str = None) -> int:
        """Add a source and return its reference number"""
        source_id = len(self.sources) + 1
        self.sources.append({
            'id': source_id,
            'url': url,
            'title': title,
            'snippet': snippet,
            'access_date': datetime.now().strftime("%Y-%m-%d")
        })
        return source_id

    def format_bibliography(self) -> str:
        """Generate formatted bibliography from sources"""
        if not self.sources:
            return "No sources cited."
            
        bibliography = "\n\nSources Cited:\n"
        for source in self.sources:
            bibliography += f"\n[{source['id']}] {source['title']}. Retrieved from {source['url']} on {source['access_date']}"
        return bibliography

    def ask_llm(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            raise

    def google_search(self, query: str) -> dict:
        """Perform Google search using Serper API"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/search", payload, headers)
            response = conn.getresponse()
            results = json.loads(response.read().decode("utf-8"))
            
            if 'organic' in results:
                for result in results['organic']:
                    self.add_source(
                        url=result['link'],
                        title=result['title'],
                        snippet=result.get('snippet', '')
                    )
            
            return results
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            raise
        finally:
            conn.close()

    def parse_llm_action(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM's response to extract the action JSON, handling markdown code blocks
        """
        try:
            # First try to find JSON within code blocks
            if "```json" in response:
                # Extract content between ```json and ```
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                    return json.loads(json_str)
            
            # If no code block or invalid JSON in code block, 
            # try to find JSON anywhere in the text
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.finditer(json_pattern, response)
            
            for match in matches:
                try:
                    json_str = match.group()
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                    
            # If still no valid JSON found, return error
            logger.error(f"No valid JSON found in response: {response}")
            return {
                "action": "error",
                "thought": "Error parsing LLM response",
                "error": "No valid JSON found in response"
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}\nResponse: {response}")
            return {
                "action": "error",
                "thought": "Error parsing LLM response",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error parsing LLM action: {e}\nResponse: {response}")
            return {
                "action": "error",
                "thought": "Error parsing LLM response",
                "error": str(e)
            }
        
    def scrape_webpage(self, url: str) -> str:
        """Scrape webpage using Serper's scraping API"""
        conn = http.client.HTTPSConnection("scrape.serper.dev")
        payload = json.dumps({"url": url})
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        conn.close()
        
        # Add source if successfully scraped
        if data.get('text'):
            self.add_source(
                url=url,
                title=data.get('title', url),
                snippet=data.get('text', '')[:200] + '...'
            )
            
        return data.get('text', '')

    def research_topic(self, topic: str, max_steps: int = 10) -> str:
        """Execute the research workflow"""
        system_prompt = """You are a research agent that decides the next best action to take in researching a topic. In the end you need to write a comphrensive research report on the topic.

        Available actions:
        1. search - Perform a Google search
        2. scrape - Scrape a webpage
        3. analyze - Analyze collected information
        4. report - Generate final report
        
        Before each action, explain your thought process clearly.
        
        When writing the final report, cite sources using [n] notation, where n is the source number.
        Each fact should be cited, and a bibliography will be automatically added.
        
        Respond with a JSON object in this exact format:
        {
            "action": "search|scrape|analyze|report",
            "thought": "Your reasoning for this action",
            "query": "Search query or URL to scrape",
            "report": "Final report content if action is report"
        }
        """+f"""The number of maximm actions you are allowed to take is {str(max_steps)}."""

        try:
            current_context = f"Research topic: {topic}\nCurrent findings: {json.dumps(self.research_history)}\nAvailable sources: {json.dumps(self.sources)}"
            
            for step in range(max_steps):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_context}
                ]

                llm_response = self.ask_llm(messages)
                action_data = self.parse_llm_action(llm_response)
                
                # Check if parsing returned an error
                if action_data.get('action') == 'error':
                    logger.error(f"Error in step {step}: {action_data.get('error')}")
                    continue
                
                # Send thought if available
                if action_data.get('thought'):
                    self._send_thought(action_data['thought'])
                    import time
                    time.sleep(2)
                
                if action_data.get('action') == 'search':
                    search_results = self.google_search(action_data['query'])
                    self.research_history.append({
                        'type': 'search',
                        'query': action_data['query'],
                        'results': search_results
                    })
                    current_context += f"\nSearch results: {json.dumps(search_results)}"
                    
                elif action_data.get('action') == 'scrape':
                    content = self.scrape_webpage(action_data['query'])
                    self.research_history.append({
                        'type': 'scrape',
                        'url': action_data['query'],
                        'content': content[:500] + '...' if content else ''
                    })
                    current_context += f"\nScraped content from {action_data['query']}"
                    
                elif action_data.get('action') == 'analyze':
                    analysis_messages = [
                        {"role": "system", "content": "Analyze the research findings so far and provide insights. Use [n] citations when referring to sources."},
                        {"role": "user", "content": current_context}
                    ]
                    analysis = self.ask_llm(analysis_messages)
                    self.research_history.append({
                        'type': 'analysis',
                        'content': analysis
                    })
                    current_context += f"\nAnalysis: {analysis}"
                    
                    
                time.sleep(1)  # Rate limiting
            
            # If max steps reached, generate a report from collected information
            logger.info("Max steps reached, generating report from collected information")
            self._send_thought("Generating final report from collected information.")
            
            report_prompt = f"""Based on our research on {topic}, please generate a comprehensive research report 
            using all available information. Use [n] citations to reference sources where n is the source number.
            
            Research history: {json.dumps(self.research_history, indent=2)}
            Available sources: {json.dumps(self.sources, indent=2)}
            
            Format the report with clear sections, insights, and proper citations. You have to use markdown as a format for the report. Write clear, well-organized reports with proper citations. Render links to the sources in the format: [text](url)"""

            messages = [
                {"role": "system", "content": "You are a research report writer."},
                {"role": "user", "content": report_prompt}
            ]
            
            final_report = self.ask_llm(messages)
            return final_report
            
        except Exception as e:
            logger.error(f"Error in research process: {e}")
            return f"Error during research: {str(e)}"


class PodcastGenerator:
    def __init__(self, ws_manager: WebSocketManager, redis_manager: RedisManager):
        self.ws_manager = ws_manager
        self.redis_manager = redis_manager
        self.current_client_id = None
        self.agent = None
        # Initialize Kokoro once during class initialization
        self.kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    
    async def send_step_update(self, client_id: str, step: str, data: dict):
        """
        Send step updates to the client through WebSocket and store in Redis
        """
        update = {
            "step": step,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.ws_manager.send_update(client_id, update)
        await self.redis_manager.set_task_status(client_id, update)
    
    async def generate_research(self, client_id: str, topic: str):
        """
        Generate research content using ResearchAgent with real-time updates
        """
        try:
            self.current_client_id = client_id

            # Define async callback for thoughts
            async def thought_callback(thought: str):
                await self.send_step_update(
                    self.current_client_id, 
                    GenerationStep.THINKING, 
                    {
                        "message": thought,
                        "progress": 50,
                        "substep": "research_thought",
                        "status": TaskStatus.PROCESSING
                    }
                )

            # Initialize research agent with callback
            self.agent = ResearchAgent(
                serper_api_key=serp_api_key,
                openai_api_key=groq_api_key,
                thought_callback=thought_callback
            )
            
            # Run research in background to not block the event loop
            report = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.agent.research_topic, 
                topic
            )
            
            # Send research completion update
            data = {
                "findings": [
                    {
                        "title": "Research Report",
                        "content": report,
                        "sources": self.agent.sources
                    }
                ],
                "progress": 75,
                "status": TaskStatus.COMPLETED
            }
            await self.send_step_update(client_id, GenerationStep.RESEARCH, data)

        except Exception as e:
            logger.error(f"Error in research generation for client {client_id}: {e}")
            error_data = {
                "error": str(e),
                "status": TaskStatus.FAILED,
                "progress": 0
            }
            await self.send_step_update(client_id, GenerationStep.RESEARCH, error_data)
            raise
    
    async def generate_podcast(self, client_id: str, topic: str):
        """
        Main podcast generation workflow
        """
        try:
            logger.info(f"Starting podcast generation for client {client_id} on topic: {topic}")
            
            generation_steps = [
                self.generate_research,
                self.finalize_podcast
            ]
            
            for step in generation_steps:
                await step(client_id, topic)
            
            logger.info(f"Completed podcast generation for client {client_id}")
                
        except Exception as e:
            logger.error(f"Error generating podcast for client {client_id}: {e}")
            error_data = {
                "error": str(e),
                "status": TaskStatus.FAILED
            }
            await self.send_step_update(client_id, "error", error_data)
            raise
        finally:
            self.current_client_id = None
    
    async def get_generation_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of podcast generation for a client
        """
        try:
            return await self.redis_manager.get_task_status(client_id)
        except Exception as e:
            logger.error(f"Error getting generation status for client {client_id}: {e}")
            return None
        
    async def convert_to_dialogue(self, research_content: str, topic: str) -> str:
        """
        Convert research content into a natural dialogue between two hosts
        """
        logger.info("Converting research to dialogue format")
        
        prompt = f"""Convert this research content into a natural, engaging dialogue between two podcast hosts: Alex and Sarah.
        Make it sound like a real conversation, with:
        - Natural back-and-forth discussion
        - Questions and responses
        - Casual interjections and reactions
        - Each line should start with the speaker's name (Alex: or Sarah:)
        - Break complex topics into digestible parts
        - Include some light humor and personality
        - Keep the tone informative but conversational
        - Please format words in a way thats spoken. For example, A.I becomes AI or U.S.A becomes USA and so on.
        
        Topic: {topic}
        Research Content: {research_content}
        
        Format the dialogue as:
        Alex: [Alex's line]
        Sarah: [Sarah's line]
        etc.
        """
        
        try:
            dialogue = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.agent.ask_llm([
                {"role": "system", "content": "You are a podcast script writer. Create engaging, natural dialogue."},
                {"role": "user", "content": prompt}
            ])
        )
            
            logger.info("Successfully generated dialogue script")
            return dialogue
            
        except Exception as e:
            logger.error(f"Error generating dialogue: {e}")
            raise

    async def finalize_podcast(self, client_id: str, topic: str):
        """
        Finalize the podcast generation process with audio synthesis
        """
        try:
            logger.info(f"Starting podcast generation for client {client_id}")
            
            # Get the research content from Redis
            status = await self.redis_manager.get_task_status(client_id)
            if not status:
                logger.error(f"No research content found in Redis for client {client_id}")
                raise ValueError("No research content found in Redis")
            
            # Find the research step data
            research_step = None
            for update in [status] if isinstance(status, dict) else status:
                if isinstance(update, dict) and update.get('step') == 'research':
                    research_step = update
                    break
            
            if not research_step or 'data' not in research_step:
                logger.error(f"Research step not found in status for client {client_id}")
                raise ValueError("Research step not found in status")
            
            research_data = research_step['data']
            if not research_data or "findings" not in research_data:
                logger.error(f"Research findings not found for client {client_id}")
                raise ValueError("Research findings not found")
            
            research_content = research_data["findings"][0]["content"]
            if not research_content:
                logger.error(f"Empty research content for client {client_id}")
                raise ValueError("Empty research content")

            # Convert research to dialogue
            logger.info(f"Converting research to dialogue for client {client_id}")
            dialogue = await self.convert_to_dialogue(research_content, topic)
            
            # Split dialogue by speaker
            logger.info("Splitting dialogue by speaker")
            lines = dialogue.strip().split('\n')
            alex_lines = []
            sarah_lines = []
            
            for line in lines:
                if line.startswith('Alex:'):
                    alex_lines.append(line.replace('Alex:', '').strip())
                elif line.startswith('Sarah:'):
                    sarah_lines.append(line.replace('Sarah:', '').strip())
            
            logger.info(f"Found {len(alex_lines)} lines for Alex and {len(sarah_lines)} lines for Sarah")

            # Initialize an empty list to store all audio segments
            all_segments = []
            pause_duration = 0.5  # 0.5 second pause
            sample_rate = None  # Will be set from first audio generation
            
            logger.info("Starting audio generation for dialogue")
            total_lines = len(list(zip_longest(alex_lines, sarah_lines)))
            current_line = 0

            for alex_line, sarah_line in zip_longest(alex_lines, sarah_lines):
                current_line += 1
                progress = int(75 + (current_line / total_lines * 25))  # Progress from 75% to 100%
                
                if alex_line:
                    logger.info(f"Generating audio for Alex line {current_line}/{total_lines}")
                    try:
                        alex_audio = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda l=alex_line: self.kokoro.create(l, voice="am_adam", speed=1.0, lang="en-us")
                        )
                        all_segments.append(alex_audio[0])
                        if sample_rate is None:
                            sample_rate = alex_audio[1]
                        
                        # Add pause after Alex's line
                        pause_samples = int(pause_duration * sample_rate)
                        all_segments.append(np.zeros(pause_samples))
                    except Exception as e:
                        logger.error(f"Error generating audio for Alex line: {e}")
                        raise
                
                if sarah_line:
                    logger.info(f"Generating audio for Sarah line {current_line}/{total_lines}")
                    try:
                        sarah_audio = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda l=sarah_line: self.kokoro.create(l, voice="af_sarah", speed=1.0, lang="en-us")
                        )
                        all_segments.append(sarah_audio[0])
                        if sample_rate is None:
                            sample_rate = sarah_audio[1]
                        
                        # Add pause after Sarah's line
                        pause_samples = int(pause_duration * sample_rate)
                        all_segments.append(np.zeros(pause_samples))
                    except Exception as e:
                        logger.error(f"Error generating audio for Sarah line: {e}")
                        raise
                
                # Update progress
                await self.send_step_update(
                    client_id,
                    GenerationStep.FINAL,
                    {
                        "status": TaskStatus.PROCESSING,
                        "progress": progress,
                        "message": f"Generating audio: {current_line}/{total_lines} lines completed"
                    }
                )

            if not all_segments:
                logger.error("No audio segments generated")
                raise ValueError("No audio segments were generated")

            # Concatenate all segments
            logger.info("Concatenating audio segments")
            try:
                final_audio = np.concatenate(all_segments)
            except Exception as e:
                logger.error(f"Error concatenating audio segments: {e}")
                raise ValueError(f"Error merging audio segments: {str(e)}")

            # Ensure audio directory exists
            os.makedirs("static/audio", exist_ok=True)
            
            # Save final audio
            audio_path = f"static/audio/{client_id}.wav"
            logger.info(f"Saving audio to {audio_path}")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: sf.write(audio_path, final_audio, sample_rate)
                )
            except Exception as e:
                logger.error(f"Error saving audio file: {e}")
                raise ValueError(f"Error saving audio file: {str(e)}")
            
            # Calculate duration
            duration_seconds = len(final_audio) / sample_rate
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_str = f"{minutes}:{seconds:02d}"
            
            logger.info(f"Audio generation complete. Duration: {duration_str}")
            
            # Save dialogue script
            script_path = f"static/audio/{client_id}_script.txt"
            logger.info(f"Saving script to {script_path}")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: open(script_path, 'w').write(dialogue)
                )
            except Exception as e:
                logger.error(f"Error saving script file: {e}")
                raise ValueError(f"Error saving script file: {str(e)}")
            
            # Final update with completion status
            data = {
                "title": f"{topic} - A Conversation with Alex and Sarah",
                "duration": duration_str,
                "status": TaskStatus.COMPLETED,
                "progress": 100,
                "audio_url": f"/api/audio/{client_id}",
                "script_url": f"/api/script/{client_id}",
                "message": "Podcast generation completed successfully"
            }
            await self.send_step_update(client_id, GenerationStep.FINAL, data)

        except Exception as e:
            logger.error(f"Error in podcast finalization for client {client_id}: {e}")
            error_data = {
                "error": str(e),
                "status": TaskStatus.FAILED,
                "progress": 75,
                "message": f"Error generating podcast: {str(e)}"
            }
            await self.send_step_update(client_id, GenerationStep.FINAL, error_data)
            raise

    def format_content_for_speech(self, content: str) -> str:
        """
        Format the research content to be more suitable for text-to-speech
        """
        # Remove markdown formatting
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Remove links but keep text
        content = re.sub(r'[*_~`]', '', content)  # Remove markdown formatting characters
        content = re.sub(r'#+\s+', '', content)  # Remove header markers
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple blank lines
        content = re.sub(r'\s+', ' ', content)  # Normalize spaces
        
        # Add pauses for better speech flow
        content = content.replace('.', '... ')
        content = content.replace('!', '... ')
        content = content.replace('?', '... ')
        
        # Limit content length if needed (you can adjust this)
        max_chars = 3000
        if len(content) > max_chars:
            content = content[:max_chars] + "... That concludes our summary."
        
        return content.strip()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerationStep(str, Enum):
    THINKING = "thinking"
    RESEARCH = "research"
    FINAL = "final"

class PodcastRequest(BaseModel):
    topic: str
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# Initialize FastAPI with dependencies
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.ws_manager = WebSocketManager()
    app.state.redis_manager = RedisManager("redis://localhost") #Port 6379 is default redis port
    await app.state.redis_manager.connect()
    app.state.podcast_generator = PodcastGenerator(
        app.state.ws_manager,
        app.state.redis_manager
    )
    yield
    # Shutdown
    await app.state.redis_manager.disconnect()

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/audio/{client_id}")
async def get_audio(client_id: str):
    audio_path = f"static/audio/{client_id}.wav"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await app.state.ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "generate":
                # Send initial status
                await websocket.send_json({"status": "generation_started"})
                
                # Run generation directly in websocket task
                await app.state.podcast_generator.generate_podcast(
                    client_id,
                    data.get("topic")
                )
    except WebSocketDisconnect:
        app.state.ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        app.state.ws_manager.disconnect(client_id)

@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str):
    status = await app.state.redis_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)