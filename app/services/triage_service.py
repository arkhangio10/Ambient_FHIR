import json
import logging
from app.config import get_settings
from app.services import voice_service

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an enthusiastic, empathetic AI Triage Nurse Assistant.
Your goal is to gently ask the patient 3 specific questions, ONE AT A TIME, to gather:
1. Chief Complaint (What brings them in today?)
2. Pain or Severity Rating (1 to 10 scale)
3. Duration (How long have they had these symptoms?)

CRITICAL RULES:
- Start by asking ONE simple question.
- Do NOT ask multiple questions in a single message.
- Keep the tone very conversational, supportive, and extremely concise.
- Once you have reasonably gathered all three pieces of information, output a final brief summary thanking them, and set "is_complete" to true.

You MUST respond strictly in the following JSON format:
{
    "reply": "Your next question or final summary to the patient.",
    "is_complete": false
}
"""

async def generate_triage_reply(messages: list) -> dict:
    settings = get_settings()
    if not settings.mistral_api_key:
        return {"reply": "API key missing.", "is_complete": False, "audio_url": None}

    try:
        from mistralai import Mistral
        client = Mistral(api_key=settings.mistral_api_key)

        mistral_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            # We must map generic roles to 'user' or 'assistant' for Mistral
            role = "user" if m.role == "user" else "assistant"
            mistral_msgs.append({"role": role, "content": m.content})
            
        # Call Mistral Large for reasoning
        response = await client.chat.complete_async(
            model=settings.mistral_large_model,
            messages=mistral_msgs,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        raw_json = response.choices[0].message.content
        parsed = json.loads(raw_json)
        
        reply = parsed.get("reply", "I'm sorry, could you repeat that?")
        is_complete = parsed.get("is_complete", False)
        
        import hashlib
        # Synthesize audio with ElevenLabs!
        # Use a hash of the text to bypass the session cache inside voice_service
        unique_id = "triage-" + hashlib.md5(reply.encode()).hexdigest()[:8]
        voice_result = await voice_service.synthesize_clarification(
            session_id=unique_id, 
            clarification_prompt=reply
        )
        audio_url = voice_result.audio_url if voice_result.success else None
        
        return {
            "reply": reply,
            "is_complete": is_complete,
            "audio_url": audio_url
        }
        
    except Exception as e:
        logger.error(f"Triage error: {e}")
        return {"reply": "I'm sorry, I encountered an error checking your data.", "is_complete": False, "audio_url": None}
