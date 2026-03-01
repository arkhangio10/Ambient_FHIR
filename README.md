# 🏥 Revenue Cycle Copilot (FHIR & Prior-Auth AI)

An AI-native, voice-first copilot that seamlessly covert unstructured clinician audio (or text) into validated, payer-ready prior-authorization packets and FHIR-aligned payloads. Built for the modern Behavioral Health professional.

## 🚀 The Problem
Clinicians spend hours every day wrestling with EHRs, translating clinical encounters into complex billing codes, and fighting prior authorization rejections. This administrative burden causes burnout and delays patient care.

## 💡 Our Solution
**Revenue Cycle Copilot** acts as an ambient assistant. 
1. **Listen & Transcribe**: Uses Mistral's Voxtral models to capture clinical encounters in real-time.
2. **Reason & Extract**: Leverages Mistral Large 3 to accurately pull out Chief Complaints, Diagnoses, CPT/ICD-10 Codes, and clinical rationale.
3. **Clarify**: If anything is missing (e.g., missing a CPT code), the AI generates a focused voice prompt using ElevenLabs TTS to ask the clinician immediately.
4. **Standardize**: Deterministically formats the validated data into an exportable Prior-Authorization Packet and an interoperable FHIR R4 Bundle.

## 🛠️ Tech Stack
*   **Frontend**: React, Vite, Tailwind CSS (Glassmorphism design)
*   **Backend**: Python, FastAPI, WebSockets
*   **AI Reasoning & Transcription**: Mistral AI (Large 3 & Voxtral)
*   **Voice Synthesis**: ElevenLabs TTS API
*   **Health Standards**: FHIR R4 (Epic Sandbox compatible)

---

## 💻 Quick Start Guide

### 1. Backend Setup (FastAPI)
```bash
# Clone & enter project
cd version_1

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment keys
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY and ELEVENLABS_API_KEY

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup (React)
```bash
cd clinical-flow-copilot
npm install
npm run dev
```
Open `http://localhost:8080` to view the agent.

## 🔑 Required Environment Variables
| Key | Service | Purpose |
|-----|---------|---------|
| `MISTRAL_API_KEY` | Mistral AI | Voxtral live transcription, Mistral Large reasoning |
| `ELEVENLABS_API_KEY` | ElevenLabs | Voice prompts for clarification loop |
| `EPIC_FHIR_TOKEN` | Epic on FHIR | Sandbox EHR integration (optional) |
