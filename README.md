<div align="center">
  <h1>🏥 Revenue Cycle Copilot</h1>
  <p><strong>Ambient FHIR & Prior-Auth AI powered by Mistral & ElevenLabs</strong></p>
</div>

---

## 🚀 The Problem
Clinicians and Behavioral Health professionals spend hours every day wrestling with Electronic Health Records (EHRs), translating clinical encounters into complex billing codes (CPT/ICD-10), and fighting prior-authorization rejections. This administrative burden limits the number of patients they can see, causes severe burnout, and ultimately delays patient care.

## 💡 Our Solution
**Revenue Cycle Copilot** acts as an ambient, voice-first intelligent assistant. 
Instead of forcing doctors to fill out endless forms, our application simply listens to the patient encounter and perfectly structures it into interoperable billing packets. 

Furthermore, we've built a **B2C Patient Triage Interface**, where patients can interact directly with an empathetic AI Voice assistant to collect clinical data before they even see the doctor.

### ✨ Key Features:
1. **Live Ambient Dictation:** Uses **Mistral Voxtral** via WebSockets to capture clinical encounters in real time with near-zero latency.
2. **Clinical Reasoning:** Leverages **Mistral Large 3** to accurately extract Chief Complaints, Diagnoses, CPT/ICD-10 Codes, and clinical rationale directly from raw transcripts.
3. **The Voice Clarification Loop:** Instead of generating broken claims, our Orchestrator runs a deterministic validation check. If a mandatory code or reasoning is missing, the AI uses **ElevenLabs TTS** to dynamically ask the clinician (or the patient!) for the missing detail.
4. **Interactive Patient Triage:** A fully conversational voice-guided triage system where the patient can speak their symptoms, and the AI will listen, analyze, and reply automatically using ElevenLabs voice.
5. **FHIR R4 Interoperability:** Deterministically formats the validated data into an exportable Prior-Authorization Packet and an interoperable FHIR Bundle (compatible with Epic Sandbox).

---

## 🛠️ Tech Stack & Architecture

### **AI Models & APIs**
*   **Mistral Voxtral Mini**: Lightning-fast, real-time audio transcription handling live buffers over WebSockets.
*   **Mistral Large 3**: Heavy clinical reasoning, entity extraction, and missing-field generation.
*   **ElevenLabs TTS**: Highly realistic voice synthesis for the dynamic Clarification Loop and Patient Triage flow.

### **Frontend (Lovable / React)**
*   **React 18** + **Vite**
*   **Tailwind CSS** with a stunning, modern Glassmorphism UI design.
*   Native browser Microphone APIs integrated with WebSocket streaming.

### **Backend (FastAPI)**
*   **Python 3.10+** + **FastAPI**
*   **WebSockets** for real-time audio chunk streaming.
*   Async architecture built to handle external API delays gracefully.

---

## 📂 Repository Structure
```text
📦 Ambient_FHIR
 ┣ 📂 app
 ┃ ┣ 📂 routes                # FastAPI endpoints (clinical, realtime WS, triage)
 ┃ ┣ 📂 schemas               # Pydantic models & validation for FHIR/Auth
 ┃ ┣ 📂 services              # Core AI integration (Mistral, ElevenLabs, Orchestrator)
 ┃ ┣ 📜 config.py             # Environment configurations
 ┃ ┗ 📜 main.py               # Application entry point
 ┣ 📂 clinical-flow-copilot   # The stunning React Frontend
 ┃ ┣ 📂 src
 ┃ ┃ ┣ 📂 components          # Reusable UI elements (Glassmorphism)
 ┃ ┃ ┣ 📂 context             # React Context for Session & Auth State
 ┃ ┃ ┣ 📂 pages               # Dashboard, Interview (Triage), etc.
 ┃ ┃ ┗ 📜 index.css           # Tailwind configuration
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 README.md               # You are here!
```

---

## 💻 Quick Start Guide

### 1. Requirements
*   Python 3.10 or higher
*   Node.js 18 or higher
*   Mistral AI API Key
*   ElevenLabs API Key

### 2. Backend Setup (FastAPI & AI Services)
```bash
# Clone the repository
git clone https://github.com/arkhangio10/Ambient_FHIR.git
cd Ambient_FHIR

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment keys
cp .env.example .env
# Open .env and insert your API keys:
# MISTRAL_API_KEY=your_key_here
# ELEVENLABS_API_KEY=your_key_here

# Run the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*The backend will be running at `http://localhost:8000`*

### 3. Frontend Setup (React Application)
```bash
# In a new terminal, navigate to the frontend directory
cd clinical-flow-copilot

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*The application UI will be available at `http://localhost:8080`*

---

## 🎥 User Flows

### Flow A: The Doctor (Revenue Cycle Copilot)
1. The Clinician visits the main Dashboard and presses the microphone.
2. The UI transcribes their dictation live (Voxtral).
3. The AI (Mistral Large 3) attempts to generate a Prior Authorization form.
4. If a piece of data is missing, the AI pauses, and ElevenLabs asks the Doctor for the missing data point.
5. The Doctor replies, the AI validates, and a perfect FHIR payload is exported.

### Flow B: The Patient (Interactive Triage)
1. The Patient visits the `Patient Interview` screen.
2. A friendly AI nurse (ElevenLabs) greets them.
3. The Patient clicks the microphone to describe their symptoms.
4. Mistral Voxtral transcribes the audio, Mistral Large 3 determines the next required diagnostic question, and ElevenLabs generates the responsive audio.
5. The conversation loops until triage is complete, automatically wrapping up the session.

---

## 👥 Hackathon Details
This project was built during the Hackathon utilizing **Mistral AI** and **ElevenLabs** as core pillars of the solution. Our architecture proves that ambient voice AI combined with strong clinical reasoning models can fundamentally fix the broken healthcare revenue cycle and patient intake processes.
