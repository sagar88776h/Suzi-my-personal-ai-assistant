# Suzy — Personal AI Operating Assistant for Windows

**Suzy** is an intelligent personal operating assistant for Windows 10/11 inspired by the simplicity and polish of Apple Siri, powered by modern AI reasoning.

## ✨ Key Features

- **Siri-Level Voice & Web Experience**: Luminous fluid animated orb, real-time push-to-talk speech recognition, natural neural voice synthesis (Indian English `en-IN-NeerjaExpressiveNeural`, Hindi `hi-IN-SwaraNeural`), live waveform visualizer, and instant voice interruption.
- **Deep Email Integration**: Official support for Outlook on Windows and Gmail/IMAP over SSL. Search inbox, read threads with attachments, summarize key communications with grounded citations, and prepare reviewable draft replies (Level 2: Prepare).
- **Persistent Personal Memory Vault**: Remember preferences, ongoing projects, and explicit facts across sessions in a private, encrypted local SQLite database (`data/memory.db`) with strict security safeguards against credential leaks.
- **Configurable Personality Moods**: Switch seamlessly between 6 tones: **Natural**, **Professional**, **Focus**, **Coach**, **Creative**, and **Quiet** (via voice commands, chat, or UI selector).
- **3-Level Safety & Permission System**:
  - **Level 1 (Read)**: Safe queries executed immediately.
  - **Level 2 (Prepare)**: Staging drafts and proposed actions for review.
  - **Level 3 (Execute)**: Consequential actions requiring explicit user confirmation.
- **Dual Interface**:
  - **Interactive Siri Web & Desktop UI** (`python main.py --ui` or double-click `Start Suzy UI.bat`)
  - **Terminal Power CLI** (`python main.py`) with global hotkey (`Ctrl+Alt+S`) and offline wake word ("Hey Suzy").

## 🚀 Quick Launch

### 1. Launch the Siri Web Interface
Double-click `Start Suzy UI.bat` or run:
```powershell
python main.py --ui
```
Opens in your browser at `http://127.0.0.1:8000`.

### 2. Launch Terminal Chat / Voice Assistant
```powershell
python main.py
```

### 3. Run Self-Tests
```powershell
pytest
```
*(512 automated tests covering brain routing, PC tools, email retrieval, memory persistence, moods, permissions, and web server)*

## 🔐 Environment Configuration (`suzy/.env`)

```env
# AI Brains (free keys)
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=AIza_your_key_here
OLLAMA_API_KEY=your_key_here

# Email Integration (Optional: connect Outlook directly, or add IMAP/Gmail credentials)
GMAIL_USER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_character_app_password
# Or custom IMAP:
# IMAP_HOST=imap.example.com
# IMAP_USER=user@example.com
# IMAP_PASSWORD=password
```
