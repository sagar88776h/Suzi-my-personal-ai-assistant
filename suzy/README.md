# Suzy — your desktop AI assistant for Windows

Suzy is a voice assistant for Windows 10/11. She's warm, a bit witty, and she can
control your PC: open apps, change the volume, search the web, report on your
system, and plenty more as the build goes on. She understands English and Hinglish.

> **Build status:** Suzy is being built in phases. **Phase 1 is done:** a text
> chat in the terminal with a real AI brain (Groq, with Gemini as a fallback) that
> can run these PC actions: open apps, set the volume, tell the time and date,
> search Google, and report system info. Voice, the wake word, the floating orb UI
> and the rest of the tools come in later phases (see the [Roadmap](#roadmap)).

---

## 1. Setup, step by step (for beginners)

You only need to do this once. Run the commands in **PowerShell**: press the
Windows key, type `powershell`, and press Enter.

### Step 1 — Install Python

1. Go to <https://www.python.org/downloads/windows/> and download
   **Python 3.12** (64-bit). 3.11 also works.
2. Run the installer. **On the first screen, tick "Add python.exe to PATH"**,
   then click **Install Now**.
3. Close PowerShell, open a new one, and check the install:

   ```powershell
   python --version
   ```

   You should see `Python 3.12.x`. If Windows opens the Microsoft Store
   instead, turn off the `python.exe` entries under
   *Settings → Apps → Advanced app settings → App execution aliases*, then try again.

### Step 2 — Get the code

If you have Git installed:

```powershell
cd $HOME\Documents
git clone https://github.com/sagar88776h/Suzi-my-personal-ai-assistant.git
cd Suzi-my-personal-ai-assistant\suzy
```

Without Git: on the GitHub page, click **Code → Download ZIP**, extract it
into `Documents`, and `cd` into the `suzy` folder inside it.

### Step 3 — Create a virtual environment (venv)

A venv keeps Suzy's libraries separate from everything else on your PC.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. If PowerShell says *"running
scripts is disabled on this system"*, run the following once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> Every time you open a new PowerShell to run Suzy, `cd` into the `suzy`
> folder and run `.\.venv\Scripts\Activate.ps1` first.

### Step 4 — Install the requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 — Get your free API keys

Suzy's "brain" is a cloud AI. Both of these are free and need no credit card.

**Groq (primary brain, very fast)**
1. Go to <https://console.groq.com/keys> and sign in with Google or email.
2. Click **Create API Key**, give it a name such as `suzy`, and copy the key (it starts with `gsk_`).

**Google Gemini (backup brain)**
1. Go to <https://aistudio.google.com/apikey> and sign in with your Google account.
2. Click **Create API key** and copy it (it starts with `AIza`).

### Step 6 — Put the keys in a `.env` file

```powershell
copy .env.example .env
notepad .env
```

Paste your keys after the `=` signs, then save and close:

```
GROQ_API_KEY=gsk_your_real_key
GEMINI_API_KEY=AIza_your_real_key
```

Keep this file private and never share it. Git is already set up to ignore it.

### Step 7 — Run Suzy

```powershell
python main.py
```

Try these:

| You type                                  | What happens                                      |
|-------------------------------------------|---------------------------------------------------|
| `open notepad`                            | Notepad opens                                     |
| `open chrome`                             | Google Chrome opens (fuzzy names work)            |
| `set the volume to 30`                    | Volume goes to 30%                                |
| `thoda louder karo`                       | Volume goes up, and Suzy answers in Hinglish      |
| `what time is it?` / `aaj kya date hai?`  | Suzy tells the time / date                        |
| `search google for best biryani in Pune`  | Your browser opens the search                     |
| `how's my PC doing?`                      | CPU, RAM, battery, disk and uptime summary        |
| `open chrome and set volume to 50`        | Both actions, one after the other                 |

Commands you can type in the chat: `/help`, `/tools` (lists everything Suzy can
do), `/reset` (forgets the conversation), `exit`.

Other ways to run it:

```powershell
python main.py --once "open calculator"   # run a single command and exit
python main.py --quiet                    # hide the "-> open_app(...)" action trace
python main.py --debug                    # show debug logs in the console
```

### Step 8 — Run the tests (optional)

```powershell
pytest
```

All tests should pass. They mock anything that would really change your PC.

---

## 2. Configuration

* **`config.yaml`**: every setting, with a comment on each one. This is where
  you switch the main brain (`llm.provider: gemini`), change models, or tune how
  strict app-name matching is.
* **`.env`**: your API keys only.

If `config.yaml` is missing or broken, Suzy falls back to safe defaults and tells you.

## 3. How it works

```
you type ──► Assistant ──► LLM router ──► Groq (retry once) ──► Gemini fallback
                 ▲              │
                 │              └── the model picks a tool via function calling
                 │                                │
                 └──── tool result ◄── ToolRegistry.execute() ◄┘
                                     (validates args, never crashes, logs to actions.log)
```

* `core/llm.py` contains the Groq and Gemini providers and the failover router. If
  the primary fails, Suzy retries once and then switches to the fallback. A bad key
  skips the retry.
* `core/assistant.py` runs the tool-calling loop and keeps the last 20
  conversation turns as context.
* `tools/` holds one typed Python function per PC action, registered with the
  `@tool` decorator along with a description and a JSON schema.
* `logs/` has `suzy.log` (everything), `errors.log` (problems only) and
  `actions.log` (every action Suzy ran, with a timestamp). All three rotate automatically.

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| *"I don't have a working API key yet"* | Create `.env` (Step 6) and restart Suzy. |
| *"My API key was rejected"* | Re-copy the key with no spaces or quotes, or create a new one. |
| *"I can't reach the internet"* | Check your connection. Phase 6 adds offline commands. |
| *"I've hit my free usage limit"* | Wait a minute. Suzy also switches to Gemini automatically if that key is set. |
| *"I couldn't find an app called X"* | Use the name shown in the Start menu, or add the app's folder to `tools.apps.extra_search_dirs` in `config.yaml`. |
| Anything else | Look in `logs/errors.log`. |

## Roadmap

- [x] **Phase 1**: terminal chat, LLM tool calling with Groq→Gemini failover, and 5 basic tools
- [ ] **Phase 2**: Suzy speaks (edge-tts) and listens with push-to-talk (faster-whisper)
- [ ] **Phase 3**: "Hey Suzy" wake word, voice activity detection, interruption, follow-up mode
- [ ] **Phase 4**: the full tool set, the safety/confirmation layer, and tests for every tool
- [ ] **Phase 5**: floating orb UI, chat panel, tray icon, settings window
- [ ] **Phase 6**: offline fallback, auto-start, reminders, standalone `.exe` build
