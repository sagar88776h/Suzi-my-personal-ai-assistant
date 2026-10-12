# Suzy — your desktop AI assistant for Windows

Suzy is a voice assistant for Windows 10/11. She's warm, a bit witty, and she can
control your PC: open apps, change the volume, search the web, report on your
system, and plenty more as the build goes on. She understands English and Hinglish.

> **Build status:** Suzy is being built in phases. **Phases 1, 2 and 3 are done.**
> Suzy has a real AI brain (Groq, with Gemini as a backup) that runs PC actions:
> open apps, set the volume, tell the time and date, search Google, and report
> system info. She **speaks** her replies in a natural female voice and
> **listens** when you say **"Suzy"** (or press Enter / **Ctrl+Alt+S**). She can also **improve
> herself**: ask her for a new ability or to fix a problem and she changes her own
> code (see [Suzy improves herself](#suzy-improves-herself)). The
> floating orb UI and the rest of the tools come in later phases (see the [Roadmap](#roadmap)).

---

## 1. Easiest: one-command install

Open **PowerShell**: press the Windows key, type `powershell`, and press Enter.
Paste this line and press Enter:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/sagar88776h/Suzi-my-personal-ai-assistant/refs/heads/claude/youthful-sagan-m97rqy/install.ps1 | iex"
```

The installer does everything for you. It installs Python if you don't have it,
downloads Suzy into `C:\Users\<you>\Suzy`, installs the libraries, asks you to
paste your free API keys (and opens the pages where you get them), runs the
self-tests, puts a **Suzy** shortcut on your desktop, and starts Suzy. No admin
rights are needed.

After that, just double-click **Suzy** on your desktop. To update Suzy later, run
the same command again. Your keys and settings are kept.

If you'd rather do it by hand, follow the manual steps below.

## 1b. Manual setup, step by step

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

**Talking to Suzy, hands-free:** just say her name, any time:

- **"Suzy"**, **"Hi Suzy"** or **"Hello Suzy"**: you hear a chime, then say what you need.
- Or say it all at once: **"Hey Suzy, open Notepad"** or **"Suzy, volume thoda kam karo"**.
- **Follow-up:** after she answers, you have 8 seconds to ask something else *without*
  her name ("and tomorrow's weather?").
- **Interrupt:** say **"Suzy"** or **"Suzy, stop"** while she's talking. With laptop
  speakers she may not always hear you over her own voice; **Ctrl+Alt+S** always works.

She listens offline. A voice detector (Silero) notices speech, and only then does a
tiny speech model (about 75 MB, downloaded on first start) check for her name.
Nothing is sent anywhere until she hears "Suzy".

> *Why not openWakeWord?* It only ships ready-made models for other names (e.g. "Hey
> Jarvis"), and a "Suzy" model would need custom training. Recognising her name from
> text works out of the box for "Suzy", "Hi Suzy" and "Hello Suzy" alike, even with the
> request in the same sentence.

**Her voice:** a natural Indian female voice (`en-IN-NeerjaExpressiveNeural` by default).
To hear all the options and pick your favourite, double-click **Suzy Voices** on your
desktop (or run `python main.py --voices`). You can also just tell her: *"Suzy, change
your voice to Swara"*, *"speak a little faster"*, or *"lower your pitch a bit"*.

**Hindi and Hinglish:** when you talk to her in Hindi or Hinglish, she replies in Hinglish
with the Hindi words written in Hindi script (e.g. "Volume कम कर दिया"), and those replies
are spoken by a Hindi voice (`hi-IN-SwaraNeural`, set by `tts.hindi_voice`) so the Hindi
words are pronounced properly. English replies keep your main voice. (If Hindi letters show
as boxes in the window, open Suzy in *Windows Terminal*, which has Hindi fonts.)

**Speed:** your speech is turned into text by Groq's cloud Whisper (about half a second,
free with the same key, very good at Hindi and Hinglish). Your PC's own model takes over
automatically when you're offline. Simple actions are answered straight away, e.g.
"Opening Notepad.", without a second AI round-trip.

**Starts with Windows:** the installer asks whether Suzy should start automatically when you
turn on the PC. She then opens minimised and greets you: **"Hello Sagar, welcome back."**
Change the words with `assistant.startup_greeting` in `config.yaml`, or switch it on or off
any time by saying *"Suzy, start with Windows"* / *"Suzy, don't start with Windows"*.

**Push-to-talk also works:** press **Enter** on an empty line, or **Ctrl+Alt+S** from
any window, then speak and pause.

> The **first time** you start Suzy, she downloads her speech model (about 480 MB)
> in the background. You can type to her while it downloads. After that,
> listening works offline.

Try these (type them or say them):

| You type or say                           | What happens                                      |
|-------------------------------------------|---------------------------------------------------|
| `open notepad`                            | Notepad opens                                     |
| `open chrome`                             | Google Chrome opens (fuzzy names work)            |
| `set the volume to 30`                    | Volume goes to 30%                                |
| `thoda louder karo`                       | Volume goes up, and Suzy answers in Hinglish      |
| `what time is it?` / `aaj kya date hai?`  | Suzy tells the time / date                        |
| `search google for best biryani in Pune`  | Your browser opens the search                     |
| `how's my PC doing?`                      | CPU, RAM, battery, disk and uptime summary        |
| `open chrome and set volume to 50`        | Both actions, one after the other                 |
| `close this window` / `close Chrome`      | Closes it like clicking X (apps still ask to save) |
| `close this tab` / `switch to Word`       | Closes the browser tab / brings Word to the front |
| `minimize this` / `show the desktop`      | Minimises the window / all windows                |
| `press ctrl+shift+t` / `type hello`       | Presses keys / types into the window you're using |

Commands you can type in the chat: `/help`, `/tools` (lists everything Suzy can
do), `/mute` (voice off/on), `/reset` (forgets the conversation), `exit`.

Other ways to run it:

```powershell
python main.py --once "open calculator"   # run a single command and exit
python main.py --mute                     # Suzy types but doesn't speak
python main.py --no-voice                 # no microphone or hotkey (typing only)
python main.py --quiet                    # hide the "-> open_app(...)" action trace
python main.py --debug                    # show debug logs in the console
python main.py --check                    # test the API keys and AI models, then exit
python main.py --mic-test                 # test the microphone and speech recognition
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

## Suzy improves herself

Ask Suzy for something she can't do yet, or tell her to fix a problem:

> "Suzy, add a feature that tells me the latest cricket score."
> "Suzy, khud ko improve karo, jab main 'good night' bolun to volume 20 kar dena."
> "Suzy, fix your errors."

What happens next:

1. **She works on a private copy** of her code in the background (usually 2–10 minutes).
   You can keep using her meanwhile, and lines like `(working on myself: editing tools/cricket.py)`
   show her progress. "What's the status of your update?" also works.
2. **She tests the change.** Her full test suite must pass, otherwise she keeps fixing it
   or gives up and changes nothing.
3. **She asks you.** She tells you what she changed and which files. Say **"yes, install it"**
   to install, or **"discard it"**. She only installs after a clear yes in your latest message.
4. **She backs up and restarts.** Her current code is saved in the `backups` folder first.
5. **Automatic rescue.** If she crashes after an update, the guard (`guard.py`)
   restores the previous version on its own and tells you. You can also say
   **"undo your last update"** at any time.

She also **notices her own errors**. On startup, if her error log has a new
unexpected error, she starts fixing it by herself and asks before installing the fix.
Turn that off with `self_improvement.auto_fix_errors: false` in `config.yaml`.

**What she can change:** any of her own code and settings, except two files. `guard.py`
is her rescue system, and `.env` holds your keys, which she can never read either.

**Limits to know:**
- She codes with the free Groq model, which is much weaker at programming than a
  professional coding AI and has a small per-minute limit. Small and medium features and
  bug fixes work best. Large ones may fail, and then nothing is changed. She waits
  automatically when she hits the free limit, and may reply more slowly while she codes.
- When you update Suzy with the installer, files she changed may be replaced by the new
  version. The installer saves a copy of her code in `backups\installer-update-...`
  first, and abilities she *added* (new files) are always kept.

## 3. How it works

```
mic ──► Recorder (stops at your pause) ──► Whisper speech-to-text ─┐
                                                                   ▼
you type ──────────────────────────────────────────────────► Assistant
                                                                   │
speakers ◄── edge-tts voice (sentence by sentence) ◄── reply ◄─────┘
```

Inside the Assistant:

```
Assistant ──► LLM router ──► Groq (retry once) ──► Gemini fallback
                 ▲              │
                 │              └── the model picks a tool via function calling
                 │                                │
                 └──── tool result ◄── ToolRegistry.execute() ◄┘
                                     (validates args, never crashes, logs to actions.log)
```

* `core/llm.py` contains the Groq and Gemini providers and the failover router. If
  the primary fails, Suzy retries once and then switches to the fallback. A bad key
  skips the retry.
* `core/voice.py` runs one voice turn: listen, transcribe, think, speak. It is
  shared by the terminal and, later, the orb UI. `core/audio.py` handles the mic
  and speakers, `core/stt.py` handles Whisper, and `core/tts.py` handles the voice.
* `core/assistant.py` runs the tool-calling loop and keeps the last 20
  conversation turns as context.
* `tools/` holds one typed Python function per PC action, registered with the
  `@tool` decorator along with a description and a JSON schema.
* `logs/` has `suzy.log` (everything), `errors.log` (problems only) and
  `actions.log` (every action Suzy ran, with a timestamp). All three rotate automatically.

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| Not sure what's wrong with the brain | Run `python main.py --check` (or just start Suzy). She tests each brain and says exactly what's wrong. |
| *"The AI models in my settings aren't available any more"* | The provider retired a model. Re-run the installer, or put a current model name first under `llm.groq.models` / `llm.gemini.models` in `config.yaml`. |
| Gemini says the key was rejected (key starts with `AQ.`) | Some new-format Google keys are refused by the API right now. Groq on its own is enough. You can also create a fresh key later. |
| *"I don't have a working API key yet"* | Create `.env` (Step 6) and restart Suzy. |
| *"My API key was rejected"* | Re-copy the key with no spaces or quotes, or create a new one. |
| *"I can't reach the internet"* | Check your connection. Phase 6 adds offline commands. |
| *"I've hit my free usage limit"* | Wait a minute. Suzy also switches to Gemini automatically if that key is set. |
| *"I couldn't find an app called X"* | Use the name shown in the Start menu, or add the app's folder to `tools.apps.extra_search_dirs` in `config.yaml`. |
| Suzy doesn't hear me | Run the **mic test**: double-click **Suzy Mic Test** on your desktop, or run `python main.py --mic-test`. It lists your mics, shows a live level meter, tells you exactly what's wrong (Windows privacy block, wrong mic, too quiet), and checks speech recognition. |
| Suzy stops listening too early / too late | Raise or lower `voice.silence_seconds` in `config.yaml` (default 1.2). |
| Suzy misunderstands words | Set `stt.beam_size: 5` (a bit slower), or `stt.model: medium` (much slower but more accurate). For English only, set `stt.language: en`. |
| Change her voice or speed | `tts.voice`, `tts.rate` (`"+15%"` = faster) and `tts.pitch` in `config.yaml`. |
| Voice sounds robotic and Suzy says she's "using the Windows voice" | The natural voice service couldn't be reached (internet or firewall). It comes back by itself. |
| Want a different voice | Double-click **Suzy Voices**, or say "Suzy, change your voice to Swara / Ava / Neerja". |
| Robotic voice | No internet, so Suzy switched to the Windows built-in voice. The natural voice comes back automatically when you're online again. |
| *"I couldn't download my speech model"* | Suzy needs internet once to download it. Check your connection and restart her. |
| Speech model fails to load (DLL error) | Install the *Microsoft Visual C++ Redistributable (x64)* from Microsoft's website, then restart Suzy. |
| Suzy wakes up by mistake (TV, other people) | Raise `voice.wake_word.vad_threshold` (e.g. 0.7), or turn the wake word off with `voice.wake_word.enabled: false` and use Ctrl+Alt+S. |
| Suzy doesn't react to "Suzy" | Speak clearly and pause briefly after her name. Try `voice.wake_word.model: base` (more accurate, a bit slower). Push-to-talk always works. |
| Ctrl+Alt+S does nothing | Another app may own that shortcut. Pick another one in `voice.hotkey` (e.g. `ctrl+alt+space`). |
| Anything else | Look in `logs/errors.log`. |

## Roadmap

- [x] **Phase 1**: terminal chat, LLM tool calling with Groq→Gemini failover, and 5 basic tools
- [x] **Phase 2**: Suzy speaks (edge-tts, offline fallback voice) and listens with push-to-talk (faster-whisper)
- [x] **Self-improvement**: Suzy adds abilities and fixes her own bugs (with tests, your yes, backups and auto-rollback)
- [x] **Phase 3**: "Suzy" wake word (Silero VAD + tiny Whisper), interruption, 8-second follow-up mode
- [ ] **Phase 4**: the full tool set, the safety/confirmation layer, and tests for every tool
- [ ] **Phase 5**: floating orb UI, chat panel, tray icon, settings window
- [ ] **Phase 6**: offline fallback, auto-start, reminders, standalone `.exe` build
