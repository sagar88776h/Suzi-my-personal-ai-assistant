/**
 * Suzy Siri-Grade Interactive UI Controller
 *
 * Orchestrates:
 *  - Apple Siri fluid animated canvas orb
 *  - Web Speech STT & Voice Interruption
 *  - Edge-TTS audio playback
 *  - Chat messaging & streaming states
 *  - Email Hub, Memory Vault, Mood switcher & Level 3 Confirmation
 */

(function () {
  "use strict";

  // State Management
  const state = {
    orbStatus: "idle", // 'idle' | 'listening' | 'thinking' | 'speaking' | 'executing'
    mood: "natural",
    isRecording: false,
    isPlayingAudio: false,
    pendingAction: null,
    reducedMotion: false,
    selectedVoice: "en-IN-NeerjaExpressiveNeural",
    speechRate: "+0%",
  };

  // DOM Elements
  const canvas = document.getElementById("siriOrbCanvas");
  const ctx = canvas.getContext("2d");
  const orbWrapper = document.getElementById("orbWrapper");
  const orbStateLabel = document.getElementById("orbStateLabel");
  const waveformBar = document.getElementById("waveformBar");
  const liveTranscript = document.getElementById("liveTranscript");
  const chatHistory = document.getElementById("chatHistory");
  const chatForm = document.getElementById("chatForm");
  const userInput = document.getElementById("userInput");
  const micBtn = document.getElementById("micBtn");
  const assistantMoodBadge = document.getElementById("assistantMoodBadge");
  const moodSelect = document.getElementById("moodSelect");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");

  // Drawers
  const emailDrawer = document.getElementById("emailDrawer");
  const memoryDrawer = document.getElementById("memoryDrawer");
  const settingsDrawer = document.getElementById("settingsDrawer");
  const openEmailBtn = document.getElementById("openEmailBtn");
  const openMemoryBtn = document.getElementById("openMemoryBtn");
  const openSettingsBtn = document.getElementById("openSettingsBtn");
  const closeEmailBtn = document.getElementById("closeEmailBtn");
  const closeMemoryBtn = document.getElementById("closeMemoryBtn");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");

  // Email Drawer Elements
  const emailStatusText = document.getElementById("emailStatusText");
  const emailList = document.getElementById("emailList");
  const emailSearchInput = document.getElementById("emailSearchInput");
  const emailSearchBtn = document.getElementById("emailSearchBtn");
  const emailUnreadBadge = document.getElementById("emailUnreadBadge");

  // Memory Drawer Elements
  const memoryList = document.getElementById("memoryList");
  const memoryInput = document.getElementById("memoryInput");
  const memoryCatSelect = document.getElementById("memoryCatSelect");
  const saveMemoryBtn = document.getElementById("saveMemoryBtn");
  const memoryCountBadge = document.getElementById("memoryCountBadge");

  // Confirmation Card
  const actionConfirmationCard = document.getElementById("actionConfirmationCard");
  const confActionDesc = document.getElementById("confActionDesc");
  const btnConfirmAction = document.getElementById("btnConfirmAction");
  const btnCancelAction = document.getElementById("btnCancelAction");

  // Settings
  const voiceSelect = document.getElementById("voiceSelect");
  const speechRateRange = document.getElementById("speechRateRange");
  const rateLabel = document.getElementById("rateLabel");
  const reducedMotionCheck = document.getElementById("reducedMotionCheck");

  // =========================================================================
  // 1. Siri Fluid Orb Canvas Renderer
  // =========================================================================
  let animationFrameId = null;
  let phase = 0;

  function renderOrb() {
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const baseRadius = width * 0.35;

    ctx.clearRect(0, 0, width, height);

    if (state.reducedMotion) {
      // Simplified static glowing orb for low-resource devices
      const grad = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, baseRadius);
      grad.addColorStop(0, "rgba(0, 242, 254, 0.9)");
      grad.addColorStop(0.5, "rgba(127, 0, 255, 0.8)");
      grad.addColorStop(1, "rgba(248, 87, 166, 0.2)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
      ctx.fill();
      return;
    }

    // Dynamic state multipliers
    let speed = 0.03;
    let amplitude = 12;
    let alpha = 0.85;

    if (state.orbStatus === "listening") {
      speed = 0.08;
      amplitude = 24;
      alpha = 0.95;
    } else if (state.orbStatus === "thinking") {
      speed = 0.12;
      amplitude = 18;
    } else if (state.orbStatus === "speaking") {
      speed = 0.07;
      amplitude = 20;
    } else if (state.orbStatus === "executing") {
      speed = 0.05;
      amplitude = 14;
    }

    phase += speed;

    // Draw multi-layered swirling harmonic waves
    const layers = [
      { color1: "rgba(0, 242, 254, 0.7)", color2: "rgba(79, 172, 254, 0.1)", freq: 4, offset: 0 },
      { color1: "rgba(127, 0, 255, 0.65)", color2: "rgba(248, 87, 166, 0.1)", freq: 5, offset: Math.PI / 3 },
      { color1: "rgba(248, 87, 166, 0.6)", color2: "rgba(0, 242, 254, 0.1)", freq: 6, offset: Math.PI / 1.5 },
    ];

    layers.forEach((layer) => {
      ctx.save();
      ctx.beginPath();

      const numPoints = 120;
      for (let i = 0; i <= numPoints; i++) {
        const angle = (i / numPoints) * Math.PI * 2;
        const wave = Math.sin(angle * layer.freq + phase + layer.offset) * amplitude;
        const r = baseRadius + wave;
        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();

      const grad = ctx.createRadialGradient(centerX, centerY, 10, centerX, centerY, baseRadius + amplitude);
      grad.addColorStop(0, layer.color1);
      grad.addColorStop(0.7, layer.color2);
      grad.addColorStop(1, "rgba(0,0,0,0)");

      ctx.fillStyle = grad;
      ctx.globalCompositeOperation = "screen";
      ctx.fill();
      ctx.restore();
    });

    // Central luminous core
    ctx.save();
    const coreGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius * 0.65);
    coreGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
    coreGrad.addColorStop(0.3, "rgba(0, 242, 254, 0.5)");
    coreGrad.addColorStop(0.7, "rgba(127, 0, 255, 0.2)");
    coreGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = coreGrad;
    ctx.globalCompositeOperation = "screen";
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius * 0.65, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    animationFrameId = requestAnimationFrame(renderOrb);
  }

  function setOrbStatus(newStatus) {
    state.orbStatus = newStatus;
    orbStateLabel.textContent = newStatus.charAt(0).toUpperCase() + newStatus.slice(1);
    orbStateLabel.style.color =
      newStatus === "listening"
        ? "var(--siri-magenta)"
        : newStatus === "thinking"
        ? "var(--accent-primary)"
        : newStatus === "speaking"
        ? "var(--success)"
        : "var(--text-muted)";

    if (newStatus === "listening") {
      waveformBar.classList.add("active");
      micBtn.classList.add("listening");
    } else {
      waveformBar.classList.remove("active");
      micBtn.classList.remove("listening");
    }
  }

  // =========================================================================
  // 2. Speech-to-Text (STT) & Microphone Handler
  // =========================================================================
  let recognition = null;

  function initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      console.warn("Web Speech API not supported in this browser. Voice input fallback active.");
      return;
    }

    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-IN"; // Default Indian English

    recognition.onstart = () => {
      state.isRecording = true;
      setOrbStatus("listening");
      liveTranscript.textContent = "Listening...";
    };

    recognition.onresult = (event) => {
      let interim = "";
      let final = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          final += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }

      liveTranscript.textContent = final || interim;

      if (final) {
        userInput.value = final;
        submitUserMessage(final);
      }
    };

    recognition.onerror = (event) => {
      console.warn("STT error:", event.error);
      stopListening();
    };

    recognition.onend = () => {
      stopListening();
    };
  }

  function startListening() {
    interruptSpeaking();
    if (recognition) {
      try {
        recognition.start();
      } catch (err) {
        console.warn("Recognition already started:", err);
      }
    } else {
      userInput.focus();
    }
  }

  function stopListening() {
    state.isRecording = false;
    if (state.orbStatus === "listening") {
      setOrbStatus("idle");
    }
    setTimeout(() => {
      if (state.orbStatus === "idle") liveTranscript.textContent = "";
    }, 2000);
    if (recognition) {
      try {
        recognition.stop();
      } catch (e) {}
    }
  }

  function toggleListening() {
    if (state.isPlayingAudio) {
      interruptSpeaking();
      return;
    }
    if (state.isRecording) {
      stopListening();
    } else {
      startListening();
    }
  }

  // =========================================================================
  // 3. Text-to-Speech (TTS) & Voice Interruption
  // =========================================================================
  function interruptSpeaking() {
    if (state.isPlayingAudio || !ttsAudioPlayer.paused) {
      ttsAudioPlayer.pause();
      ttsAudioPlayer.currentTime = 0;
      state.isPlayingAudio = false;
      setOrbStatus("idle");
    }
  }

  async function speakText(text) {
    if (!text || state.mood === "quiet") return;
    interruptSpeaking();

    setOrbStatus("speaking");
    state.isPlayingAudio = true;

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          voice: state.selectedVoice,
          rate: state.speechRate,
        }),
      });

      if (!res.ok) {
        setOrbStatus("idle");
        state.isPlayingAudio = false;
        return;
      }

      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      ttsAudioPlayer.src = audioUrl;

      ttsAudioPlayer.onended = () => {
        state.isPlayingAudio = false;
        setOrbStatus("idle");
        URL.revokeObjectURL(audioUrl);
      };

      ttsAudioPlayer.onerror = () => {
        state.isPlayingAudio = false;
        setOrbStatus("idle");
      };

      await ttsAudioPlayer.play();
    } catch (err) {
      console.warn("TTS playback error:", err);
      state.isPlayingAudio = false;
      setOrbStatus("idle");
    }
  }

  // =========================================================================
  // 4. Chat & Message Processing
  // =========================================================================
  function appendMessage(sender, text, extra = {}) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender === "user" ? "user-message" : "assistant-message"}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = sender === "user" ? "You" : "S";

    const contentDiv = document.createElement("div");
    contentDiv.className = "msg-content";

    const author = document.createElement("div");
    author.className = "msg-author";
    author.textContent = sender === "user" ? "You" : "Suzy";

    if (extra.latency_ms) {
      const latSpan = document.createElement("span");
      latSpan.style.color = "var(--text-muted)";
      latSpan.style.fontSize = "0.7rem";
      latSpan.textContent = `(${extra.latency_ms}ms)`;
      author.appendChild(latSpan);
    }

    contentDiv.appendChild(author);

    const textP = document.createElement("p");
    textP.textContent = text;
    contentDiv.appendChild(textP);

    // Tool actions badges if any
    if (extra.actions && extra.actions.length > 0) {
      extra.actions.forEach((act) => {
        const badge = document.createElement("div");
        badge.className = "tool-badge";
        badge.textContent = `⚡ ${act.tool}()`;
        contentDiv.appendChild(badge);
      });
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  async function submitUserMessage(userText) {
    if (!userText || !userText.trim()) return;
    const cleanText = userText.trim();
    userInput.value = "";
    liveTranscript.textContent = "";

    appendMessage("user", cleanText);
    setOrbStatus("thinking");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: cleanText, mood: state.mood }),
      });

      const data = await res.json();
      setOrbStatus("speaking");

      appendMessage("assistant", data.text, {
        actions: data.actions,
        latency_ms: data.latency_ms,
      });

      // Handle Level 3 Pending Action
      if (data.pending_action) {
        state.pendingAction = data.pending_action;
        confActionDesc.textContent = `${data.pending_action.summary} (${data.pending_action.tool})`;
        actionConfirmationCard.classList.remove("hidden");
      } else {
        actionConfirmationCard.classList.add("hidden");
      }

      // Voice output
      speakText(data.text);
    } catch (err) {
      setOrbStatus("idle");
      appendMessage("assistant", "Sorry, I had trouble connecting to my service.");
    }
  }

  // =========================================================================
  // 5. Drawers: Email Hub, Memory Vault, Settings
  // =========================================================================
  function toggleDrawer(drawer) {
    [emailDrawer, memoryDrawer, settingsDrawer].forEach((d) => {
      if (d !== drawer) d.classList.remove("open");
    });
    drawer.classList.toggle("open");
  }

  async function loadEmailHub() {
    emailList.innerHTML = '<div class="empty-state">Loading latest emails...</div>';
    try {
      const res = await fetch("/api/email?limit=6");
      const data = await res.json();

      if (data.status && data.status.configured) {
        emailStatusText.textContent = `Connected (${data.status.active_provider.toUpperCase()})`;
      } else {
        emailStatusText.textContent = "No email connected (see .env)";
      }

      if (!data.messages || data.messages.length === 0) {
        emailList.innerHTML = '<div class="empty-state">No emails found in Inbox.</div>';
        return;
      }

      emailList.innerHTML = "";
      data.messages.forEach((msg) => {
        const item = document.createElement("div");
        item.className = "email-item";
        item.innerHTML = `
          <div class="email-item-header">
            <span class="email-item-sender">${msg.sender_name || msg.sender_email}</span>
            <span>${msg.date}</span>
          </div>
          <div class="email-item-subj">${msg.subject}</div>
          <div class="email-item-snippet">${msg.snippet}</div>
          <div class="email-item-actions">
            <button class="email-action-link" data-draft-id="${msg.id}">Draft Reply</button>
          </div>
        `;
        emailList.appendChild(item);
      });

      // Bind draft reply buttons
      emailList.querySelectorAll("[data-draft-id]").forEach((btn) => {
        btn.onclick = (e) => {
          const mid = e.target.getAttribute("data-draft-id");
          userInput.value = `Draft a reply to email ${mid}`;
          emailDrawer.classList.remove("open");
          submitUserMessage(userInput.value);
        };
      });
    } catch (err) {
      emailList.innerHTML = '<div class="empty-state">Failed to load emails.</div>';
    }
  }

  async function loadMemoryVault() {
    memoryList.innerHTML = '<div class="empty-state">Loading memories...</div>';
    try {
      const res = await fetch("/api/memory");
      const data = await res.json();
      const mems = data.memories || [];
      memoryCountBadge.textContent = mems.length;

      if (mems.length === 0) {
        memoryList.innerHTML = '<div class="empty-state">No saved memories yet.</div>';
        return;
      }

      memoryList.innerHTML = "";
      mems.forEach((m) => {
        const item = document.createElement("div");
        item.className = "memory-item";
        item.innerHTML = `
          <div>
            <span class="memory-tag">${m.category}</span>
            <span>${m.content}</span>
          </div>
          <button class="del-mem-btn" data-del-id="${m.id}" title="Delete memory">&times;</button>
        `;
        memoryList.appendChild(item);
      });

      memoryList.querySelectorAll("[data-del-id]").forEach((btn) => {
        btn.onclick = async (e) => {
          const id = e.target.getAttribute("data-del-id");
          await fetch(`/api/memory/${id}`, { method: "DELETE" });
          loadMemoryVault();
        };
      });
    } catch (err) {
      memoryList.innerHTML = '<div class="empty-state">Failed to load memories.</div>';
    }
  }

  // =========================================================================
  // 6. Action Confirmation (Level 3: Execute)
  // =========================================================================
  async function resolveAction(confirmed) {
    actionConfirmationCard.classList.add("hidden");
    try {
      const res = await fetch("/api/action/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmed: confirmed }),
      });
      const data = await res.json();
      appendMessage("assistant", data.message);
      speakText(data.message);
    } catch (err) {
      appendMessage("assistant", "Could not complete action.");
    }
    state.pendingAction = null;
  }

  // =========================================================================
  // 7. Event Listeners & Initialization
  // =========================================================================
  function initEvents() {
    // Form submission
    chatForm.onsubmit = (e) => {
      e.preventDefault();
      submitUserMessage(userInput.value);
    };

    // Quick chips
    document.querySelectorAll(".chip").forEach((chip) => {
      chip.onclick = () => {
        const cmd = chip.getAttribute("data-cmd");
        userInput.value = cmd;
        submitUserMessage(cmd);
      };
    });

    // Mic & Orb clicks
    micBtn.onclick = toggleListening;
    orbWrapper.onclick = toggleListening;

    // Mood Selector
    moodSelect.onchange = async () => {
      const newMood = moodSelect.value;
      state.mood = newMood;
      assistantMoodBadge.textContent = newMood.charAt(0).toUpperCase() + newMood.slice(1);
      await fetch("/api/mood", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mood: newMood }),
      });
    };

    // Theme Toggle
    themeToggleBtn.onclick = () => {
      document.body.classList.toggle("theme-light");
      document.body.classList.toggle("theme-dark");
    };

    // Drawers
    openEmailBtn.onclick = () => {
      toggleDrawer(emailDrawer);
      if (emailDrawer.classList.contains("open")) loadEmailHub();
    };
    closeEmailBtn.onclick = () => emailDrawer.classList.remove("open");

    openMemoryBtn.onclick = () => {
      toggleDrawer(memoryDrawer);
      if (memoryDrawer.classList.contains("open")) loadMemoryVault();
    };
    closeMemoryBtn.onclick = () => memoryDrawer.classList.remove("open");

    openSettingsBtn.onclick = () => toggleDrawer(settingsDrawer);
    closeSettingsBtn.onclick = () => settingsDrawer.classList.remove("open");

    // Email Search
    emailSearchBtn.onclick = async () => {
      const q = emailSearchInput.value.trim();
      if (!q) return loadEmailHub();
      emailList.innerHTML = '<div class="empty-state">Searching...</div>';
      const res = await fetch(`/api/email/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      const results = data.results || [];
      if (results.length === 0) {
        emailList.innerHTML = '<div class="empty-state">No matching emails found.</div>';
        return;
      }
      emailList.innerHTML = "";
      results.forEach((msg) => {
        const item = document.createElement("div");
        item.className = "email-item";
        item.innerHTML = `
          <div class="email-item-header">
            <span class="email-item-sender">${msg.sender_name || msg.sender_email}</span>
            <span>${msg.date}</span>
          </div>
          <div class="email-item-subj">${msg.subject}</div>
          <div class="email-item-snippet">${msg.snippet}</div>
        `;
        emailList.appendChild(item);
      });
    };

    // Save Memory
    saveMemoryBtn.onclick = async () => {
      const fact = memoryInput.value.trim();
      const cat = memoryCatSelect.value;
      if (!fact) return;
      await fetch("/api/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fact: fact, category: cat }),
      });
      memoryInput.value = "";
      loadMemoryVault();
    };

    // Confirmation Buttons
    btnConfirmAction.onclick = () => resolveAction(true);
    btnCancelAction.onclick = () => resolveAction(false);

    // Settings
    voiceSelect.onchange = () => {
      state.selectedVoice = voiceSelect.value;
    };
    speechRateRange.oninput = () => {
      const val = parseInt(speechRateRange.value);
      state.speechRate = val >= 0 ? `+${val}%` : `${val}%`;
      rateLabel.textContent = `${(1 + val / 100).toFixed(1)}x`;
    };
    reducedMotionCheck.onchange = () => {
      state.reducedMotion = reducedMotionCheck.checked;
    };

    // Keyboard Shortcuts
    window.onkeydown = (e) => {
      if (e.key === "Escape") {
        interruptSpeaking();
        stopListening();
        [emailDrawer, memoryDrawer, settingsDrawer].forEach((d) => d.classList.remove("open"));
      } else if (e.code === "Space" && e.ctrlKey) {
        e.preventDefault();
        toggleListening();
      }
    };
  }

  // =========================================================================
  // 8. Start Application
  // =========================================================================
  async function init() {
    initEvents();
    initSpeechRecognition();
    renderOrb();

    // Fetch initial status
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.mood) {
        state.mood = data.mood;
        moodSelect.value = data.mood;
        assistantMoodBadge.textContent = data.mood.charAt(0).toUpperCase() + data.mood.slice(1);
      }
      if (data.tts_voice) {
        state.selectedVoice = data.tts_voice;
        voiceSelect.value = data.tts_voice;
      }
      if (data.memories_count) {
        memoryCountBadge.textContent = data.memories_count;
      }
      if (data.brains) {
        document.getElementById("brainPrimary").textContent = data.brains.primary;
        document.getElementById("brainAvailable").textContent = data.brains.configured.join(", ");
      }
    } catch (e) {
      console.warn("Could not fetch server status:", e);
    }
  }

  window.onload = init;
})();
