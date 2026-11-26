/**
 * Voicebot Demo - Frontend Client (app.js)
 * Tích hợp AudioWorklet 16kHz + Butterworth filter, PCM 48kHz scheduling.
 */

(() => {
    const DEFAULT_TTS_SR = 48000;

    const STATE_LABELS = {
        idle: "Rảnh",
        listening: "Đang nghe...",
        thinking: "Đang suy nghĩ...",
        speaking: "Bot đang nói...",
        error: "Lỗi hệ thống",
    };

    // Trạng thái kết nối & đàm thoại
    let ws = null;
    let currentTurnId = null;
    let serverState = "idle";
    let recording = false;
    let callActive = false;
    let discarding = false;

    // Web Audio Context & Nodes
    let audioCtx = null;
    let micStream = null;
    let micSource = null;
    let processorNode = null;
    let workletRegistered = false;
    let onWorkletFlushed = null;
    let ttsStreamDest = null;
    let ttsAudioEl = null;
    let aecPcs = [];

    // Playback scheduling
    let ttsSampleRate = DEFAULT_TTS_SR;
    let playHead = 0;
    const playingSources = new Set();
    let waitAudioEndTimer = null;
    let autoListenTimer = null;
    let pendingChunkMeta = null;

    // DOM Elements
    const $ = (id) => document.getElementById(id);
    const connectionBadge = $("connectionBadge");
    const connectionText = $("connectionText");
    const stateEl = $("state");
    const errEl = $("err");
    const buttonsRow = document.querySelector(".buttons-row");
    const btnTalk = $("btnTalk");
    const btnFinish = $("btnFinish");
    const btnStop = $("btnStop");
    const btnReset = $("btnReset");
    const youBox = $("you");
    const botBox = $("bot");
    const DEFAULT_YOU_HINT = youBox.dataset.placeholder || "";
    const DEFAULT_BOT_HINT = botBox.dataset.placeholder || "";
    const setPlaceholder = (el, text) => {
        el.textContent = "";
        el.dataset.placeholder = text;
    };

    // AudioWorklet Processor đa luồng resample 16kHz + đệm 100ms
    const WORKLET_CODE = `
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.resamplePhase = 0;
    this.resamplePrevSample = null;
    this.buf = new Float32Array(1600);
    this.fill = 0;
    this.port.onmessage = (e) => {
      if (e.data?.type === 'reset') {
        this.fill = 0;
        this.resamplePhase = 0;
        this.resamplePrevSample = null;
      } else if (e.data?.type === 'flush') {
        this.flush();
      }
    };
  }
  _sendPcm(flushed) {
    if (this.fill === 0) {
      if (flushed) this.port.postMessage({ flushed: true, pcm: null });
      return;
    }
    const pcm = new ArrayBuffer(this.fill * 2);
    const view = new DataView(pcm);
    for (let i = 0; i < this.fill; i++) {
      const s = Math.max(-1, Math.min(1, this.buf[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    this.port.postMessage({ flushed, pcm }, [pcm]);
    this.fill = 0;
  }
  flush() {
    this._sendPcm(true);
  }
  process(inputs) {
    const chunk = inputs[0]?.[0];
    if (!chunk || chunk.length === 0) return true;
    const ratio = sampleRate / 16000;
    let pos = this.resamplePhase;
    while (pos < chunk.length - 1) {
      const idx = Math.floor(pos);
      const p0 = pos < 0 ? (this.resamplePrevSample ?? chunk[0]) : chunk[idx];
      const p1 = pos < 0 ? chunk[0] : chunk[idx + 1];
      const frac = pos < 0 ? pos + 1 : pos - idx;
      this.buf[this.fill++] = p0 + (p1 - p0) * frac;
      if (this.fill === 1600) {
        this._sendPcm(false);
      }
      pos += ratio;
    }
    this.resamplePhase = pos - chunk.length;
    this.resamplePrevSample = chunk[chunk.length - 1];
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);
`;

    function log(msg, cls = "") {
        console.log(`[voicebot][${cls || "info"}]`, msg);
    }

    function setErr(msg) {
        errEl.textContent = msg || "";
    }

    function isWsOpen() {
        return ws && ws.readyState === WebSocket.OPEN;
    }

    function sendJson(obj) {
        if (isWsOpen()) {
            const str = JSON.stringify(obj);
            log(str, "out");
            ws.send(str);
        }
    }

    function connect() {
        if (ws) {
            ws.onopen = null;
            ws.onmessage = null;
            ws.onerror = null;
            ws.onclose = null;
            try { ws.close(); } catch (_) { }
            ws = null;
        }

        currentTurnId = null;
        pendingChunkMeta = null;
        serverState = "idle";
        discarding = false;

        const proto = location.protocol === "https:" ? "wss://" : "ws://";
        const url = `${proto}${location.host}/ws/voice`;
        log(`Đang kết nối tới ${url}...`, "info");
        stateEl.textContent = "Đang kết nối...";
        connectionBadge.classList.remove("connected");
        connectionText.textContent = "Đang kết nối...";

        try {
            ws = new WebSocket(url);
            ws.binaryType = "arraybuffer";
        } catch (err) {
            setErr("Lỗi khởi tạo WebSocket: " + err.message);
            return;
        }

        ws.onopen = () => {
            log("WebSocket đã kết nối.", "info");
            connectionBadge.classList.add("connected");
            connectionText.textContent = "Đã kết nối";
            setErr("");
            refreshState();
        };

        ws.onclose = (e) => {
            log(`WebSocket đã đóng (code=${e.code}).`, "info");
            connectionBadge.classList.remove("connected");
            connectionText.textContent = "Mất kết nối";
            teardownCallState();
            setErr("Mất kết nối với máy chủ.");
        };

        ws.onerror = () => setErr("Lỗi kết nối máy chủ WebSocket.");
        ws.onmessage = onMessage;
    }

    function onMessage(ev) {
        if (typeof ev.data !== "string") {
            if (!pendingChunkMeta) return;
            log(`Nhận audio chunk #${pendingChunkMeta.chunk_id}: ${ev.data.byteLength} bytes`, "bin-in");
            pendingChunkMeta = null;
            playPcm(ev.data);
            return;
        }

        let m;
        try {
            m = JSON.parse(ev.data);
            log(ev.data, "in");
        } catch (_) {
            return;
        }

        const isMatchTurn = !m.turn_id || m.turn_id === currentTurnId;

        switch (m.type) {
            case "voice.state": {
                const prevState = serverState;
                serverState = m.state;
                if (m.turn_id) {
                    currentTurnId = m.turn_id;
                }
                if (m.state === "listening") {
                    clearAutoListenTimers();
                    discarding = false;
                    if (callActive) recording = true;
                } else if (m.state === "thinking" || m.state === "speaking") {
                    recording = false;
                    resetAudioInputState();
                } else if (m.state === "error") {
                    clearAutoListenTimers();
                    recording = false;
                } else if (m.state === "idle") {
                    recording = false;
                    // Nếu TTS không phát âm thanh (không có assistant.audio.done), tự mở lại lượt nghe tiếp
                    if (prevState === "thinking" && !isPlaying() && !waitAudioEndTimer) {
                        queueNextTurn();
                    }
                }
                refreshState();
                break;
            }
            case "transcript.partial":
                if (isMatchTurn) youBox.textContent = m.text || "";
                break;
            case "transcript.final":
                if (isMatchTurn) {
                    if (m.text) {
                        youBox.textContent = m.text;
                    } else {
                        setPlaceholder(youBox, "(Không nhận diện được giọng nói)");
                    }
                    setPlaceholder(botBox, "Đang nhận phản hồi...");
                }
                break;
            case "assistant.text.delta":
                if (isMatchTurn) {
                    let delta = m.text || "";
                    if (!botBox.textContent) delta = delta.replace(/^[\r\n\s]+/, "");
                    if (delta) botBox.textContent += delta;
                }
                break;
            case "assistant.text.done":
                if (isMatchTurn && m.text) botBox.textContent = m.text.trim();
                break;
            case "assistant.audio.chunk":
                if (isMatchTurn) {
                    pendingChunkMeta = m;
                    if (m.sample_rate) ttsSampleRate = m.sample_rate;
                } else {
                    pendingChunkMeta = null;
                }
                break;
            case "assistant.audio.done":
                if (isMatchTurn) scheduleAutoListen();
                break;
            case "playback.stop":
                stopPlayback();
                break;
            case "session.error":
                setErr(m.message || "Lỗi xử lý phiên.");
                if (callActive) teardownCallState();
                break;
        }
    }

    async function setupAecLoopback() {
        if (ttsAudioEl || !audioCtx) return;
        let pc1 = null;
        let pc2 = null;
        try {
            ttsStreamDest = audioCtx.createMediaStreamDestination();
            pc1 = new RTCPeerConnection();
            pc2 = new RTCPeerConnection();

            pc1.onicecandidate = (e) => e.candidate && pc2.addIceCandidate(e.candidate);
            pc2.onicecandidate = (e) => e.candidate && pc1.addIceCandidate(e.candidate);

            const audioEl = document.createElement("audio");
            audioEl.autoplay = true;

            pc2.ontrack = (e) => {
                audioEl.srcObject = e.streams[0];
                audioEl.play().catch(() => { });
            };

            ttsStreamDest.stream.getTracks().forEach((track) => pc1.addTrack(track, ttsStreamDest.stream));

            const offer = await pc1.createOffer();
            await pc1.setLocalDescription(offer);
            await pc2.setRemoteDescription(offer);

            const answer = await pc2.createAnswer();
            await pc2.setLocalDescription(answer);
            await pc1.setRemoteDescription(answer);
            aecPcs = [pc1, pc2];
            ttsAudioEl = audioEl;
        } catch (e) {
            console.warn("Không thể khởi tạo AEC loopback, dùng audioCtx.destination dự phòng:", e);
            try { pc1?.close(); } catch (_) { }
            try { pc2?.close(); } catch (_) { }
            ttsStreamDest = null;
            ttsAudioEl = null;
        }
    }

    async function ensureAudio() {
        if (!audioCtx) audioCtx = new AudioContext();
        if (audioCtx.state === "suspended") await audioCtx.resume();
        await setupAecLoopback();
        if (!micStream) {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false, // Tắt AGC để không tự động khuếch đại tiếng người nói ở xa / đồng nghiệp
                },
            });
            micSource = audioCtx.createMediaStreamSource(micStream);

            // Highpass filter 85Hz (Q=0.7) lọc tạp âm trầm, tiếng quạt gió, điều hòa, rung chấn bàn
            const hp = audioCtx.createBiquadFilter();
            hp.type = "highpass";
            hp.frequency.value = 85;
            hp.Q.value = 0.707;

            // Butterworth bậc 4 (7.2 kHz)
            const lp1 = audioCtx.createBiquadFilter();
            lp1.type = "lowpass"; lp1.frequency.value = 7200; lp1.Q.value = 0.5411961;
            const lp2 = audioCtx.createBiquadFilter();
            lp2.type = "lowpass"; lp2.frequency.value = 7200; lp2.Q.value = 1.306563;
            micSource.connect(hp);
            hp.connect(lp1);
            lp1.connect(lp2);

            if (!workletRegistered) {
                const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
                const workletUrl = URL.createObjectURL(blob);
                await audioCtx.audioWorklet.addModule(workletUrl);
                URL.revokeObjectURL(workletUrl);
                workletRegistered = true;
            }

            processorNode = new AudioWorkletNode(audioCtx, "mic-processor");
            processorNode.port.onmessage = (e) => {
                const { flushed, pcm } = e.data || {};
                if (
                    pcm &&
                    isWsOpen() &&
                    (flushed || (callActive && recording && !isBotBusy()))
                ) {
                    ws.send(pcm);
                }
                if (flushed && typeof onWorkletFlushed === "function") {
                    const cb = onWorkletFlushed;
                    onWorkletFlushed = null;
                    cb();
                }
            };
            lp2.connect(processorNode);
            const mute = audioCtx.createGain();
            mute.gain.value = 0;
            processorNode.connect(mute);
            mute.connect(audioCtx.destination);
        }
    }

    function resetAudioInputState() {
        if (processorNode?.port) processorNode.port.postMessage({ type: "reset" });
    }

    function releaseMicrophone() {
        if (processorNode) {
            if (processorNode.port) processorNode.port.onmessage = null;
            try { processorNode.disconnect(); } catch (_) { }
            processorNode = null;
        }
        if (micSource) {
            try { micSource.disconnect(); } catch (_) { }
            micSource = null;
        }
        if (micStream) {
            micStream.getTracks().forEach((t) => t.stop());
            micStream = null;
        }
    }

    function playPcm(arrayBuf) {
        const evenLen = arrayBuf ? arrayBuf.byteLength & ~1 : 0;
        if (!audioCtx || discarding || evenLen < 2) return;
        const i16 = new Int16Array(arrayBuf, 0, evenLen / 2);
        const audioBuf = audioCtx.createBuffer(1, i16.length, ttsSampleRate);
        const channelData = audioBuf.getChannelData(0);
        for (let i = 0; i < i16.length; i++) channelData[i] = i16[i] / 32768;

        const source = audioCtx.createBufferSource();
        source.buffer = audioBuf;
        source.connect(ttsStreamDest || audioCtx.destination);

        const now = audioCtx.currentTime;
        if (playHead < now + 0.05) playHead = now + 0.08;
        source.start(playHead);
        playHead += audioBuf.duration;
        playingSources.add(source);

        source.onended = () => {
            playingSources.delete(source);
            refreshState();
        };
        refreshState();
    }

    function stopPlayback() {
        clearAutoListenTimers();
        pendingChunkMeta = null;
        for (const src of playingSources) {
            try { src.stop(); } catch (_) { }
        }
        playingSources.clear();
        playHead = 0;
        discarding = true;
        refreshState();
    }

    function isPlaying() { return playingSources.size > 0; }
    function isBotBusy() { return serverState === "thinking" || serverState === "speaking" || isPlaying(); }

    function setCallButtons(inCall) {
        buttonsRow?.classList.toggle("in-call", inCall);
    }

    function refreshState() {
        btnFinish.disabled = (serverState !== "listening");
        if (!isWsOpen()) { stateEl.textContent = "Mất kết nối"; return; }
        if (isPlaying() || serverState === "speaking") { stateEl.textContent = STATE_LABELS.speaking; return; }
        if (serverState === "thinking") { stateEl.textContent = STATE_LABELS.thinking; return; }
        if (recording || serverState === "listening") { stateEl.textContent = STATE_LABELS.listening; return; }
        if (serverState === "idle" && callActive) { stateEl.textContent = "Chờ nghe tiếp..."; return; }
        stateEl.textContent = STATE_LABELS[serverState] || serverState;
    }

    function clearAutoListenTimers() {
        if (waitAudioEndTimer) { clearInterval(waitAudioEndTimer); waitAudioEndTimer = null; }
        if (autoListenTimer) { clearTimeout(autoListenTimer); autoListenTimer = null; }
    }

    function queueNextTurn() {
        if (autoListenTimer || !callActive || recording) return;
        autoListenTimer = setTimeout(() => {
            autoListenTimer = null;
            if (callActive && !recording && (serverState === "idle" || serverState === "speaking")) {
                startListeningTurn();
            }
        }, 600);
    }

    function scheduleAutoListen() {
        recording = false;
        resetAudioInputState();
        clearAutoListenTimers();
        waitAudioEndTimer = setInterval(() => {
            if (!isPlaying()) {
                clearInterval(waitAudioEndTimer);
                waitAudioEndTimer = null;
                if (isWsOpen() && currentTurnId) {
                    sendJson({ type: "playback.done", turn_id: currentTurnId });
                }
                queueNextTurn();
            }
        }, 100);
    }

    function teardownCallState() {
        onWorkletFlushed = null;
        callActive = false;
        recording = false;
        stopPlayback();
        releaseMicrophone();
        setCallButtons(false);
    }

    function startListeningTurn() {
        currentTurnId = null;
        resetAudioInputState();
        sendJson({ type: "turn.start" });
        setCallButtons(true);
        refreshState();
    }

    function stopTalking() {
        if (!recording) return;
        recording = false;
        refreshState();

        if (processorNode?.port) {
            let sent = false;
            let fallbackTimer = null;
            const finishTurn = () => {
                if (sent) return;
                sent = true;
                if (fallbackTimer) clearTimeout(fallbackTimer);
                onWorkletFlushed = null;
                resetAudioInputState();
                sendJson({ type: "turn.finish", turn_id: currentTurnId });
            };
            onWorkletFlushed = finishTurn;
            processorNode.port.postMessage({ type: "flush" });
            fallbackTimer = setTimeout(finishTurn, 500);
        } else {
            resetAudioInputState();
            sendJson({ type: "turn.finish", turn_id: currentTurnId });
        }
    }

    // Gán sự kiện điều khiển
    btnTalk.onclick = async () => {
        if (!isWsOpen()) return setErr("Chưa kết nối được máy chủ. Bấm Hội thoại mới.");
        btnTalk.disabled = true;
        try {
            await ensureAudio();
        } catch (e) {
            return setErr("Không thể truy cập microphone: " + e.message);
        } finally {
            btnTalk.disabled = false;
        }
        setErr("");
        setPlaceholder(youBox, "Đang nghe...");
        setPlaceholder(botBox, "Đang chờ câu hỏi của bạn...");
        callActive = true;
        startListeningTurn();
    };

    btnFinish.onclick = stopTalking;
    btnStop.onclick = () => {
        teardownCallState();
        sendJson({ type: "session.cancel", turn_id: currentTurnId });
    };
    btnReset.onclick = () => {
        teardownCallState();
        setPlaceholder(youBox, DEFAULT_YOU_HINT);
        setPlaceholder(botBox, DEFAULT_BOT_HINT);
        setErr("");
        connect();
    };

    window.addEventListener("pagehide", () => {
        aecPcs.forEach((pc) => { try { pc.close(); } catch (_) { } });
        aecPcs = [];
    });
    connect();
})();
