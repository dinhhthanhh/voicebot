/**
 * Voicebot Demo - Frontend Client (app.js)
 * Tích hợp AudioWorklet 16kHz + Butterworth filter, PCM 48kHz scheduling, Barge-in.
 */

(() => {
    const DEFAULT_TTS_SR = 48000;

    const STATE_LABELS = {
        idle: "Rảnh",
        listening: "Đang nghe...",
        thinking: "Đang suy nghĩ...",
        speaking: "Đang nói...",
        interrupted: "Đã ngắt lời",
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
    const btnTalk = $("btnTalk");
    const btnFinish = $("btnFinish");
    const btnStop = $("btnStop");
    const btnReset = $("btnReset");
    const youBox = $("you");
    const botBox = $("bot");

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
  flush() {
    if (this.fill > 0) {
      const pcm = new ArrayBuffer(this.fill * 2);
      const view = new DataView(pcm);
      for (let i = 0; i < this.fill; i++) {
        const s = Math.max(-1, Math.min(1, this.buf[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      this.port.postMessage({ flushed: true, pcm }, [pcm]);
      this.fill = 0;
    } else {
      this.port.postMessage({ flushed: true, pcm: null });
    }
  }
  process(inputs) {
    const chunk = inputs[0]?.[0];
    if (!chunk || chunk.length === 0) return true;
    const ratio = sampleRate / 16000;
    let pos = this.resamplePhase;
    while (pos < chunk.length - 1) {
      const p0 = pos < 0 ? (this.resamplePrevSample ?? chunk[0]) : chunk[Math.floor(pos)];
      const p1 = pos < 0 ? chunk[0] : chunk[Math.floor(pos) + 1];
      const frac = pos < 0 ? pos + 1 : pos - Math.floor(pos);
      this.buf[this.fill++] = p0 + (p1 - p0) * frac;
      if (this.fill === 1600) {
        const pcm = new ArrayBuffer(3200);
        const view = new DataView(pcm);
        for (let j = 0; j < 1600; j++) {
          const v = Math.max(-1, Math.min(1, this.buf[j]));
          view.setInt16(j * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
        }
        this.port.postMessage({ flushed: false, pcm }, [pcm]);
        this.fill = 0;
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

    function sendBinary(data) {
        if (isWsOpen()) ws.send(data);
    }

    function getWsEndpointUrl() {
        const proto = location.protocol === "https:" ? "wss://" : "ws://";
        return proto + location.host + "/ws/voice";
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

        const url = getWsEndpointUrl();
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
            sendJson({ type: "session.start" });
        };

        ws.onclose = (e) => {
            log(`WebSocket đã đóng (code=${e.code}).`, "info");
            connectionBadge.classList.remove("connected");
            connectionText.textContent = "Mất kết nối";
            teardownCallState();
            stateEl.textContent = "Mất kết nối";
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

        switch (m.type) {
            case "session.ready":
                refreshState();
                break;
            case "voice.state":
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
                }
                refreshState();
                break;
            case "transcript.partial":
                if (!m.turn_id || m.turn_id === currentTurnId) youBox.textContent = m.text || "";
                break;
            case "transcript.final":
                if (!m.turn_id || m.turn_id === currentTurnId) {
                    youBox.textContent = m.text || "";
                    botBox.innerHTML = '<span class="placeholder-text">Đang nhận phản hồi...</span>';
                }
                break;
            case "assistant.text.delta":
                if (!m.turn_id || m.turn_id === currentTurnId) {
                    if (botBox.querySelector(".placeholder-text")) botBox.textContent = "";
                    let delta = m.text || "";
                    if (!botBox.textContent) delta = delta.replace(/^[\r\n\s]+/, "");
                    if (delta) botBox.textContent += delta;
                }
                break;
            case "assistant.text.done":
                if (!m.turn_id || m.turn_id === currentTurnId) {
                    if (m.text) botBox.textContent = m.text.trim();
                }
                break;
            case "assistant.audio.chunk":
                if (!m.turn_id || m.turn_id === currentTurnId) {
                    pendingChunkMeta = m;
                    if (m.sample_rate) ttsSampleRate = m.sample_rate;
                }
                break;
            case "assistant.audio.done":
                if (!m.turn_id || m.turn_id === currentTurnId) scheduleAutoListen();
                break;
            case "playback.stop":
                clearAutoListenTimers();
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
        try {
            ttsStreamDest = audioCtx.createMediaStreamDestination();
            const pc1 = new RTCPeerConnection();
            const pc2 = new RTCPeerConnection();

            pc1.onicecandidate = (e) => e.candidate && pc2.addIceCandidate(e.candidate);
            pc2.onicecandidate = (e) => e.candidate && pc1.addIceCandidate(e.candidate);

            ttsAudioEl = document.createElement("audio");
            ttsAudioEl.autoplay = true;

            pc2.ontrack = (e) => {
                ttsAudioEl.srcObject = e.streams[0];
                ttsAudioEl.play().catch(() => { });
            };

            ttsStreamDest.stream.getTracks().forEach((track) => pc1.addTrack(track, ttsStreamDest.stream));

            const offer = await pc1.createOffer();
            await pc1.setLocalDescription(offer);
            await pc2.setRemoteDescription(offer);

            const answer = await pc2.createAnswer();
            await pc2.setLocalDescription(answer);
            await pc1.setRemoteDescription(answer);
        } catch (e) {
            console.warn("Không thể khởi tạo AEC loopback, dùng audioCtx.destination dự phòng:", e);
            ttsStreamDest = null;
        }
    }

    async function ensureAudio() {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
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
                if (pcm && (flushed || callActive)) {
                    sendBinary(pcm);
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
        if (!audioCtx || discarding || !arrayBuf || arrayBuf.byteLength < 2) return;
        const i16 = new Int16Array(arrayBuf);
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
        btnTalk.style.display = inCall ? "none" : "inline-flex";
        btnFinish.style.display = inCall ? "inline-flex" : "none";
        btnStop.style.display = inCall ? "inline-flex" : "none";
    }

    function refreshState() {
        btnFinish.disabled = (serverState !== "listening");
        if (!isWsOpen()) { stateEl.textContent = "Mất kết nối"; return; }
        if (isPlaying() || serverState === "speaking") { stateEl.textContent = "Bot đang nói..."; return; }
        if (serverState === "thinking") { stateEl.textContent = "Đang suy nghĩ..."; return; }
        if (recording || serverState === "listening") { stateEl.textContent = "Đang nghe..."; return; }
        if (serverState === "idle" && callActive && !recording) { stateEl.textContent = "Chờ nghe tiếp..."; return; }
        stateEl.textContent = STATE_LABELS[serverState] || serverState;
    }

    function clearAutoListenTimers() {
        if (waitAudioEndTimer) { clearInterval(waitAudioEndTimer); waitAudioEndTimer = null; }
        if (autoListenTimer) { clearTimeout(autoListenTimer); autoListenTimer = null; }
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
                autoListenTimer = setTimeout(() => {
                    autoListenTimer = null;
                    if (callActive && !recording && serverState === "idle") startListeningTurn();
                }, 600);
            }
        }, 100);
    }

    function teardownCallState() {
        clearAutoListenTimers();
        onWorkletFlushed = null;
        callActive = false;
        recording = false;
        resetAudioInputState();
        stopPlayback();
        releaseMicrophone();
        setCallButtons(false);
        refreshState();
    }

    function startListeningTurn() {
        resetAudioInputState();
        recording = true;
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
        clearAutoListenTimers();
        if (isBotBusy()) return setErr("Bot đang nói. Vui lòng chờ hoặc bấm Dừng nói.");
        teardownCallState();
        try { await ensureAudio(); } catch (e) { return setErr("Không thể truy cập microphone: " + e.message); }
        setErr("");
        youBox.innerHTML = '<span class="placeholder-text">Đang nghe...</span>';
        botBox.innerHTML = '<span class="placeholder-text">Đang chờ câu hỏi của bạn...</span>';
        callActive = true;
        startListeningTurn();
    };

    btnFinish.onclick = () => stopTalking();
    btnStop.onclick = () => {
        teardownCallState();
        sendJson({ type: "session.cancel", turn_id: currentTurnId });
        refreshState();
    };
    btnReset.onclick = () => {
        teardownCallState();
        youBox.innerHTML = '<span class="placeholder-text">Nội dung giọng nói của bạn sẽ hiển thị tại đây...</span>';
        botBox.innerHTML = '<span class="placeholder-text">Câu trả lời từ trợ lý sẽ hiển thị tại đây...</span>';
        setErr("");
        connect();
    };

    connect();
})();
