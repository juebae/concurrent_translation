# concurrent_translation
Concurrent, offline speech-to-speech translation pipeline for the NVIDIA Jetson Nano (4GB): microphone capture → Whisper ASR → Opus-MT EN→ES → quality estimation (mBERT/COMET options) → eSpeak-ng TTS, with profiling, noise suppression, and optimisation experiments to benchmark latency/quality under real edge constraints.
