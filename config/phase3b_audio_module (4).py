#!/usr/bin/env python3
"""
PHASE 3B - Real-time Microphone Audio Input Module (sounddevice version - FIXED)
Captures live English audio from microphone for speech translation pipeline
Jetson Nano compatible - uses sounddevice instead of PyAudio
"""

import sounddevice as sd
import numpy as np
import time
from collections import deque
from threading import Thread, Event
from pathlib import Path

class MicrophoneAudioCapture:
    """
    Real-time microphone audio capture with:
    - Configurable sample rate (16kHz for Whisper optimal)
    - Audio buffering for continuous stream
    - Simple Voice Activity Detection (VAD)
    - Thread-safe audio queue
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 chunk_size=1024,
                 channels=1,
                 silence_threshold=0.02,
                 silence_duration_sec=2.0):
        """
        Initialize microphone capture
        
        Args:
            sample_rate: Audio sample rate in Hz (16000 optimal for Whisper)
            chunk_size: Samples per audio chunk
            channels: Number of audio channels (1 = mono)
            silence_threshold: Energy threshold for silence detection
            silence_duration_sec: Seconds of silence before stopping recording
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration_sec = silence_duration_sec
        
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=int(sample_rate * 10))  # 10 sec buffer
        self.recording_buffer = []
        self.capture_thread = None
        self.stop_event = Event()
        
        self.bytes_recorded = 0
        self.record_time = 0
        
    def _calculate_energy(self, audio_chunk):
        """Calculate RMS energy of audio chunk for VAD"""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def _capture_audio_stream(self):
        """Background thread for continuous audio capture"""
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                               channels=self.channels,
                               blocksize=self.chunk_size,
                               dtype='float32') as stream:
                while not self.stop_event.is_set():
                    try:
                        audio_chunk = stream.read(self.chunk_size)[0]
                        self.audio_buffer.append(audio_chunk.flatten())
                        self.bytes_recorded += len(audio_chunk) * 4  # 4 bytes per float32
                    except Exception as e:
                        print(f"Capture error: {e}")
                        break
        except Exception as e:
            print(f"Stream error: {e}")
    
    def start_recording(self):
        """Start microphone recording in background thread"""
        try:
            self.is_recording = True
            self.recording_buffer = []
            self.bytes_recorded = 0
            self.stop_event.clear()
            
            self.capture_thread = Thread(target=self._capture_audio_stream, daemon=True)
            self.capture_thread.start()
            
            return True, f"Recording started (sample_rate={self.sample_rate}Hz)"
        except Exception as e:
            return False, f"Failed to start recording: {str(e)[:100]}"
    
    def record_until_silence(self, timeout_sec=30):
        """
        Record audio until silence is detected
        
        Returns:
            (success, audio_array, duration_sec)
        """
        if not self.is_recording:
            return False, np.array([]), 0
        
        start_time = time.time()
        silent_chunks = 0
        max_silent_chunks = int((self.silence_duration_sec * self.sample_rate) / self.chunk_size)
        recording_started = False
        
        print(f"[MIC] Listening for speech (timeout: {timeout_sec}s, silence threshold: {self.silence_threshold})...")
        
        while time.time() - start_time < timeout_sec:
            if len(self.audio_buffer) > 0:
                chunk = self.audio_buffer[0]
                energy = self._calculate_energy(chunk)
                
                if energy > self.silence_threshold:
                    # Speech detected
                    recording_started = True
                    silent_chunks = 0
                    self.recording_buffer.append(chunk.copy())
                    self.audio_buffer.popleft()
                    print(f"[MIC] Speech detected (energy={energy:.4f})")
                elif recording_started:
                    # Recording speech, check for silence
                    self.recording_buffer.append(chunk.copy())
                    self.audio_buffer.popleft()
                    silent_chunks += 1
                    
                    if silent_chunks >= max_silent_chunks:
                        # Silence detected after speech
                        print(f"[MIC] End of speech detected (silence={silent_chunks} chunks)")
                        break
                else:
                    # No speech yet, clear buffer
                    self.audio_buffer.popleft()
            
            time.sleep(0.01)  # Small delay to avoid busy loop
        
        if not self.recording_buffer:
            return False, np.array([]), 0
        
        audio_array = np.concatenate(self.recording_buffer)
        duration = len(audio_array) / self.sample_rate
        self.record_time = duration
        
        return True, audio_array, duration
    
    def stop_recording(self):
        """Stop microphone recording"""
        self.stop_event.set()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        self.is_recording = False
        return True, "Recording stopped"
    
    def save_audio(self, audio_array, filename):
        """Save audio array to WAV file"""
        try:
            import soundfile as sf
            sf.write(filename, audio_array, self.sample_rate)
            return True, f"Saved to {filename}"
        except ImportError:
            print("[WARNING] soundfile not available, trying scipy")
            try:
                from scipy.io import wavfile
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wavfile.write(filename, self.sample_rate, audio_int16)
                return True, f"Saved to {filename}"
            except Exception as e:
                return False, f"Failed to save: {str(e)[:100]}"
    
    def get_device_info(self):
        """List available audio input devices"""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, info in enumerate(device_list):
                if info['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['max_input_channels'],
                        'rate': info['default_samplerate']
                    })
        except Exception as e:
            print(f"Error querying devices: {e}")
        return devices
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.is_recording:
            self.stop_recording()
