"""
Sign Language Gesture Recognition - Text to Speech Engine
Provides non-blocking, thread-safe asynchronous speech output.
Uses native macOS 'say' command on Darwin with fallback to pyttsx3.
"""

import sys
import os
import queue
import threading
import subprocess


class SpeechEngine:
    def __init__(self, voice=None, rate=180):
        self.is_mac = sys.platform == "darwin"
        self.voice = voice
        self.rate = rate
        self.queue = queue.Queue()
        self.current_process = None
        self._running = True
        self._lock = threading.Lock()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        while self._running:
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not text or not text.strip():
                self.queue.task_done()
                continue

            clean_text = text.strip()

            if self.is_mac:
                # Use native macOS 'say' - fast, natural, zero external dependencies
                cmd = ["say"]
                if self.voice:
                    cmd.extend(["-v", self.voice])
                if self.rate:
                    cmd.extend(["-r", str(self.rate)])
                cmd.append(clean_text)

                with self._lock:
                    try:
                        self.current_process = subprocess.Popen(
                            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                    except Exception as e:
                        print(f"[TTS Error] Failed to execute say: {e}")
                        self.current_process = None

                if self.current_process:
                    self.current_process.wait()
                    with self._lock:
                        self.current_process = None
            else:
                # Fallback to pyttsx3 for Windows/Linux
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.rate)
                    engine.say(clean_text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS Error] pyttsx3 failed: {e}")

            self.queue.task_done()

    def speak(self, text):
        """
        Enqueues text to be spoken asynchronously.
        Does not block the calling thread or video frame rate.
        """
        if text and text.strip():
            self.queue.put(text.strip())

    def stop(self):
        """Stops any current ongoing speech."""
        with self._lock:
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                except Exception:
                    pass
                self.current_process = None

        # Clear remaining queued items
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

    def is_busy(self):
        with self._lock:
            return (self.current_process is not None and self.current_process.poll() is None) or not self.queue.empty()


# Global singleton instance for easy import
tts = SpeechEngine()

if __name__ == "__main__":
    print("Testing Text to Speech engine...")
    tts.speak("Hello! The hand gesture speech system is ready.")
    import time
    time.sleep(2.5)
    print("Speech test completed.")
