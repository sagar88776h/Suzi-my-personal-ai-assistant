"""A stand-in for the ``sounddevice`` module so audio code runs without hardware.

Install it with ``install_fake_sounddevice(monkeypatch, ...)``. The fake
microphone feeds a prepared signal block by block through the real callback;
the fake speaker pulls audio through the callback and records what it played.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np


class CallbackStop(Exception):
    pass


class FakeSoundDevice(types.ModuleType):
    def __init__(self, mic_signal: np.ndarray | None = None, realtime: float = 0.0,
                 input_device: str | None = "Fake Mic", output_device: str | None = "Fake Speakers",
                 fail_open_input: bool = False, fail_open_output: bool = False) -> None:
        super().__init__("sounddevice")
        self.CallbackStop = CallbackStop
        self.mic_signal = mic_signal if mic_signal is not None else np.zeros(0, dtype=np.float32)
        self.realtime = realtime  # seconds of sleep per block (0 = as fast as possible)
        self.input_device = input_device
        self.output_device = output_device
        self.fail_open_input = fail_open_input
        self.fail_open_output = fail_open_output
        self.played: list[np.ndarray] = []
        fake = self

        class InputStream:
            def __init__(self, samplerate, channels, dtype, blocksize, device, callback):
                if fake.fail_open_input:
                    raise RuntimeError("Error opening InputStream: no device")
                self.blocksize, self.callback = blocksize, callback
                self._stop = threading.Event()

            def _run(self):
                signal = fake.mic_signal
                position = 0
                while not self._stop.is_set():
                    block = signal[position:position + self.blocksize]
                    if len(block) < self.blocksize:  # after the signal ends: silence
                        block = np.concatenate([block, np.zeros(self.blocksize - len(block), np.float32)])
                    position += self.blocksize
                    self.callback(block.reshape(-1, 1).astype(np.float32), self.blocksize, None, None)
                    time.sleep(fake.realtime)

            def __enter__(self):
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
                return self

            def __exit__(self, *exc):
                self._stop.set()
                self._thread.join(timeout=2)
                return False

        class OutputStream:
            def __init__(self, samplerate, channels, dtype, device, callback, finished_callback):
                if fake.fail_open_output:
                    raise RuntimeError("Error opening OutputStream: no device")
                self.callback, self.finished_callback = callback, finished_callback
                self._closed = threading.Event()
                self.output: list[np.ndarray] = []

            def _run(self):
                try:
                    while not self._closed.is_set():
                        out = np.zeros((256, 1), dtype=np.float32)
                        try:
                            self.callback(out, 256, None, None)
                        except CallbackStop:
                            self.output.append(out[:, 0].copy())
                            break
                        self.output.append(out[:, 0].copy())
                        time.sleep(fake.realtime)
                finally:
                    self.finished_callback()

            def __enter__(self):
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
                return self

            def __exit__(self, *exc):
                self._closed.set()
                self._thread.join(timeout=2)
                fake.played.append(np.concatenate(self.output) if self.output else np.zeros(0, np.float32))
                return False

        self.InputStream = InputStream
        self.OutputStream = OutputStream

    def query_devices(self, kind=None):
        if kind is None:  # full device list, like sounddevice.query_devices()
            devices = []
            if self.input_device is not None:
                devices.append({"name": self.input_device, "max_input_channels": 1, "max_output_channels": 0})
            devices.append({"name": "USB Headset Microphone", "max_input_channels": 1, "max_output_channels": 0})
            devices.append({"name": self.output_device or "Speakers", "max_input_channels": 0,
                            "max_output_channels": 2})
            return devices
        name = self.input_device if kind == "input" else self.output_device
        if name is None:
            raise RuntimeError(f"no default {kind} device")
        return {"name": name}

    @property
    def default(self):
        return types.SimpleNamespace(device=(0 if self.input_device is not None else -1, 2))


def install_fake_sounddevice(monkeypatch, **kwargs) -> FakeSoundDevice:
    fake = FakeSoundDevice(**kwargs)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def speech_signal(silence_before: float = 0.5, speech: float = 1.0, silence_after: float = 2.0,
                  rate: int = 16_000, loudness: float = 0.2) -> np.ndarray:
    """Background hiss, then a loud tone ("speech"), then hiss again."""
    rng = np.random.default_rng(0)

    def noise(seconds: float) -> np.ndarray:
        return (rng.standard_normal(int(rate * seconds)) * 0.002).astype(np.float32)

    t = np.arange(int(rate * speech)) / rate
    voice = (np.sin(2 * np.pi * 220 * t) * loudness).astype(np.float32)
    return np.concatenate([noise(silence_before), voice, noise(silence_after)])
