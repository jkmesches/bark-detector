#!/usr/bin/env python3
"""
Background bark detector.

Listens on a configurable mic input with a bandpass filter.
When the filtered RMS exceeds a threshold, it:
  1. Records the entire barking event (including a pre-roll buffer).
  2. Sends ONE MQTT message per event with amplitude, threshold, and timestamp.
  3. Groups rapid successive barks into a single event using a cooldown window.

All settings are managed via config.toml (default: ./config.toml).

Usage:
    python bark_detector.py
    python bark_detector.py --config /path/to/config.toml
"""

import argparse
import datetime
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import wave

import numpy as np
from scipy import signal as sig
import sounddevice as sd
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

try:
    import tomllib            # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib   # pip install tomli


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "audio": {
        "sample_rate": 48000,
        "chunk": 1024,
        "device_match": "ALC897 Analog",
        "device_exclude": "Alt",
        "pipewire_source": "",  # PulseAudio/PipeWire source name (enables shared access)
        "alsa": {"rec_level": 49, "mic_boost": 1},
        "gain": {"digital_db": 19},
    },
    "filter": [
        {
            "type": "bandpass",
            "low_hz": 500,
            "high_hz": 1500,
            "order": 10,
            "enabled": True,
        },
    ],
    "detection": {
        "threshold_dbfs": -14.0,
        "pre_roll_s": 1.0,
        "cooldown_s": 2.0,
        "min_event_s": 0.2,
        "max_event_s": 10.0,
    },
    "recording": {
        "dir": "recordings",
        "include_pre_roll": True,
    },
    "mqtt": {
        "broker": "localhost",
        "port": 1883,
        "username": "",
        "password": "",
        "client_id": "bark-detector",
        "qos": 1,
        "connect_timeout": 5,
        "event": {
            "topic": "/mic/dog",
            "include_recording": True,
            "extra": {},
        },
        "homeassistant": {
            "discovery": False,
            "discovery_prefix": "homeassistant",
            "object_id": "bark_detector",
            "device_name": "Bark Detector",
        },
    },
    "influxdb": {
        "enabled": False,
        "url": "http://localhost:8086",
        "token": "",
        "org": "",
        "bucket": "bark_events",
        "measurement": "bark_event",
    },
}


def _deep_merge(base, override):
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path):
    """Load config.toml and merge with defaults."""
    if not os.path.isfile(path):
        print(f"[config] {path} not found, using built-in defaults",
              file=sys.stderr)
        return DEFAULT_CONFIG

    with open(path, "rb") as f:
        user = tomllib.load(f)

    cfg = _deep_merge(DEFAULT_CONFIG, user)
    print(f"[config] Loaded {path}")
    return cfg


# ---------------------------------------------------------------------------
# ALSA helpers
# ---------------------------------------------------------------------------
def alsa_set(card, control, value):
    try:
        subprocess.check_call(
            ["amixer", "-c", str(card), "-q", "sset", control, str(value)],
            stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[alsa] failed to set {control}={value}: {e}", file=sys.stderr)


def find_alsa_card():
    """Find the ALSA card number for HDA Intel PCH / ALC897."""
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                line = line.strip()
                if "HDA Intel PCH" in line or "ALC897" in line:
                    return int(line.split()[0])
    except Exception:
        pass
    return 0


def find_input_device(match, exclude):
    """Find the sounddevice index matching the configured name."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and match in d["name"]:
            if exclude and exclude in d["name"]:
                continue
            return i
    # Fallback: first input device
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            return i
    return None


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------
class RingBuffer:
    def __init__(self, capacity):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.write_pos = 0

    def write(self, data):
        n = len(data)
        pos = self.write_pos % self.capacity
        if pos + n <= self.capacity:
            self.buf[pos:pos + n] = data
        else:
            first = self.capacity - pos
            self.buf[pos:] = data[:first]
            self.buf[:n - first] = data[first:]
        self.write_pos += n

    def read_last(self, count):
        count = min(count, self.capacity, self.write_pos)
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        end = self.write_pos % self.capacity
        if end >= count:
            return self.buf[end - count:end].copy()
        return np.concatenate([
            self.buf[self.capacity - (count - end):], self.buf[:end]])


# ---------------------------------------------------------------------------
# MQTT with connectivity test
# ---------------------------------------------------------------------------
def make_mqtt_client(cfg_mqtt, will_topic=None, will_payload=None):
    """Create, configure, and return an MQTT client (not yet connected)."""
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"{cfg_mqtt['client_id']}-{os.getpid()}")

    username = cfg_mqtt.get("username", "")
    password = cfg_mqtt.get("password", "")
    if username:
        client.username_pw_set(username, password or None)

    if will_topic and will_payload:
        client.will_set(will_topic, will_payload, qos=1, retain=True)

    return client


def test_mqtt_connection(cfg_mqtt):
    """
    Connect to the broker, publish a test message, and disconnect.
    Returns True on success, raises on failure.
    """
    broker = cfg_mqtt["broker"]
    port = cfg_mqtt["port"]
    timeout = cfg_mqtt.get("connect_timeout", 5)
    topic = cfg_mqtt["event"]["topic"]

    client = make_mqtt_client(cfg_mqtt)

    connected_event = {"ok": False, "err": None}

    def on_connect(c, userdata, flags, rc, properties=None):
        if rc == 0:
            connected_event["ok"] = True
        else:
            connected_event["err"] = mqtt.connack_string(rc)

    client.on_connect = on_connect
    client.connect_async(broker, port)
    client.loop_start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connected_event["ok"] or connected_event["err"]:
            break
        time.sleep(0.1)

    client.loop_stop()

    if connected_event["err"]:
        raise ConnectionError(
            f"MQTT broker {broker}:{port} rejected connection: "
            f"{connected_event['err']}")

    if not connected_event["ok"]:
        raise TimeoutError(
            f"MQTT broker {broker}:{port} did not respond within {timeout}s")

    # Publish a test message
    result = client.publish(
        topic,
        json.dumps({"test": True,
                     "timestamp": datetime.datetime.now(
                         datetime.timezone.utc).isoformat()}),
        qos=cfg_mqtt.get("qos", 1))
    result.wait_for_publish(timeout=timeout)

    client.disconnect()
    return True


# ---------------------------------------------------------------------------
# Bark detector
# ---------------------------------------------------------------------------
class BarkDetector:
    IDLE = 0
    BARKING = 1

    def __init__(self, cfg):
        self.cfg = cfg
        c_audio = cfg["audio"]
        c_filters = cfg["filter"]
        # Migration: old single-dict format → list
        if isinstance(c_filters, dict):
            c_filters = [c_filters]
        c_det = cfg["detection"]
        c_rec = cfg["recording"]
        c_mqtt = cfg["mqtt"]

        self.sample_rate = c_audio["sample_rate"]
        self.chunk = c_audio["chunk"]

        # Audio source — prefer PipeWire if configured
        self.pipewire_source = c_audio.get("pipewire_source", "")
        if self.pipewire_source:
            self.device_index = None
        else:
            self.device_index = find_input_device(
                c_audio.get("device_match", ""),
                c_audio.get("device_exclude", ""))
            if self.device_index is None:
                print("No input device found!", file=sys.stderr)
                sys.exit(1)

        # Build filter chain (stack SOS sections from each enabled filter)
        nyq = self.sample_rate / 2
        sos_list = []
        self._filter_descs = []
        for fc in c_filters:
            if not fc.get("enabled", True):
                continue
            ftype = fc["type"].lower()
            order = fc.get("order", 4)
            if ftype == "bandpass":
                sos_list.append(sig.butter(
                    order, [fc["low_hz"] / nyq, fc["high_hz"] / nyq],
                    btype="band", output="sos"))
                self._filter_descs.append(
                    f"BP {fc['low_hz']}-{fc['high_hz']} Hz ord {order}")
            elif ftype == "lowpass":
                sos_list.append(sig.butter(
                    order, fc["high_hz"] / nyq, btype="low", output="sos"))
                self._filter_descs.append(
                    f"LP <{fc['high_hz']} Hz ord {order}")
            elif ftype == "highpass":
                sos_list.append(sig.butter(
                    order, fc["low_hz"] / nyq, btype="high", output="sos"))
                self._filter_descs.append(
                    f"HP >{fc['low_hz']} Hz ord {order}")
            elif ftype == "notch":
                center = fc.get("center_hz", 1000)
                q = fc.get("q_factor", 30.0)
                b, a = sig.iirnotch(center / nyq, q)
                sos_list.append(sig.tf2sos(b, a))
                self._filter_descs.append(f"Notch {center} Hz Q={q}")
            else:
                print(f"[filter] Unknown type '{ftype}', skipping",
                      file=sys.stderr)
        if sos_list:
            self.sos = np.vstack(sos_list)
        else:
            # Fallback: passthrough (no filtering)
            self.sos = np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float64)
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

        # Gain
        self.gain = 10 ** (c_audio["gain"]["digital_db"] / 20.0)

        # Detection
        self.threshold_dbfs = c_det["threshold_dbfs"]
        self.pre_roll_samples = int(c_det["pre_roll_s"] * self.sample_rate)
        self.cooldown_s = c_det["cooldown_s"]
        self.min_event_s = c_det["min_event_s"]
        self.max_event_s = c_det["max_event_s"]

        # Ring buffers: pre-roll + up to 60s
        ring_cap = int(self.sample_rate * max(c_det["pre_roll_s"] + 60, 10))
        self.ring = RingBuffer(ring_cap)
        self.rec_ring = RingBuffer(ring_cap)

        # State
        self.state = self.IDLE
        self.event_start_write_pos = 0
        self.event_peak_rms_db = -120.0
        self.event_bark_count = 0
        self.last_bark_time = 0.0
        self.event_start_time = None

        # Recording
        rec_dir = c_rec["dir"]
        if not os.path.isabs(rec_dir):
            rec_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), rec_dir)
        self.recording_dir = rec_dir
        self.include_pre_roll = c_rec.get("include_pre_roll", True)
        os.makedirs(self.recording_dir, exist_ok=True)

        # MQTT
        self.mqtt_topic = c_mqtt["event"]["topic"]
        self.mqtt_qos = c_mqtt.get("qos", 1)
        self.mqtt_include_recording = c_mqtt["event"].get(
            "include_recording", True)
        self.mqtt_extra = c_mqtt["event"].get("extra", {})
        self.mqtt_cfg = c_mqtt
        self.mqtt_client = None

        # Home Assistant MQTT Discovery
        c_ha = c_mqtt.get("homeassistant", {})
        self.ha_discovery = c_ha.get("discovery", False)
        self.ha_prefix = c_ha.get("discovery_prefix", "homeassistant")
        self.ha_object_id = c_ha.get("object_id", "bark_detector")
        self.ha_device_name = c_ha.get("device_name", "Bark Detector")
        self.ha_state_topic = f"{self.ha_prefix}/sensor/{self.ha_object_id}/state"
        self._daily_bark_total = 0
        self._daily_event_total = 0
        self._daily_reset_date = datetime.date.today()

        # InfluxDB
        c_influx = cfg.get("influxdb", {})
        self.influx_enabled = c_influx.get("enabled", False)
        self.influx_write_api = None
        self.influx_client = None
        self.influx_bucket = c_influx.get("bucket", "bark_events")
        self.influx_org = c_influx.get("org", "")
        self.influx_measurement = c_influx.get("measurement", "bark_event")
        if self.influx_enabled:
            self._setup_influxdb(c_influx)

        # ALSA
        self._setup_alsa(c_audio["alsa"])

        self.overflow_count = 0
        self._audio_q = queue.Queue()
        self._stop_event = threading.Event()

    def _setup_alsa(self, alsa_cfg):
        card = find_alsa_card()
        print(f"[alsa] Using card {card}")
        alsa_set(card, "Capture", alsa_cfg["rec_level"])
        alsa_set(card, "Rear Mic Boost", alsa_cfg["mic_boost"])
        print(f"[alsa] Capture={alsa_cfg['rec_level']}, "
              f"Mic Boost={alsa_cfg['mic_boost']}")

    def _setup_influxdb(self, c_influx):
        try:
            self.influx_client = InfluxDBClient(
                url=c_influx["url"],
                token=c_influx["token"],
                org=c_influx["org"])
            # Test connectivity
            health = self.influx_client.health()
            if health.status != "pass":
                raise ConnectionError(f"InfluxDB health: {health.status}")
            self.influx_write_api = self.influx_client.write_api(
                write_options=SYNCHRONOUS)
            print(f"[influx] Connected to {c_influx['url']}, "
                  f"bucket={self.influx_bucket}")
        except Exception as e:
            print(f"[influx] Setup failed: {e}", file=sys.stderr)
            self.influx_write_api = None

    def _write_influxdb(self, duration, filename):
        if not self.influx_write_api:
            return
        try:
            point = (
                Point(self.influx_measurement)
                .tag("source", "bark_detector")
                .field("peak_rms_dbfs", float(self.event_peak_rms_db))
                .field("threshold_dbfs", float(self.threshold_dbfs))
                .field("bark_count", int(self.event_bark_count))
                .field("duration_s", float(round(duration, 1)))
                .field("recording", filename)
                .time(self.event_start_time, WritePrecision.S)
            )
            # Merge extra tags from mqtt.event.extra (reuse for influx too)
            for k, v in self.mqtt_extra.items():
                point = point.tag(k, str(v))

            self.influx_write_api.write(
                bucket=self.influx_bucket, org=self.influx_org, record=point)
            print(f"[influx] Wrote point to {self.influx_bucket}")
        except Exception as e:
            print(f"[influx] Write failed: {e}", file=sys.stderr)

    # ---- Home Assistant MQTT Discovery ------------------------------------
    def _ha_device_block(self):
        """Shared device info for all HA discovery configs."""
        return {
            "identifiers": [self.ha_object_id],
            "name": self.ha_device_name,
            "manufacturer": "OutdoorMic",
            "model": "Bark Detector",
            "sw_version": "1.0",
        }

    def _publish_ha_discovery(self, client):
        """Publish MQTT Discovery config messages so HA auto-creates sensors."""
        if not self.ha_discovery:
            return

        device = self._ha_device_block()
        base = f"{self.ha_prefix}/sensor/{self.ha_object_id}"
        state = self.ha_state_topic

        sensors = [
            {
                "key": "last_event",
                "name": "Last Bark Event",
                "device_class": "timestamp",
                "value_template": "{{ value_json.last_event }}",
                "icon": "mdi:dog",
            },
            {
                "key": "peak_rms",
                "name": "Peak RMS",
                "unit_of_measurement": "dBFS",
                "state_class": "measurement",
                "value_template": "{{ value_json.peak_rms_dbfs }}",
                "icon": "mdi:volume-high",
            },
            {
                "key": "bark_count",
                "name": "Barks in Last Event",
                "state_class": "measurement",
                "value_template": "{{ value_json.bark_count }}",
                "icon": "mdi:dog-side",
            },
            {
                "key": "duration",
                "name": "Last Event Duration",
                "unit_of_measurement": "s",
                "state_class": "measurement",
                "value_template": "{{ value_json.duration_s }}",
                "icon": "mdi:timer-outline",
            },
            {
                "key": "threshold",
                "name": "Threshold",
                "unit_of_measurement": "dBFS",
                "state_class": "measurement",
                "value_template": "{{ value_json.threshold_dbfs }}",
                "icon": "mdi:tune-vertical",
                "entity_category": "diagnostic",
            },
            {
                "key": "daily_barks",
                "name": "Daily Bark Total",
                "state_class": "total",
                "value_template": "{{ value_json.daily_barks }}",
                "icon": "mdi:counter",
            },
            {
                "key": "daily_events",
                "name": "Daily Event Total",
                "state_class": "total",
                "value_template": "{{ value_json.daily_events }}",
                "icon": "mdi:calendar-today",
            },
            {
                "key": "status",
                "name": "Status",
                "value_template": "{{ value_json.status }}",
                "icon": "mdi:microphone",
                "entity_category": "diagnostic",
            },
        ]

        for s in sensors:
            config_topic = f"{base}_{s['key']}/config"
            payload = {
                "name": s["name"],
                "unique_id": f"{self.ha_object_id}_{s['key']}",
                "state_topic": state,
                "value_template": s["value_template"],
                "device": device,
            }
            if "unit_of_measurement" in s:
                payload["unit_of_measurement"] = s["unit_of_measurement"]
            if "device_class" in s:
                payload["device_class"] = s["device_class"]
            if "icon" in s:
                payload["icon"] = s["icon"]
            if "entity_category" in s:
                payload["entity_category"] = s["entity_category"]
            if "state_class" in s:
                payload["state_class"] = s["state_class"]

            client.publish(
                config_topic, json.dumps(payload), qos=1, retain=True)
            print(f"[ha] Published discovery: {config_topic}")

        # Publish initial state
        self._publish_ha_state(client, init=True)

    def _publish_ha_state(self, client=None, init=False, **event_data):
        """Publish combined state JSON to the shared state topic."""
        c = client or self.mqtt_client
        if c is None:
            return

        # Reset daily counters at midnight
        today = datetime.date.today()
        if today != self._daily_reset_date:
            self._daily_bark_total = 0
            self._daily_event_total = 0
            self._daily_reset_date = today

        if init:
            state = {
                "last_event": "",
                "peak_rms_dbfs": 0,
                "bark_count": 0,
                "duration_s": 0,
                "threshold_dbfs": float(self.threshold_dbfs),
                "daily_barks": self._daily_bark_total,
                "daily_events": self._daily_event_total,
                "status": "listening",
            }
        else:
            state = {
                "last_event": event_data.get("timestamp", ""),
                "peak_rms_dbfs": event_data.get("peak_rms_dbfs", 0),
                "bark_count": event_data.get("bark_count", 0),
                "duration_s": event_data.get("duration_s", 0),
                "threshold_dbfs": float(self.threshold_dbfs),
                "daily_barks": self._daily_bark_total,
                "daily_events": self._daily_event_total,
                "status": "listening",
            }

        c.publish(self.ha_state_topic, json.dumps(state), qos=1, retain=True)

    def _offline_state_payload(self):
        """JSON payload indicating the detector is offline."""
        return json.dumps({
            "last_event": "",
            "peak_rms_dbfs": 0,
            "bark_count": 0,
            "duration_s": 0,
            "threshold_dbfs": float(self.threshold_dbfs),
            "daily_barks": self._daily_bark_total,
            "daily_events": self._daily_event_total,
            "status": "offline",
        })

    def _connect_mqtt(self):
        try:
            will_topic = self.ha_state_topic if self.ha_discovery else None
            will_payload = self._offline_state_payload() if self.ha_discovery else None
            self.mqtt_client = make_mqtt_client(
                self.mqtt_cfg, will_topic=will_topic, will_payload=will_payload)
            self.mqtt_client.connect(
                self.mqtt_cfg["broker"], self.mqtt_cfg["port"])
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f"[mqtt] Connect failed: {e}", file=sys.stderr)
            self.mqtt_client = None
            return False

    def _input_callback(self, indata, frames, time_info, status):
        """Realtime audio callback — only copies data, no processing."""
        if status and status.input_overflow:
            self.overflow_count += 1
        self._audio_q.put_nowait(indata[:, 0].copy())

    def _process_loop(self):
        """Worker thread: pulls chunks from the queue and does all processing."""
        while not self._stop_event.is_set():
            try:
                mono = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.gain != 1.0:
                mono *= self.gain
            np.clip(mono, -1.0, 1.0, out=mono)

            self.ring.write(mono)

            filtered, self.zi = sig.sosfilt(self.sos, mono, zi=self.zi)
            filtered = np.clip(filtered, -1.0, 1.0).astype(np.float32)
            self.rec_ring.write(filtered)

            rms = np.sqrt(np.mean(filtered ** 2))
            rms_db = 20 * np.log10(rms + 1e-12)
            now = time.monotonic()

            if self.state == self.IDLE:
                if rms_db >= self.threshold_dbfs:
                    self.state = self.BARKING
                    self.event_start_write_pos = (
                        self.ring.write_pos - len(mono) - self.pre_roll_samples)
                    self.event_start_time = datetime.datetime.now(
                        datetime.timezone.utc)
                    self.event_peak_rms_db = rms_db
                    self.event_bark_count = 1
                    self.last_bark_time = now
                    print(f"[bark] Event started at "
                          f"{self.event_start_time.isoformat()} "
                          f"RMS={rms_db:.1f} dBFS")

            elif self.state == self.BARKING:
                elapsed = (datetime.datetime.now(datetime.timezone.utc)
                           - self.event_start_time).total_seconds()
                if rms_db >= self.threshold_dbfs:
                    self.event_peak_rms_db = max(self.event_peak_rms_db, rms_db)
                    self.event_bark_count += 1
                    self.last_bark_time = now
                if elapsed >= self.max_event_s:
                    # Force-finalize long event and start a new one
                    self._finalize_event()
                    if rms_db >= self.threshold_dbfs:
                        self.state = self.BARKING
                        self.event_start_write_pos = (
                            self.ring.write_pos - len(mono))
                        self.event_start_time = datetime.datetime.now(
                            datetime.timezone.utc)
                        self.event_peak_rms_db = rms_db
                        self.event_bark_count = 1
                        self.last_bark_time = now
                elif now - self.last_bark_time >= self.cooldown_s:
                    self._finalize_event()

    def _finalize_event(self):
        event_end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = (event_end_time - self.event_start_time).total_seconds()

        if duration < self.min_event_s:
            print(f"[bark] Event too short ({duration:.2f}s), discarding")
            self.state = self.IDLE
            return

        print(f"[bark] Event ended: {self.event_bark_count} barks, "
              f"peak={self.event_peak_rms_db:.1f} dBFS, "
              f"duration={duration:.1f}s")

        filename = self._save_recording()
        self._send_mqtt(duration, filename)
        if self.influx_enabled:
            self._write_influxdb(duration, filename)
        self.state = self.IDLE

    def _save_recording(self):
        duration_samples = self.ring.write_pos - max(
            self.event_start_write_pos,
            self.ring.write_pos - self.ring.capacity)
        duration_samples = max(0, min(duration_samples, self.ring.capacity))

        audio = self.ring.read_last(duration_samples)

        ts = self.event_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"bark_{ts}.wav"
        filepath = os.path.join(self.recording_dir, filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        print(f"[rec] Saved {filepath} ({len(audio)/self.sample_rate:.1f}s)")
        return filename

    def _send_mqtt(self, duration, filename):
        payload = {
            "timestamp": self.event_start_time.isoformat(),
            "peak_rms_dbfs": float(round(self.event_peak_rms_db, 1)),
            "threshold_dbfs": float(self.threshold_dbfs),
            "bark_count": int(self.event_bark_count),
            "duration_s": float(round(duration, 1)),
        }
        if self.mqtt_include_recording:
            payload["recording"] = filename
        # Merge any extra fields from config
        payload.update(self.mqtt_extra)

        msg = json.dumps(payload)

        if self.mqtt_client is None:
            if not self._connect_mqtt():
                print(f"[mqtt] WOULD SEND on {self.mqtt_topic}: {msg}")
                return

        result = self.mqtt_client.publish(
            self.mqtt_topic, msg, qos=self.mqtt_qos)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[mqtt] Published to {self.mqtt_topic}: {msg}")
        else:
            print(f"[mqtt] Publish failed (rc={result.rc}): {msg}",
                  file=sys.stderr)

        # Update HA sensors
        if self.ha_discovery:
            self._daily_bark_total += int(self.event_bark_count)
            self._daily_event_total += 1
            self._publish_ha_state(
                timestamp=self.event_start_time.isoformat(),
                peak_rms_dbfs=float(round(self.event_peak_rms_db, 1)),
                bark_count=int(self.event_bark_count),
                duration_s=float(round(duration, 1)),
            )

    def _parec_reader(self):
        """Read raw PCM from parec and feed chunks into _audio_q."""
        nbytes = self.chunk * 2  # 16-bit = 2 bytes per sample
        # Discard first few chunks (parec startup noise)
        for _ in range(3):
            if self._stop_event.is_set():
                return
            self._parec_proc.stdout.read(nbytes)
        while not self._stop_event.is_set():
            try:
                data = self._parec_proc.stdout.read(nbytes)
                if not data:
                    break
                mono = np.frombuffer(
                    data, dtype=np.int16).astype(np.float32) / 32768.0
                self._audio_q.put_nowait(mono)
            except Exception:
                if self._stop_event.is_set():
                    break

    def run(self):
        if self.pipewire_source:
            print(f"[bark] PipeWire source: {self.pipewire_source}")
        else:
            dev_name = sd.query_devices(self.device_index)["name"]
            print(f"[bark] Device: [{self.device_index}] {dev_name}")
        if self._filter_descs:
            print(f"[bark] Filters: {' → '.join(self._filter_descs)}")
        else:
            print("[bark] Filters: (none / passthrough)")
        print(f"[bark] Threshold: {self.threshold_dbfs} dBFS, "
              f"cooldown: {self.cooldown_s}s, "
              f"max-event: {self.max_event_s}s, "
              f"pre-roll: {self.pre_roll_samples/self.sample_rate:.1f}s")
        print(f"[bark] Recordings: {self.recording_dir}")
        print(f"[bark] MQTT topic: {self.mqtt_topic}")

        # Connect MQTT early so we can publish HA discovery before listening
        if self._connect_mqtt() and self.ha_discovery:
            self._publish_ha_discovery(self.mqtt_client)
            print(f"[ha] Discovery published, state topic: {self.ha_state_topic}")
        print()

        worker = threading.Thread(target=self._process_loop, daemon=True)
        worker.start()

        if self.pipewire_source:
            # PipeWire backend via parec subprocess
            self._parec_proc = subprocess.Popen(
                ["parec", "-d", self.pipewire_source,
                 f"--rate={self.sample_rate}", "--channels=1", "--format=s16le",
                 "--latency-msec=20"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            reader = threading.Thread(target=self._parec_reader, daemon=True)
            reader.start()
            try:
                print("[bark] Detector running (PipeWire). Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
                    ov = self.overflow_count
                    if ov > 0:
                        self.overflow_count = 0
                        print(f"[warn] {ov} buffer overflows",
                              file=sys.stderr)
            except KeyboardInterrupt:
                print("\n[bark] Stopping...")
            self._parec_proc.terminate()
            try:
                self._parec_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._parec_proc.kill()
        else:
            # sounddevice/ALSA backend (exclusive device access)
            stream = sd.InputStream(
                device=self.device_index, channels=1,
                samplerate=self.sample_rate, blocksize=self.chunk,
                dtype="float32", callback=self._input_callback, latency=0.5)
            try:
                with stream:
                    print("[bark] Detector running. Press Ctrl+C to stop.")
                    while True:
                        time.sleep(1)
                        ov = self.overflow_count
                        if ov > 0:
                            self.overflow_count = 0
                            print(f"[warn] {ov} buffer overflows",
                                  file=sys.stderr)
            except KeyboardInterrupt:
                print("\n[bark] Stopping...")

        self._stop_event.set()
        worker.join(timeout=2)
        if self.state == self.BARKING:
            self._finalize_event()

        if self.mqtt_client:
            if self.ha_discovery:
                self.mqtt_client.publish(
                    self.ha_state_topic, self._offline_state_payload(),
                    qos=1, retain=True)
                print("[ha] Published offline status")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        if self.influx_client:
            self.influx_client.close()

        print("[bark] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
class _Tee(io.TextIOBase):
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def main():
    p = argparse.ArgumentParser(
        description="Background bark detector with MQTT notifications")
    p.add_argument(
        "-c", "--config", default="config.toml",
        help="Path to config.toml (default: ./config.toml)")
    args = p.parse_args()

    # Resolve config path relative to script dir if not absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path)

    # Set up log file tee — all print output goes to stdout AND log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "bark_detector.log")
    log_file = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    cfg = load_config(config_path)

    # --- Test MQTT connectivity ---
    print(f"[mqtt] Testing connection to "
          f"{cfg['mqtt']['broker']}:{cfg['mqtt']['port']} ...")
    try:
        test_mqtt_connection(cfg["mqtt"])
        print(f"[mqtt] OK — broker reachable, test message published to "
              f"{cfg['mqtt']['event']['topic']}")
    except (ConnectionError, TimeoutError, OSError) as e:
        print(f"[mqtt] FAILED — {e}", file=sys.stderr)
        print("[mqtt] The detector will still run and retry on each event.",
              file=sys.stderr)

    # --- Test InfluxDB connectivity ---
    c_influx = cfg.get("influxdb", {})
    if c_influx.get("enabled", False):
        print(f"[influx] Testing connection to {c_influx['url']} ...")
        try:
            client = InfluxDBClient(
                url=c_influx["url"],
                token=c_influx["token"],
                org=c_influx["org"])
            health = client.health()
            if health.status != "pass":
                raise ConnectionError(f"health status: {health.status}")
            # Verify bucket exists
            buckets_api = client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(c_influx["bucket"])
            if bucket is None:
                raise ConnectionError(
                    f"bucket '{c_influx['bucket']}' not found")
            client.close()
            print(f"[influx] OK — bucket '{c_influx['bucket']}' ready")
        except Exception as e:
            print(f"[influx] FAILED — {e}", file=sys.stderr)
            print("[influx] The detector will still run without InfluxDB.",
                  file=sys.stderr)
    else:
        print("[influx] Disabled in config")

    detector = BarkDetector(cfg)
    detector.run()


if __name__ == "__main__":
    main()
