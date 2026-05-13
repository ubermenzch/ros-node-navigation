#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record the Unitree Go2 front camera UDP H.264 stream directly to a video file.

This node intentionally does not decode or re-encode video. It mirrors the
working test264.py pipeline: RTP H.264 -> depay -> parse -> mux -> filesink.
On SIGINT/SIGTERM it sends EOS so MP4/MKV files are finalized cleanly.
"""

from datetime import datetime
import os
import signal
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from utils.config_loader import get_config
from utils.logger import NodeLogger


class FrontVideoRecorderNode:
    """GStreamer based front camera recorder for multi_main.py."""

    VALID_CONTAINERS = ("mp4", "mkv")

    def __init__(self, log_dir: Optional[str] = None, log_timestamp: Optional[str] = None):
        self.log_dir = log_dir
        self.log_timestamp = log_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        config = get_config()
        node_config = config.get("front_video_recorder_node", {})

        self.iface = str(node_config.get("iface", "eth0")).strip()
        self.address = str(node_config.get("address", "230.1.1.1")).strip()
        self.port = int(node_config.get("port", 1720))
        self.container = str(node_config.get("container", "mp4")).strip().lower()
        if self.container not in self.VALID_CONTAINERS:
            self.container = "mp4"

        self.latency_ms = int(node_config.get("latency_ms", 100))
        self.eos_timeout_sec = float(node_config.get("eos_timeout_sec", 8.0))
        self.output_dir = str(node_config.get("output_dir", "videos")).strip()
        self.output_path = str(node_config.get("output_path", "")).strip()
        self.output_filename = str(node_config.get("output_filename", "")).strip()
        self.filename_prefix = str(node_config.get("filename_prefix", "go2_front")).strip() or "go2_front"
        self.overwrite_existing = self._get_bool(node_config, "overwrite_existing", False)

        self.output = self._resolve_output_path()
        self._ensure_output_dir(self.output)

        self.logger = NodeLogger(
            node_name="front_video_recorder_node",
            log_dir=self.log_dir,
            log_timestamp=self.log_timestamp,
            enabled=self._get_bool(node_config, "log_enabled", True),
        )

        self.loop = GLib.MainLoop()
        self.pipeline = None
        self.bus = None
        self.eos_sent = False
        self.eos_received = False
        self._signal_source_ids = []

    @staticmethod
    def _get_bool(config: dict, key: str, default: bool) -> bool:
        value = config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        return bool(value)

    @staticmethod
    def _project_root() -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self._project_root(), path)

    def _format_output_filename(self, filename: str) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        values = {
            "timestamp": now,
            "session_timestamp": self.log_timestamp,
            "container": self.container,
        }
        try:
            formatted = filename.format(**values)
        except (KeyError, ValueError):
            formatted = filename

        root, ext = os.path.splitext(formatted)
        if not ext:
            formatted = f"{root}.{self.container}"
        return formatted

    def _dedupe_path(self, path: str) -> str:
        if self.overwrite_existing or not os.path.exists(path):
            return path

        root, ext = os.path.splitext(path)
        for index in range(1, 1000):
            candidate = f"{root}_{index:03d}{ext}"
            if not os.path.exists(candidate):
                return candidate
        return path

    @staticmethod
    def _ensure_output_dir(path: str) -> None:
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)

    def _resolve_output_path(self) -> str:
        if self.output_path:
            return self._dedupe_path(self._resolve_path(self.output_path))

        output_dir = self._resolve_path(self.output_dir or "videos")
        if self.output_filename:
            filename = self._format_output_filename(self.output_filename)
        else:
            filename = f"{self.filename_prefix}_{self.log_timestamp}.{self.container}"
        return self._dedupe_path(os.path.join(output_dir, filename))

    def _quote_launch_path(self, path: str) -> str:
        return path.replace("\\", "\\\\").replace('"', '\\"')

    def _log(self, level: str, msg: str) -> None:
        print(f"[front_video_recorder_node] {msg}", flush=True)
        log_func = getattr(self.logger, level)
        log_func(msg, include_caller=False)

    def _log_info(self, msg: str) -> None:
        self._log("info", msg)

    def _log_warning(self, msg: str) -> None:
        self._log("warning", msg)

    def _log_error(self, msg: str) -> None:
        self._log("error", msg)

    def build_pipeline_desc(self) -> str:
        iface_part = f"multicast-iface={self.iface} " if self.iface else ""
        common = (
            f"udpsrc address={self.address} port={self.port} "
            f"{iface_part}auto-multicast=true "
            'caps="application/x-rtp,media=(string)video,encoding-name=(string)H264,clock-rate=(int)90000" '
            f"! rtpjitterbuffer latency={self.latency_ms} drop-on-latency=true "
            "! rtph264depay "
            "! h264parse config-interval=-1 "
        )

        output = self._quote_launch_path(self.output)
        if self.container == "mp4":
            return (
                common
                + "! video/x-h264,stream-format=avc,alignment=au "
                + "! mp4mux "
                + f'! filesink location="{output}" sync=false'
            )

        if self.container == "mkv":
            return common + "! matroskamux " + f'! filesink location="{output}" sync=false'

        raise ValueError("container must be mp4 or mkv")

    def _on_bus_message(self, bus, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._log_error(f"GStreamer error: {err}")
            if debug:
                self._log_error(f"GStreamer debug: {debug}")
            self.loop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            self._log_warning(f"GStreamer warning: {warn}")
            if debug:
                self._log_warning(f"GStreamer debug: {debug}")

        elif msg_type == Gst.MessageType.EOS:
            self.eos_received = True
            self._log_info("EOS received, file finalized.")
            self.loop.quit()

        return True

    def _on_eos_timeout(self):
        if self.eos_sent and not self.eos_received:
            self._log_warning(
                f"EOS timeout after {self.eos_timeout_sec:.1f}s; stopping pipeline anyway."
            )
            self.loop.quit()
        return False

    def send_eos(self):
        if self.eos_sent:
            return False

        self.eos_sent = True
        if self.pipeline is None:
            self._log_warning("Pipeline is not ready; nothing to finalize.")
            self.loop.quit()
            return False

        self._log_info("Stopping recording, sending EOS...")
        if not self.pipeline.send_event(Gst.Event.new_eos()):
            self._log_warning("Failed to send EOS to pipeline.")
            self.loop.quit()
            return False

        if self.eos_timeout_sec > 0:
            timeout_ms = max(1, int(self.eos_timeout_sec * 1000))
            GLib.timeout_add(timeout_ms, self._on_eos_timeout)
        return False

    def _handle_signal(self, signum):
        self._log_info(f"Signal {signum} received; finalizing recording.")
        self.send_eos()
        return False

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                source_id = GLib.unix_signal_add(
                    GLib.PRIORITY_DEFAULT,
                    signum,
                    self._handle_signal,
                    signum,
                )
                self._signal_source_ids.append(source_id)
            except (AttributeError, TypeError):
                signal.signal(signum, lambda sig, frame: GLib.idle_add(self._handle_signal, sig))

    def _remove_signal_handlers(self) -> None:
        for source_id in self._signal_source_ids:
            try:
                GLib.source_remove(source_id)
            except Exception:
                pass
        self._signal_source_ids = []

    def _report_output_file(self) -> None:
        if os.path.exists(self.output):
            size_mb = os.path.getsize(self.output) / 1024 / 1024
            self._log_info(f"Saved file: {self.output}, size={size_mb:.2f} MB")
        else:
            self._log_warning("Output file was not created.")

    def run(self) -> None:
        Gst.init(None)

        pipeline_desc = self.build_pipeline_desc()
        self.logger.log_init([
            "Front Video Recorder Node initialized",
            f"  Interface: {self.iface if self.iface else '<auto>'}",
            f"  Multicast: {self.address}:{self.port}",
            f"  Container: {self.container}",
            f"  Latency: {self.latency_ms}ms",
            f"  Output: {self.output}",
            f"  Log file: {self.logger.log_file}",
        ])
        self._log_info("Pipeline:")
        self._log_info(pipeline_desc)
        self._log_info(f"Recording to: {self.output}")

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

        self._install_signal_handlers()

        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING")

        try:
            self.loop.run()
        finally:
            self._remove_signal_handlers()
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            if self.bus is not None:
                self.bus.remove_signal_watch()
            self._report_output_file()


def run_front_video_recorder_node(log_dir: str = None, log_timestamp: str = None, args=None):
    """Entry point used by multi_main.py."""
    node = FrontVideoRecorderNode(log_dir=log_dir, log_timestamp=log_timestamp)
    node.run()


def main():
    run_front_video_recorder_node()


if __name__ == "__main__":
    main()
