"""Explicit machine events; does not command rotation or invent PLC registers.

TCP JSON is an opt-in bridge contract for the plant's PLC gateway. It is not a
claim of support for an unspecified proprietary PLC protocol.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class MachineEvent:
    kind: str
    part_id: str | None = None
    angle: int | None = None
    fit_check_complete: bool = False

    @classmethod
    def parse(cls, line: str | bytes) -> "MachineEvent":
        data = json.loads(line)
        kind = data.get("event")
        if kind not in {"begin", "stopped", "moving", "abort", "heartbeat"}:
            raise ValueError("Unknown machine event")
        if kind == "begin" and (not isinstance(data.get("part_id"), str) or not data["part_id"].strip() or data.get("fit_check_complete") is not True):
            raise ValueError("Begin requires part_id and fit_check_complete=true after the 360° fitting rotation")
        if kind == "stopped" and (type(data.get("angle")) is not int or data["angle"] not in {60, 120, 180, 240, 300, 360}):
            raise ValueError("Stopped requires an integer angle at a fixed position")
        return cls(kind, data.get("part_id"), data.get("angle"), data.get("fit_check_complete", False))


class TCPMachineConnection:
    def __init__(self, host: str, port: int) -> None:
        self.socket = socket.create_connection((host, port), timeout=1.0)
        self.socket.settimeout(.05)
        self.buffer = b""

    def poll(self) -> list[MachineEvent]:
        try:
            data = self.socket.recv(4096)
        except socket.timeout:
            return []
        if not data:
            raise ConnectionError("PLC event bridge disconnected")
        self.buffer += data
        if len(self.buffer) > 65536:
            raise ValueError("PLC event exceeds maximum size")
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if line.strip():
                events.append(MachineEvent.parse(line))
        return events

    def acknowledge(self, payload: dict) -> None:
        self.socket.sendall((json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"))

    def close(self) -> None:
        self.socket.close()
