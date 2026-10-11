"""Known-tester allowlist: suppress port-scan alerts from approved internal scanners.

Problem (from the project risk review): legitimate IT tools such as Nessus or Nmap inventory sweeps look
exactly like an attacker's port scan, so a detector would raise alerts for them every time they run.

Design:
  * Each approved scanner is registered in config/known_scanners.csv with: source IP or CIDR block,
    the ports it is allowed to probe, the days and time window it runs, an owner, a change ticket and an
    expiry date.
  * An alert is suppressed only if ALL conditions match (source, port, day, time window, not expired).
    A registered scanner that probes outside its approved ports or schedule is still alerted, because a
    compromised or spoofed scanner host is a real risk.
  * Suppressed alerts are never deleted. They are kept with status "suppressed" and the matching entry
    name, so analysts can audit them.

The CICIDS2017 MachineLearningCVE files used in Deliverable 2 have no source-IP or timestamp columns,
so this module is unit-tested on synthetic alerts (tests/test_allowlist.py) and will be evaluated on the
CIC "GeneratedLabelledFlows" release, which includes Source IP and Timestamp.
"""
from __future__ import annotations

import csv
import ipaddress
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "known_scanners.csv"


def _parse_ports(spec: str) -> set[int] | tuple[int, int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return (lo, hi)
    return {int(p) for p in spec.split(",") if p.strip()}


def _parse_days(spec: str) -> set[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        i, j = DAY_NAMES.index(a), DAY_NAMES.index(b)
        return set(range(i, j + 1)) if i <= j else set(range(i, 7)) | set(range(0, j + 1))
    return {DAY_NAMES.index(d.strip()) for d in spec.split(",")}


@dataclass(frozen=True)
class Entry:
    name: str
    network: ipaddress.IPv4Network
    ports: object
    days: frozenset
    start: time
    end: time
    owner: str
    ticket: str
    expires: date

    def port_ok(self, port: int) -> bool:
        if isinstance(self.ports, tuple):
            return self.ports[0] <= port <= self.ports[1]
        return port in self.ports

    def time_ok(self, ts: datetime) -> bool:
        t = ts.time()
        in_window = self.start <= t <= self.end if self.start <= self.end else (t >= self.start or t <= self.end)
        return ts.weekday() in self.days and in_window

    def matches(self, src_ip: str, dst_port: int, ts: datetime) -> bool:
        return (ipaddress.ip_address(src_ip) in self.network and self.port_ok(int(dst_port))
                and self.time_ok(ts) and ts.date() <= self.expires)


def load_allowlist(path: str | Path = DEFAULT_PATH) -> list[Entry]:
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entries.append(Entry(
                name=row["name"], network=ipaddress.ip_network(row["source"], strict=False),
                ports=_parse_ports(row["allowed_ports"]), days=frozenset(_parse_days(row["days"])),
                start=time.fromisoformat(row["start_time"]), end=time.fromisoformat(row["end_time"]),
                owner=row["owner"], ticket=row["ticket"], expires=date.fromisoformat(row["expires"])))
    return entries


def match(src_ip: str, dst_port: int, ts: datetime, entries: list[Entry]) -> Entry | None:
    for e in entries:
        if e.matches(src_ip, dst_port, ts):
            return e
    return None


def apply_allowlist(alerts, entries: list[Entry]):
    """alerts: pandas DataFrame with columns src_ip, dst_port, timestamp (datetime).
    Returns a copy with 'status' ('alert' or 'suppressed') and 'allowlist_entry' columns."""
    out = alerts.copy()
    hits = [match(r.src_ip, r.dst_port, r.timestamp, entries) for r in out.itertuples()]
    out["status"] = ["suppressed" if h else "alert" for h in hits]
    out["allowlist_entry"] = [h.name if h else "" for h in hits]
    return out
