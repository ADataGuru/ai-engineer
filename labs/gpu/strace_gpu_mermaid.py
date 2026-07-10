#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strace_gpu_mermaid.py

Transforme une trace strace contenant des appels NVIDIA en un fichier Markdown
avec des diagrammes Mermaid montrant les interactions probables entre :
CPU / Linux kernel / driver NVIDIA / RAM / GPU / HBM.

Usage typique :

    strace -f -ttt -T -yy -s 256 \
      -e trace=openat,close,ioctl,mmap,mmap2,munmap,mprotect,read,write,readv,writev,pread64,pwrite64 \
      python ton_programme.py 2> trace.strace

    python strace_gpu_mermaid.py trace.strace --out gpu_trace.md --split 50

Limites importantes :
- strace voit les appels système CPU -> kernel/driver.
- strace ne voit pas les copies CUDA réelles HtoD/DtoH si elles sont cachées
  dans des ioctl opaques.
- strace ne permet pas de savoir si une donnée est lue depuis L1, L2 ou HBM.
  Pour cela, il faut utiliser Nsight Compute / CUPTI / compteurs GPU.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from collections import Counter
from typing import Optional


# Commandes NVIDIA connues : type 0x46 = 'F'
# Ces valeurs sont observées côté driver NVIDIA Linux, mais le payload exact
# de beaucoup d'ioctl reste propriétaire / opaque.
NVIDIA_IOCTLS = {
    0x27: "NV_ESC_RM_MAP_MEMORY",          # mapping mémoire GPU
    0x2A: "NV_ESC_RM_ALLOC_MEMORY",        # allocation mémoire
    0x2B: "NV_ESC_RM_FREE",                # libération ressource
    0x4E: "NV_ESC_RM_ALLOC_OBJECT",        # allocation objet CUDA/driver
    0xC9: "NV_ESC_REGISTER_FD",            # enregistrement fd
    0xCE: "NV_ESC_RM_CONTROL",             # contrôle général opaque
    0x49: "NV_ESC_RM_NVLOG",               # logging interne
    0x21: "NV_ESC_RM_GET_EVENT_DATA",      # récupération événements
    0x41: "NV_ESC_RM_VID_HEAP_CONTROL",    # gestion heap vidéo
    0x44: "NV_ESC_RM_I2C_ACCESS",          # accès I2C
    0x48: "NV_ESC_RM_IDLE_CHANNELS",       # idle channels
    0x17: "NV_ESC_RM_DUP_OBJECT",          # duplication objet
    0x1B: "NV_ESC_RM_SHARE",               # partage ressource
}


IMPORTANT_IOCTLS = {
    "NV_ESC_RM_MAP_MEMORY",
    "NV_ESC_RM_ALLOC_MEMORY",
    "NV_ESC_RM_FREE",
    "NV_ESC_RM_ALLOC_OBJECT",
    "NV_ESC_REGISTER_FD",
    "NV_ESC_RM_CONTROL",
    "NV_ESC_RM_VID_HEAP_CONTROL",
}


GPU_PATH_HINTS = (
    "/dev/nvidia",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-modeset",
    "cuda",
    "uvm",
)


@dataclass
class Event:
    idx: int
    pid: str
    ts: Optional[str]
    syscall: str
    kind: str
    label: str
    detail: str
    raw: str
    fd: Optional[str] = None
    path: Optional[str] = None
    important: bool = True


def mermaid_escape(text: str, max_len: int = 220) -> str:
    """Échappe les caractères gênants pour Mermaid."""
    text = text.replace("\n", " ").strip()
    text = text.replace(":", " -")
    text = text.replace("|", "/")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace('"', "'")
    return text[:max_len]


def split_top_args(text: str) -> list[str]:
    """
    Coupe les arguments d'un syscall en respectant les parenthèses,
    crochets, accolades et chaînes entre guillemets.
    """
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote = False
    escape = False

    for ch in text:
        if in_quote:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue

        if ch == '"':
            in_quote = True
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if buf:
        args.append("".join(buf).strip())

    return args


def parse_fd_arg(arg: str) -> tuple[Optional[str], Optional[str]]:
    """
    Exemples reconnus :
        12
        12</dev/nvidiactl>
        -1
    """
    m = re.match(r"(?P<fd>-?\d+)(?:<(?P<path>[^>]+)>)?", arg.strip())
    if not m:
        return None, None
    return m.group("fd"), m.group("path")


def parse_ret_fd(ret: str) -> tuple[Optional[str], Optional[str]]:
    """
    Exemples reconnus :
        3
        3</dev/nvidiactl>
        -1 ENOENT (...)
    """
    m = re.match(r"(?P<fd>-?\d+)(?:<(?P<path>[^>]+)>)?", ret.strip())
    if not m:
        return None, None
    return m.group("fd"), m.group("path")


def is_gpu_path(path: Optional[str]) -> bool:
    if not path:
        return False
    path_l = path.lower()
    return any(hint.lower() in path_l for hint in GPU_PATH_HINTS)


def parse_syscall(line: str) -> Optional[dict]:
    """Parse une ligne strace standard."""
    original = line.rstrip()
    body = original.strip()

    if not body:
        return None

    # Les lignes unfinished/resumed peuvent être reconstruites, mais pour garder
    # le script robuste et simple on les ignore.
    if "unfinished ..." in body or "resumed>" in body:
        return None

    pid = "main"
    ts: Optional[str] = None

    # Format avec pid : [pid 1234] syscall(...)
    m = re.match(r"^\[pid\s+(?P<pid>\d+)\]\s+(?P<body>.*)$", body)
    if m:
        pid = m.group("pid")
        body = m.group("body").strip()

    # Format avec timestamp -ttt : 1712345678.123456 syscall(...)
    m = re.match(r"^(?P<ts>\d+\.\d+)\s+(?P<body>.*)$", body)
    if m:
        ts = m.group("ts")
        body = m.group("body").strip()

    # Format principal : syscall(args) = ret <duration>
    m = re.match(
        r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\((?P<args>.*)\)\s+=\s+(?P<ret>.*?)(?:\s+<(?P<dur>[\d.]+)>)?$",
        body,
    )
    if not m:
        return None

    return {
        "pid": pid,
        "ts": ts,
        "name": m.group("name"),
        "args": split_top_args(m.group("args")),
        "ret": m.group("ret"),
        "raw": original,
    }


def decode_ioctl_request(req: str) -> tuple[str, Optional[int]]:
    """
    Essaie de décoder une requête ioctl NVIDIA affichée par strace.

    Format fréquent :
        _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0xce, 0x20)
    """
    m = re.search(
        r"_IOC\([^,]+,\s*(?P<type>0x[0-9a-fA-F]+|\d+),\s*(?P<cmd>0x[0-9a-fA-F]+|\d+)",
        req,
    )
    if m:
        cmd_id = int(m.group("cmd"), 0)
        return NVIDIA_IOCTLS.get(cmd_id, f"UNKNOWN_0x{cmd_id:02x}"), cmd_id

    # Fallback si strace affiche seulement une valeur brute.
    # Attention : ce fallback ne sait pas extraire proprement _IOC_NR.
    m2 = re.search(r"0x[0-9a-fA-F]+|\d+", req)
    if m2:
        raw_id = int(m2.group(0), 0)
        return f"RAW_IOCTL_0x{raw_id:x}", raw_id

    return "UNKNOWN_IOCTL", None


def classify(parsed: dict, idx: int, fd_paths: dict[str, str]) -> Optional[Event]:
    """Classe un syscall strace en événement GPU/RAM/driver exploitable."""
    name = parsed["name"]
    args = parsed["args"]
    ret = parsed["ret"]
    pid = parsed["pid"]
    ts = parsed["ts"]
    raw = parsed["raw"]

    # open/openat/openat2 : associe fd -> /dev/nvidia*
    if name in {"open", "openat", "openat2"}:
        quoted = re.findall(r'"([^"]+)"', raw)
        path = quoted[-1] if quoted else None
        fd, ret_path = parse_ret_fd(ret)
        if ret_path:
            path = ret_path
        if fd and path:
            fd_paths[fd] = path

        if is_gpu_path(path):
            return Event(
                idx=idx,
                pid=pid,
                ts=ts,
                syscall=name,
                kind="open",
                label=f"open {path}",
                detail=f"fd={fd}",
                raw=raw,
                fd=fd,
                path=path,
            )

    # close : fermeture d'un fd GPU connu
    if name == "close" and args:
        fd, path = parse_fd_arg(args[0])
        path = path or fd_paths.get(fd or "")
        if is_gpu_path(path):
            if fd:
                fd_paths.pop(fd, None)
            return Event(
                idx=idx,
                pid=pid,
                ts=ts,
                syscall=name,
                kind="close",
                label=f"close {path}",
                detail=f"fd={fd}",
                raw=raw,
                fd=fd,
                path=path,
            )

    # ioctl : cœur des interactions avec le driver NVIDIA
    if name == "ioctl" and len(args) >= 2:
        fd, path = parse_fd_arg(args[0])
        path = path or fd_paths.get(fd or "")

        req = args[1]
        cmd_name, _cmd_id = decode_ioctl_request(req)

        gpu_related = is_gpu_path(path) or cmd_name.startswith("NV_ESC") or "0x46" in req
        if gpu_related:
            important = cmd_name in IMPORTANT_IOCTLS or cmd_name.startswith("UNKNOWN")
            return Event(
                idx=idx,
                pid=pid,
                ts=ts,
                syscall=name,
                kind="ioctl",
                label=cmd_name,
                detail=f"fd={fd}, dev={path or '?'}, req={req}",
                raw=raw,
                fd=fd,
                path=path,
                important=important,
            )

    # mmap/mmap2 : mapping mémoire user-space vers fd NVIDIA
    if name in {"mmap", "mmap2"} and len(args) >= 5:
        length = args[1] if len(args) > 1 else "?"
        fd_arg = args[4] if len(args) > 4 else ""
        fd, path = parse_fd_arg(fd_arg)
        path = path or fd_paths.get(fd or "")
        addr = ret.split()[0] if ret else "?"

        if is_gpu_path(path):
            return Event(
                idx=idx,
                pid=pid,
                ts=ts,
                syscall=name,
                kind="mmap",
                label=f"mmap {path}",
                detail=f"addr={addr}, len={length}, fd={fd}",
                raw=raw,
                fd=fd,
                path=path,
            )

    # munmap : fin d'un mapping user-space. On ne sait pas toujours si c'était GPU,
    # mais cela peut aider à lire la timeline.
    if name == "munmap" and len(args) >= 2:
        return Event(
            idx=idx,
            pid=pid,
            ts=ts,
            syscall=name,
            kind="munmap",
            label="munmap user mapping",
            detail=f"addr={args[0]}, len={args[1]}",
            raw=raw,
            important=True,
        )

    # read/write/pread/pwrite vers fd NVIDIA : rarement les gros transferts CUDA,
    # mais utile si une application lit/écrit directement sur /dev/nvidia*.
    if name in {"read", "write", "readv", "writev", "pread64", "pwrite64"} and args:
        fd, path = parse_fd_arg(args[0])
        path = path or fd_paths.get(fd or "")
        if is_gpu_path(path):
            size = args[2] if len(args) > 2 else "?"
            return Event(
                idx=idx,
                pid=pid,
                ts=ts,
                syscall=name,
                kind="io",
                label=f"{name} {path}",
                detail=f"fd={fd}, size={size}",
                raw=raw,
                fd=fd,
                path=path,
            )

    return None


def event_to_mermaid(ev: Event) -> list[str]:
    """Convertit un événement en lignes Mermaid sequenceDiagram."""
    label = mermaid_escape(ev.label)
    detail = mermaid_escape(ev.detail)

    prefix = f"pid={ev.pid}"
    if ev.ts:
        prefix += f", t={ev.ts}"

    if ev.kind == "open":
        return [
            f"CPU->>Kernel: {prefix} open device",
            f"Kernel-->>CPU: {label} / {detail}",
        ]

    if ev.kind == "close":
        return [
            f"CPU->>Kernel: {prefix} close fd",
            f"Kernel-->>CPU: {label} / {detail}",
        ]

    if ev.kind == "mmap":
        return [
            f"CPU->>Kernel: {prefix} mmap NVIDIA fd",
            "Kernel->>Driver: demande de mapping",
            f"Driver-->>CPU: mapping userspace / {detail}",
            "Note over CPU,HBM: mmap indique une fenêtre mappée, pas une copie CPU-GPU prouvée",
        ]

    if ev.kind == "munmap":
        return [
            f"CPU->>Kernel: {prefix} munmap",
            f"Kernel-->>CPU: {detail}",
        ]

    if ev.kind == "io":
        return [
            f"CPU->>Driver: {prefix} {label}",
            f"Driver-->>CPU: {detail}",
        ]

    if ev.kind == "ioctl":
        lines = [f"CPU->>Driver: {prefix} ioctl {label}"]

        if ev.label == "NV_ESC_RM_ALLOC_MEMORY":
            lines += [
                "Driver->>HBM: réserve mémoire GPU / VRAM, inféré",
                f"Driver-->>CPU: handle ou statut / {detail}",
            ]
        elif ev.label == "NV_ESC_RM_MAP_MEMORY":
            lines += [
                "Driver->>HBM: prépare mapping mémoire GPU, inféré",
                "Driver-->>CPU: mémoire GPU mappable côté CPU, inféré",
                "Note over CPU,HBM: mapping != transfert de données",
            ]
        elif ev.label == "NV_ESC_RM_FREE":
            lines += [
                "Driver->>HBM: libère ressource GPU, inféré",
                f"Driver-->>CPU: statut / {detail}",
            ]
        elif ev.label in {"NV_ESC_RM_ALLOC_OBJECT", "NV_ESC_REGISTER_FD"}:
            lines += [
                "Driver->>GPU: crée ou associe un objet/contexte GPU, inféré",
                f"Driver-->>CPU: statut / {detail}",
            ]
        elif ev.label in {"NV_ESC_RM_CONTROL", "NV_ESC_RM_VID_HEAP_CONTROL"}:
            lines += [
                "Driver->>GPU: contrôle opaque - launch/copy/sync possible selon payload",
                f"Driver-->>CPU: statut / {detail}",
                "Note over Driver,GPU: strace ne décode pas le payload NVIDIA propriétaire",
            ]
        else:
            lines += [f"Driver-->>CPU: {detail}"]

        return lines

    return [
        f"CPU->>Kernel: {prefix} {label}",
        f"Kernel-->>CPU: {detail}",
    ]


def make_overview(events: list[Event]) -> str:
    counts = Counter(ev.label for ev in events if ev.kind == "ioctl")
    top = "\\n".join(f"{name}: {count}" for name, count in counts.most_common(8))
    if not top:
        top = "aucun ioctl NVIDIA détecté"

    return f"""```mermaid
flowchart LR
    CPU["CPU process / user-space"]
    RAM["RAM / mappings user-space"]
    Kernel["Linux kernel syscall boundary"]
    Driver["NVIDIA kernel driver"]
    GPU["GPU engines / context"]
    L1["L1 / TEX cache"]
    L2["L2 cache"]
    HBM["HBM / VRAM / device memory"]

    CPU -->|"openat / ioctl / mmap / munmap"| Kernel
    Kernel -->|"dispatch fd /dev/nvidia*"| Driver
    Driver -->|"alloc/map/free/control, inféré"| GPU
    CPU <-->|"mmap: fenêtre d'adresse, pas forcément copie"| RAM
    Driver <-->|"DMA / migration possible selon payload"| HBM

    GPU -. "non visible avec strace" .-> L1
    L1 -. "compteurs requis" .-> L2
    L2 -. "compteurs requis" .-> HBM

    Summary["ioctl NVIDIA principaux\\n{mermaid_escape(top)}"]
    Summary --- Driver
```
"""


def make_sequence(events: list[Event], title: str) -> str:
    lines = [
        "```mermaid",
        "sequenceDiagram",
        "    autonumber",
        "    participant CPU as CPU process",
        "    participant Kernel as Linux kernel",
        "    participant Driver as NVIDIA driver",
        "    participant RAM as RAM/user mappings",
        "    participant GPU as GPU engines",
        "    participant HBM as HBM/VRAM",
        f"    Note over CPU,HBM: {mermaid_escape(title)}",
    ]

    for ev in events:
        for line in event_to_mermaid(ev):
            lines.append("    " + line)

    lines.append("```")
    return "\n".join(lines) + "\n"


def write_markdown(events: list[Event], out_path: str, split: int) -> None:
    by_kind = Counter(ev.kind for ev in events)
    by_ioctl = Counter(ev.label for ev in events if ev.kind == "ioctl")

    md: list[str] = []
    md.append("# Vue Mermaid des interactions CPU / RAM / driver NVIDIA / GPU\n\n")
    md.append("## Résumé\n\n")
    md.append(f"- Événements retenus : `{len(events)}`\n")
    md.append(f"- Types : `{dict(by_kind)}`\n")
    md.append(f"- ioctl principaux : `{dict(by_ioctl.most_common(12))}`\n\n")

    md.append(
        "> Limite : ce graphe montre les appels système et des inférences raisonnables. "
        "Il ne prouve pas les transferts réels HtoD/DtoH ni les hits/misses L1/L2.\n\n"
    )

    md.append("## Vue globale\n\n")
    md.append(make_overview(events))

    md.append("\n## Timeline détaillée\n\n")
    if not events:
        md.append(
            "Aucun événement GPU NVIDIA important n'a été détecté. "
            "Vérifie que la trace contient `ioctl`, `mmap`, `openat` et les chemins `/dev/nvidia*`.\n"
        )
    else:
        for n in range(0, len(events), split):
            chunk = events[n:n + split]
            md.append(f"### Segment {n // split + 1}\n\n")
            md.append(make_sequence(chunk, f"strace events {n + 1} à {n + len(chunk)}"))
            md.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(md))


def parse_trace_file(path: str, include_all: bool) -> list[Event]:
    fd_paths: dict[str, str] = {}
    events: list[Event] = []

    with open(path, "r", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            parsed = parse_syscall(line)
            if not parsed:
                continue

            ev = classify(parsed, idx, fd_paths)
            if not ev:
                continue

            if include_all or ev.important:
                events.append(ev)

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des diagrammes Mermaid depuis une trace strace NVIDIA/CUDA."
    )
    parser.add_argument(
        "strace_file",
        help="Fichier strace à analyser, par exemple trace.strace",
    )
    parser.add_argument(
        "--out",
        default="gpu_trace_mermaid.md",
        help="Fichier Markdown de sortie, par défaut gpu_trace_mermaid.md",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=60,
        help="Nombre d'événements par diagramme sequenceDiagram, par défaut 60",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inclure aussi les ioctl NVIDIA jugés moins importants",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Affiche un résumé console plus détaillé",
    )

    args = parser.parse_args()

    if args.split <= 0:
        raise SystemExit("Erreur : --split doit être supérieur à 0")

    events = parse_trace_file(args.strace_file, include_all=args.all)
    write_markdown(events, args.out, split=args.split)

    by_kind = Counter(ev.kind for ev in events)
    by_ioctl = Counter(ev.label for ev in events if ev.kind == "ioctl")

    print(f"Fichier généré : {args.out}")
    print(f"Événements retenus : {len(events)}")

    if args.print_summary:
        print(f"Types : {dict(by_kind)}")
        print(f"Top ioctl : {dict(by_ioctl.most_common(15))}")


if __name__ == "__main__":
    main()
