"""Model store: /etc/faceup/models/<user>.json (DESIGN.md §5.2). Embeddings only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

RECOGNISER = "dlib_face_recognition_resnet_model_v1"
VERSION = 3            # v3 adds optional pose (yaw, pitch) per model; v2 files still load


@dataclass
class Model:
    id: int
    label: str
    spectrum: str            # "ir" | "rgb"
    embedding: np.ndarray    # 128 floats, L2-normalised
    quality: float
    time: str
    pose: tuple[float, float] | None = None    # (yaw, pitch) degrees, set by free-move enrollment
    virtual: bool = False                       # interpolated between two enrolled poses (never saved)


@dataclass
class Store:
    user: str
    models: list[Model] = field(default_factory=list)
    calib: dict = field(default_factory=lambda: {"dx": 0.0, "dy": 0.0, "s": 1.0})
    created: str = ""
    path: Path = field(default_factory=lambda: config.MODELS_DIR / "unknown.json")

    # -- queries ------------------------------------------------------------
    def by_spectrum(self, spectrum: str) -> list[Model]:
        return [m for m in self.models if m.spectrum == spectrum]

    def spectra(self) -> set[str]:
        return {m.spectrum for m in self.models}

    def min_distance(self, embedding: np.ndarray, spectrum: str) -> tuple[float, str | None]:
        """Nearest enrolled OR interpolated model of that spectrum (DESIGN.md §8, interpolation)."""
        best, label = float("inf"), None
        for m in self.by_spectrum(spectrum) + self.virtuals(spectrum):
            d = float(np.linalg.norm(m.embedding - embedding))
            if d < best:
                best, label = d, m.label
        return best, label

    def virtuals(self, spectrum: str) -> list[Model]:
        """Midpoints between pose-adjacent enrolled models: covers the angles in between
        two registered cells without asking the user to hold them."""
        from . import pose as posemod
        by_cell: dict[tuple[int, int], Model] = {}
        for m in self.by_spectrum(spectrum):
            if m.pose is None:
                continue
            c = posemod.Pose(m.pose[0], m.pose[1], 0.0).cell()
            if c is not None and (c not in by_cell or m.quality > by_cell[c].quality):
                by_cell[c] = m
        out = []
        for c, m in by_cell.items():
            for n in posemod.neighbours(c):
                if n in by_cell:
                    e = m.embedding + by_cell[n].embedding
                    e = e / (np.linalg.norm(e) or 1.0)
                    out.append(Model(-1, f"~{m.label}+{by_cell[n].label}", spectrum, e,
                                     min(m.quality, by_cell[n].quality), "", None, True))
        return out

    # -- mutation -------------------------------------------------------------
    def add(self, label: str, spectrum: str, embedding: np.ndarray, quality: float,
            pose: tuple[float, float] | None = None) -> Model:
        nid = max((m.id for m in self.models), default=0) + 1
        m = Model(nid, label, spectrum, np.asarray(embedding, dtype=np.float64), quality,
                  time.strftime("%Y-%m-%dT%H:%M:%S%z"), pose)
        self.models.append(m)
        return m

    def remove_label(self, label: str) -> int:
        before = len(self.models)
        self.models = [m for m in self.models if m.label != label]
        return before - len(self.models)

    # -- persistence -----------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": VERSION, "user": self.user,
            "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "recogniser": RECOGNISER, "calib": self.calib,
            "models": [{"id": m.id, "label": m.label, "spectrum": m.spectrum,
                        "embedding": [round(float(x), 6) for x in m.embedding],
                        "quality": round(m.quality, 3), "time": m.time,
                        **({"pose": [round(m.pose[0], 1), round(m.pose[1], 1)]} if m.pose else {})}
                       for m in self.models if not m.virtual],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


class StoreError(Exception):
    pass


def path_for(user: str) -> Path:
    if not user or "/" in user or user.startswith("."):
        raise StoreError("invalid user name")
    return config.MODELS_DIR / f"{user}.json"


def load(user: str) -> Store:
    p = path_for(user)
    st = Store(user=user, path=p)
    if not p.exists():
        return st
    data = json.loads(p.read_text())
    if data.get("version") not in (2, VERSION):
        raise StoreError(f"model store version {data.get('version')} not supported (expected {VERSION})")
    if data.get("recogniser") != RECOGNISER:
        raise StoreError(f"model store was written by {data.get('recogniser')}; re-enroll")
    st.created = data.get("created", "")
    st.calib = data.get("calib", st.calib)
    for m in data.get("models", []):
        pose = tuple(float(x) for x in m["pose"]) if m.get("pose") else None
        st.models.append(Model(int(m["id"]), str(m["label"]), str(m["spectrum"]),
                              np.asarray(m["embedding"], dtype=np.float64), float(m.get("quality", 0)),
                              str(m.get("time", "")), pose))
    return st
