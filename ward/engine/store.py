"""Model store: /etc/ward/models/<user>.json (DESIGN.md §5.2). Embeddings only."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

RECOGNISER = "dlib_face_recognition_resnet_model_v1"     # default embedding space (recognise.DLIB_NAME)
VERSION = 4            # v4: per-model samples, learned models, within-class whitening; v2/v3 load
EMB_DIM = 128          # dlib; ArcFace stores carry 512-d vectors (dimension taken from the data)


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
    samples: list = field(default_factory=list) # the individual embeddings this model was fused from
    learned: bool = False                       # added by online adaptation (a confident night-time accept…)
    identity: str = ""                          # who this is: "" = the account owner, else a named person on this login
    condition: str = ""                         # "" = plain, else "glasses" / "headset" / "night"… (a short extra sweep)


@dataclass
class Store:
    user: str
    models: list[Model] = field(default_factory=list)
    calib: dict = field(default_factory=lambda: {"dx": 0.0, "dy": 0.0, "s": 1.0})
    created: str = ""
    path: Path = field(default_factory=lambda: config.MODELS_DIR / "unknown.json")
    whitening: dict = field(default_factory=dict)   # spectrum -> {"W": DxD, "threshold": float, "n": int}
    recogniser: str = RECOGNISER                    # embedding space these models live in
    scale: float = 1.0                              # raw distance × scale = dlib-equivalent units (Models.distance_scale)

    @property
    def dim(self) -> int:
        return len(self.models[0].embedding) if self.models else EMB_DIM

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two embeddings of this store's space, in config (dlib-equivalent) units."""
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) * self.scale

    # -- within-class whitening (DESIGN.md §6.5a) ------------------------------
    def fit_whitening(self, spectrum: str, extra: list[tuple[str, np.ndarray]] | None = None,
                      shrink: float = 1.0, margin: float = 1.15, clamp: float | None = None) -> dict | None:
        """Learn the directions along which THIS user's embeddings vary (pose, expression, and —
        via augmented samples — lighting) and build a whitening matrix that scales them down:
            W = (Σ_w + λI)^(-1/2),   Σ_w = within-class scatter of samples around their model mean,
            λ = shrink · trace(Σ_w)/128  (shrinkage: 128 dims from a few hundred samples).
        The whitened distance |W(x − μ)| treats a night-time embedding of the user as close to
        the daytime models if it moved along a known variation direction, while a stranger,
        who differs along identity directions, stays far. `extra`: (label, embedding) pairs to
        include in the scatter only (e.g. synthetic lighting variants).
        Threshold: 95th percentile of leave-one-out whitened distances (each sample against the
        mean of the OTHER samples of its bin — the same situation as a live fused embedding
        against a stored model) × margin. shrink=1.0 / margin 1.15 chosen on a synthetic sweep:
        same-user-at-night accepted, a 28 %-different "sibling" and strangers rejected."""
        groups: dict[str, list[np.ndarray]] = {}
        for m in self.by_spectrum(spectrum):
            if m.samples:
                groups.setdefault(m.label, []).extend(np.asarray(v, dtype=np.float64) for v in m.samples)
        for label, e in (extra or []):
            groups.setdefault(label, []).append(np.asarray(e, dtype=np.float64))
        devs, loo = [], []
        for label, vecs in groups.items():
            if len(vecs) < 2:
                continue
            mu = np.mean(vecs, axis=0)
            devs.extend(v - mu for v in vecs)
            if len(vecs) >= 3:
                for i, v in enumerate(vecs):
                    others = np.mean([x for j, x in enumerate(vecs) if j != i], axis=0)
                    loo.append(v - others)
        if len(devs) < 8:
            return None
        D = np.array(devs)
        dim = D.shape[1]
        cov = D.T @ D / len(D)
        lam = shrink * np.trace(cov) / dim
        w, V = np.linalg.eigh(cov + lam * np.eye(dim))
        W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        basis = loo if len(loo) >= 8 else devs
        wd = np.array([float(np.linalg.norm(W @ d)) for d in basis])
        thr = float(np.percentile(wd, 95) * margin)
        prev = self.whitening.get(spectrum) or {}
        if clamp is not None and prev.get("threshold"):
            # a refit from use may move the threshold only a little at a time (a learn-path refit once
            # jumped it 6.4 → 47, making the whitened path accept nearly anything under the hard cap)
            p = float(prev["threshold"])
            thr = min(max(thr, p * (1 - clamp)), p * (1 + clamp))
        ent = {"W": W, "threshold": thr, "n": int(len(devs)), "lam": float(lam)}
        if prev.get("floor") is not None:                 # the calibration session's floor survives refits
            ent["floor"] = float(prev["floor"])
            ent["threshold"] = max(ent["threshold"], ent["floor"]) if clamp is not None else ent["threshold"]
        self.whitening[spectrum] = ent
        return self.whitening[spectrum]

    def wdist(self, embedding: np.ndarray, spectrum: str) -> tuple[float | None, float | None]:
        """(whitened distance to the nearest model of that spectrum, threshold) or (None, None)."""
        ent = self.whitening.get(spectrum)
        if not ent:
            return None, None
        W = np.asarray(ent["W"])
        best = float("inf")
        for m in self.by_spectrum(spectrum) + self.virtuals(spectrum):
            d = float(np.linalg.norm(W @ (m.embedding - embedding)))
            best = min(best, d)
        return (best if best < float("inf") else None), float(ent["threshold"])

    # -- queries ------------------------------------------------------------
    def by_spectrum(self, spectrum: str) -> list[Model]:
        return [m for m in self.models if m.spectrum == spectrum]

    def identities(self) -> list[str]:
        """Named identities on this login ("" = the owner), in first-seen order."""
        seen: list[str] = []
        for m in self.models:
            if m.identity not in seen:
                seen.append(m.identity)
        return seen

    def conditions(self, identity: str | None = None) -> list[str]:
        seen: list[str] = []
        for m in self.models:
            if identity is not None and m.identity != identity:
                continue
            if m.condition and m.condition not in seen:
                seen.append(m.condition)
        return seen

    def nearest(self, embedding: np.ndarray, spectrum: str) -> tuple[float, "Model | None"]:
        """Nearest stored (or interpolated) model with its distance in config units — the model
        carries the identity/condition that matched, for the log and for learning."""
        best, who = float("inf"), None
        for m in self.by_spectrum(spectrum) + self.virtuals(spectrum):
            d = float(np.linalg.norm(m.embedding - embedding)) * self.scale
            if d < best:
                best, who = d, m
        return best, who

    def remove_identity(self, identity: str) -> int:
        n = len(self.models); self.models = [m for m in self.models if m.identity != identity]; return n - len(self.models)

    def remove_condition(self, condition: str, identity: str | None = None) -> int:
        n = len(self.models)
        self.models = [m for m in self.models if not (m.condition == condition and (identity is None or m.identity == identity))]
        return n - len(self.models)

    def spectra(self) -> set[str]:
        return {m.spectrum for m in self.models}

    def min_distance(self, embedding: np.ndarray, spectrum: str) -> tuple[float, str | None]:
        """Nearest enrolled OR interpolated model of that spectrum (DESIGN.md §8, interpolation)."""
        best, label = float("inf"), None
        for m in self.by_spectrum(spectrum) + self.virtuals(spectrum):
            d = float(np.linalg.norm(m.embedding - embedding)) * self.scale
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
            pose: tuple[float, float] | None = None, samples: list | None = None, learned: bool = False,
            identity: str = "", condition: str = "") -> Model:
        nid = max((m.id for m in self.models), default=0) + 1
        m = Model(nid, label, spectrum, np.asarray(embedding, dtype=np.float64), quality,
                  time.strftime("%Y-%m-%dT%H:%M:%S%z"), pose, False,
                  [np.asarray(v, dtype=np.float64) for v in (samples or [])], learned, identity, condition)
        self.models.append(m)
        return m

    def remove_label(self, label: str, identity: str | None = None, condition: str | None = None) -> int:
        """Remove models with this pose label; when identity/condition are given only that
        person's / that condition's models go (the fused setup clears per sweep, never the rest)."""
        if identity is not None or condition is not None:
            n = len(self.models)
            self.models = [m for m in self.models if not (m.label == label and (identity is None or m.identity == identity)
                                                          and (condition is None or m.condition == condition))]
            return n - len(self.models)
        return self._remove_label_all(label)

    def _remove_label_all(self, label: str) -> int:
        before = len(self.models)
        self.models = [m for m in self.models if m.label != label]
        return before - len(self.models)

    # -- persistence -----------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": VERSION, "user": self.user,
            "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "recogniser": self.recogniser, "calib": self.calib,
            "models": [{"id": m.id, "label": m.label, "spectrum": m.spectrum,
                        "embedding": [round(float(x), 6) for x in m.embedding],
                        "quality": round(m.quality, 3), "time": m.time,
                        **({"pose": [round(m.pose[0], 1), round(m.pose[1], 1)]} if m.pose else {}),
                        **({"samples": [[round(float(x), 5) for x in v] for v in m.samples]} if m.samples else {}),
                        **({"learned": True} if m.learned else {}),
                        **({"identity": m.identity} if m.identity else {}),
                        **({"condition": m.condition} if m.condition else {})}
                       for m in self.models if not m.virtual],
            "whitening": {sp: {"W": [[round(float(x), 6) for x in row] for row in np.asarray(e["W"])],
                               "threshold": round(float(e["threshold"]), 4), "n": int(e["n"]), "lam": round(float(e.get("lam", 0)), 6)}
                          for sp, e in self.whitening.items()},
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


def load(user: str, recogniser: str | None = None, scale: float = 1.0) -> Store:
    """`recogniser`: the embedding space the caller will produce (Models.recogniser_name). A store
    written in another space raises StoreError — its models cannot be compared, re-enroll."""
    p = path_for(user)
    want = recogniser or RECOGNISER
    st = Store(user=user, path=p, recogniser=want, scale=scale)
    if not p.exists():
        return st
    data = json.loads(p.read_text())
    if data.get("version") not in (2, 3, VERSION):
        raise StoreError(f"model store version {data.get('version')} not supported (expected {VERSION})")
    have = data.get("recogniser")
    if have != want:
        raise StoreError(f"model store was written with recogniser {have}, config now uses {want}; re-enroll (`sudo ward enroll --fresh`)")
    st.created = data.get("created", "")
    st.calib = data.get("calib", st.calib)
    for m in data.get("models", []):
        pose = tuple(float(x) for x in m["pose"]) if m.get("pose") else None
        st.models.append(Model(int(m["id"]), str(m["label"]), str(m["spectrum"]),
                              np.asarray(m["embedding"], dtype=np.float64), float(m.get("quality", 0)),
                              str(m.get("time", "")), pose, False,
                              [np.asarray(v, dtype=np.float64) for v in m.get("samples", [])], bool(m.get("learned", False)),
                              str(m.get("identity", "")), str(m.get("condition", ""))))
    for sp, e in (data.get("whitening") or {}).items():
        st.whitening[sp] = {"W": np.asarray(e["W"], dtype=np.float64), "threshold": float(e["threshold"]), "n": int(e.get("n", 0)), "lam": float(e.get("lam", 0))}
    return st
