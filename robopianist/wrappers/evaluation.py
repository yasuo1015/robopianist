# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A wrapper for tracking episode statistics pertaining to music performance.

Includes both the original F1-based key press / sustain metrics and dynamic
performance metrics (VRS, DCS, GCS, VAS) that evaluate the musical expressiveness
of a performance based on MIDI velocity information.
"""

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support

from robopianist.music.midi_message import NoteOn


# ---------------------------------------------------------------------------
# Dynamic Performance Metrics (standalone functions)
# ---------------------------------------------------------------------------

def compute_velocity_range_score(velocities: np.ndarray) -> float:
    """Velocity Range Score (VRS): how much of the [1, 127] range is used.

    Combines the actual range with the standard deviation so that a performance
    with only two extreme values does not get full marks.

    Returns a value in [0, 1].
    """
    if len(velocities) < 2:
        return 0.0
    v_min, v_max = velocities.min(), velocities.max()
    range_ratio = (v_max - v_min) / 126.0
    std_ratio = np.std(velocities) / 63.5
    return float(np.clip(0.5 * range_ratio + 0.5 * std_ratio, 0.0, 1.0))


def compute_dynamic_contrast_score(
    velocities: np.ndarray,
    beat_positions: np.ndarray,
    beats_per_measure: int = 4,
) -> float:
    """Dynamic Contrast Score (DCS): strong-beat vs. weak-beat velocity contrast.

    In 4/4 time beats 0 and 2 are strong; in 3/4 only beat 0.
    A well-shaped performance shows moderate contrast (~0.15 on a [0, 1] scale).

    Returns a value in [0, 1].
    """
    if len(velocities) < 2 or len(beat_positions) < 2:
        return 0.5

    if beats_per_measure == 4:
        strong = {0, 2}
    elif beats_per_measure == 3:
        strong = {0}
    else:
        strong = {0}

    int_beats = np.floor(beat_positions).astype(int) % beats_per_measure
    strong_mask = np.isin(int_beats, list(strong))

    if not strong_mask.any() or strong_mask.all():
        return 0.5

    mean_strong = velocities[strong_mask].mean()
    mean_weak = velocities[~strong_mask].mean()
    contrast = (mean_strong - mean_weak) / 127.0

    ideal_contrast = 0.15
    dcs = float(np.exp(-((contrast - ideal_contrast) ** 2) / (2 * 0.1 ** 2)))
    return np.clip(dcs, 0.0, 1.0)


def compute_gradual_change_smoothness(velocities: np.ndarray) -> float:
    """Gradual Change Smoothness (GCS): penalises abrupt velocity jumps.

    Uses the second-order finite difference (jerk) of the velocity sequence.

    Returns a value in [0, 1] where 1 is perfectly smooth.
    """
    if len(velocities) < 3:
        return 1.0
    second_diff = np.diff(velocities.astype(float), n=2)
    mean_jerk = np.mean(np.abs(second_diff)) / 126.0
    return float(np.clip(1.0 - mean_jerk, 0.0, 1.0))


def compute_velocity_accuracy_score(
    actual_velocities: np.ndarray,
    target_velocities: np.ndarray,
) -> float:
    """Velocity Accuracy Score (VAS): how closely actual velocities match targets.

    Uses a Gaussian tolerance on the absolute error normalised to [0, 1].

    Returns a value in [0, 1].
    """
    if len(actual_velocities) == 0 or len(target_velocities) == 0:
        return 0.0
    n = min(len(actual_velocities), len(target_velocities))
    errors = np.abs(
        actual_velocities[:n].astype(float) - target_velocities[:n].astype(float)
    ) / 127.0
    margin = 0.2
    scores = np.exp(-(errors ** 2) / (2 * margin ** 2))
    return float(np.mean(scores))


def compute_dynamic_performance_score(
    vrs: float,
    dcs: float,
    gcs: float,
    vas: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted combination of VRS, DCS, GCS, VAS into a single DPS value."""
    if weights is None:
        weights = {"vrs": 0.2, "dcs": 0.1, "gcs": 0.1, "vas": 0.6}
    return float(
        weights["vrs"] * vrs
        + weights["dcs"] * dcs
        + weights["gcs"] * gcs
        + weights["vas"] * vas
    )


# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------

class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class DynamicsMetrics(NamedTuple):
    """Container for dynamic performance metrics of a single episode."""

    velocity_range_score: float
    dynamic_contrast_score: float
    gradual_change_smoothness: float
    velocity_accuracy_score: float
    dynamic_performance_score: float


# ---------------------------------------------------------------------------
# Helper: extract tempo / time-signature from the task's MIDI
# ---------------------------------------------------------------------------

def _extract_beat_info(
    task,
) -> Tuple[float, int]:
    """Return (seconds_per_beat, beats_per_measure) from the task's MIDI file."""
    try:
        midi = task._midi
        seq = midi.seq
        if seq.tempos:
            qpm = seq.tempos[0].qpm
        else:
            qpm = 120.0
        if seq.time_signatures:
            beats_per_measure = seq.time_signatures[0].numerator
        else:
            beats_per_measure = 4
    except Exception:
        qpm = 120.0
        beats_per_measure = 4
    seconds_per_beat = 60.0 / max(qpm, 1.0)
    return seconds_per_beat, beats_per_measure


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    When the task exposes a MIDI module (via ``task.piano.midi_module``), dynamic
    performance metrics (VRS, DCS, GCS, VAS, DPS) are also computed from the
    NoteOn velocity values generated during the episode.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(self, environment: dm_env.Environment, deque_size: int = 1) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []
        self._sustain_presses: List[np.ndarray] = []

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

        # Sustain metrics.
        self._sustain_precisions: Deque[float] = deque(maxlen=deque_size)
        self._sustain_recalls: Deque[float] = deque(maxlen=deque_size)
        self._sustain_f1s: Deque[float] = deque(maxlen=deque_size)

        # Dynamic performance metrics.
        self._vrs_scores: Deque[float] = deque(maxlen=deque_size)
        self._dcs_scores: Deque[float] = deque(maxlen=deque_size)
        self._gcs_scores: Deque[float] = deque(maxlen=deque_size)
        self._vas_scores: Deque[float] = deque(maxlen=deque_size)
        self._dps_scores: Deque[float] = deque(maxlen=deque_size)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = self._environment.task.piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))

        if timestep.last():
            key_press_metrics = self._compute_key_press_metrics()
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            sustain_metrics = self._compute_sustain_metrics()
            self._sustain_precisions.append(sustain_metrics.precision)
            self._sustain_recalls.append(sustain_metrics.recall)
            self._sustain_f1s.append(sustain_metrics.f1)

            dynamics = self._compute_dynamics_metrics()
            self._vrs_scores.append(dynamics.velocity_range_score)
            self._dcs_scores.append(dynamics.dynamic_contrast_score)
            self._gcs_scores.append(dynamics.gradual_change_smoothness)
            self._vas_scores.append(dynamics.velocity_accuracy_score)
            self._dps_scores.append(dynamics.dynamic_performance_score)

            self._key_presses = []
            self._sustain_presses = []
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        return self._environment.reset()

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean metrics over the last `deque_size` episodes."""
        if not self._key_press_precisions:
            raise ValueError("No episode metrics available yet.")

        def _mean(seq: Sequence[float]) -> float:
            return sum(seq) / len(seq)

        metrics: Dict[str, float] = {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
            "sustain_precision": _mean(self._sustain_precisions),
            "sustain_recall": _mean(self._sustain_recalls),
            "sustain_f1": _mean(self._sustain_f1s),
        }

        if self._dps_scores:
            metrics.update({
                "velocity_range_score": _mean(self._vrs_scores),
                "dynamic_contrast_score": _mean(self._dcs_scores),
                "gradual_change_smoothness": _mean(self._gcs_scores),
                "velocity_accuracy_score": _mean(self._vas_scores),
                "dynamic_performance_score": _mean(self._dps_scores),
            })

        return metrics

    # Helper methods.

    def _compute_dynamics_metrics(self) -> DynamicsMetrics:
        """Compute dynamic performance metrics for the current episode."""
        task = self._environment.task

        # --- Collect actual velocities from the piano's MIDI module ---
        actual_vels: List[int] = []
        actual_times: List[float] = []
        try:
            all_msgs = task.piano.midi_module.get_all_midi_messages()
            for msg in all_msgs:
                if isinstance(msg, NoteOn):
                    actual_vels.append(msg.velocity)
                    actual_times.append(msg.time)
        except Exception:
            pass

        # --- Collect target velocities from the ground-truth trajectory ---
        target_vels: List[int] = []
        try:
            note_seq = task._notes
            n_steps = min(len(note_seq), len(self._key_presses))
            for t in range(n_steps):
                for note in note_seq[t]:
                    target_vels.append(note.velocity)
        except Exception:
            pass

        actual_arr = np.array(actual_vels, dtype=np.int32)
        target_arr = np.array(target_vels, dtype=np.int32)

        # --- Beat position for each actual note ---
        seconds_per_beat, beats_per_measure = _extract_beat_info(task)
        if len(actual_times) > 0 and seconds_per_beat > 0:
            beat_positions = np.array(actual_times) / seconds_per_beat
        else:
            beat_positions = np.array([], dtype=np.float64)

        # --- Compute sub-metrics ---
        vrs = compute_velocity_range_score(actual_arr)
        dcs = compute_dynamic_contrast_score(actual_arr, beat_positions, beats_per_measure)
        gcs = compute_gradual_change_smoothness(actual_arr)
        vas = compute_velocity_accuracy_score(actual_arr, target_arr)
        dps = compute_dynamic_performance_score(vrs, dcs, gcs, vas)

        return DynamicsMetrics(
            velocity_range_score=vrs,
            dynamic_contrast_score=dcs,
            gradual_change_smoothness=gcs,
            velocity_accuracy_score=vas,
            dynamic_performance_score=dps,
        )

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for key presses over the episode."""
        # Get the ground truth key presses.
        note_seq = self._environment.task._notes
        ground_truth = []
        for notes in note_seq:
            presses = np.zeros((self._environment.task.piano.n_keys,), dtype=np.float64)
            keys = [note.key for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # Deal with the case where the episode gets truncated due to a failure. In this
        # case, the length of the key presses will be less than or equal to the length
        # of the ground truth.
        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._key_presses)]

        assert len(ground_truth) == len(self._key_presses)

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._key_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for sustain presses over the episode."""
        # Get the ground truth sustain presses.
        ground_truth = [
            np.atleast_1d(v).astype(float) for v in self._environment.task._sustains
        ]

        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._sustain_presses)]

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._sustain_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)
