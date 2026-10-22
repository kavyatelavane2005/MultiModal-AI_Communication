"""Device-local enrollment and matching for user-specific sign vocabularies.

The shared TGCN checkpoint stays unchanged. Personal signs are stored as a few
compressed landmark sequences and compared with a small, time-warped template
matcher after normalizing for signer position and camera scale.
"""
import base64
import json
import os
import tempfile
import zlib

import numpy as np


SEQUENCE_SHAPE = (50, 55, 2)
REQUIRED_EXAMPLES = 3
MAX_EXAMPLES_PER_SIGN = 8
MAX_SIGNS = 40


class PersonalSignStore:
    """Persist and match a small private vocabulary for this local install."""

    def __init__(self, path):
        self.path = path
        self._signs = {}
        self._normalized = {}
        self._thresholds = {}
        self._load()
        self._refresh_all()

    def list_signs(self):
        return [
            {
                'name': item['name'],
                'text': item['text'],
                'examples': len(item['samples']),
            }
            for item in sorted(self._signs.values(), key=lambda value: value['name'].casefold())
        ]

    def delete(self, name):
        key = str(name or '').casefold()
        if key not in self._signs:
            return False
        del self._signs[key]
        self._normalized.pop(key, None)
        self._thresholds.pop(key, None)
        self._save()
        return True

    def add_examples(self, name, output_text, sequences):
        name = _clean_text(name, 48)
        output_text = _clean_text(output_text, 100)
        if not name or not output_text:
            raise ValueError('Enter both a sign name and the text it should add to the sentence.')

        key = name.casefold()
        if key not in self._signs and len(self._signs) >= MAX_SIGNS:
            raise ValueError(f'The local vocabulary is limited to {MAX_SIGNS} signs.')

        item = self._signs.setdefault(key, {'name': name, 'text': output_text, 'samples': []})
        item['name'] = name
        item['text'] = output_text
        item['samples'].extend(_encode_sequence(sequence) for sequence in sequences)
        item['samples'] = item['samples'][-MAX_EXAMPLES_PER_SIGN:]
        self._refresh_class(key)
        self._save()

    def match(self, sequence):
        """Return the best confident custom match, or None when uncertain."""
        query = _normalize_sequence(sequence)
        if query is None or not self._signs:
            return None

        candidates = []
        for key, item in self._signs.items():
            templates = self._normalized.get(key, [])
            if not templates:
                continue

            distances = [_sequence_distance(query, template) for template in templates]
            distance = min(distances)
            threshold = self._thresholds.get(key, 0.16)
            candidates.append((distance / threshold, distance, threshold, item))

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate[0])
        normalized_distance, distance, threshold, item = candidates[0]
        if normalized_distance > 1.0:
            return None
        if len(candidates) > 1 and candidates[1][0] - normalized_distance < 0.12:
            return None

        confidence = max(0.35, min(0.99, 0.99 - 0.44 * normalized_distance))
        return {
            'name': item['name'],
            'text': item['text'],
            'confidence': confidence,
            'distance': distance,
            'threshold': threshold,
        }

    def _load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as source:
                document = json.load(source)
            for item in document.get('signs', []):
                name = _clean_text(item.get('name'), 48)
                output_text = _clean_text(item.get('text'), 100)
                samples = item.get('samples', [])
                if not name or not output_text or not isinstance(samples, list):
                    continue
                valid_samples = []
                for blob in samples[-MAX_EXAMPLES_PER_SIGN:]:
                    try:
                        _decode_sequence(blob)
                        valid_samples.append(blob)
                    except (ValueError, TypeError, zlib.error):
                        continue
                if valid_samples:
                    self._signs[name.casefold()] = {
                        'name': name,
                        'text': output_text,
                        'samples': valid_samples,
                    }
        except (OSError, json.JSONDecodeError, TypeError, AttributeError) as error:
            print(f'Personal sign library could not be loaded: {error}')

    def _save(self):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        document = {'version': 1, 'signs': list(self._signs.values())}
        fd, temporary_path = tempfile.mkstemp(prefix='.personal-signs-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                json.dump(document, output, separators=(',', ':'), ensure_ascii=False)
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _refresh_all(self):
        for key in self._signs:
            self._refresh_class(key)

    def _refresh_class(self, key):
        item = self._signs[key]
        templates = []
        for blob in item['samples']:
            normalized = _normalize_sequence(_decode_sequence(blob))
            if normalized is not None:
                templates.append(normalized)
        self._normalized[key] = templates
        spread = _within_class_spread(templates)
        # Class-specific tolerance adapts to the user's enrollment variation,
        # while fixed bounds prevent broad examples from making every clip match.
        self._thresholds[key] = min(0.65, max(0.16, spread * 2.4))


def _clean_text(value, maximum):
    value = ' '.join(str(value or '').strip().split())
    return value[:maximum]


def _encode_sequence(sequence):
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape != SEQUENCE_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError(f'Expected a finite landmark sequence shaped {SEQUENCE_SHAPE}.')
    quantized = np.rint(np.clip(array, -3.2767, 3.2767) * 10000).astype('<i2')
    return base64.b64encode(zlib.compress(quantized.tobytes(), level=6)).decode('ascii')


def _decode_sequence(blob):
    raw = zlib.decompress(base64.b64decode(blob, validate=True))
    expected_bytes = int(np.prod(SEQUENCE_SHAPE)) * np.dtype('<i2').itemsize
    if len(raw) != expected_bytes:
        raise ValueError('Invalid personal sign sample size.')
    return np.frombuffer(raw, dtype='<i2').reshape(SEQUENCE_SHAPE).astype(np.float32) / 10000.0


def _normalize_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != SEQUENCE_SHAPE:
        return None

    visible = np.all(np.isfinite(sequence), axis=2) & ~np.all(np.isclose(sequence, -1.0, atol=1e-4), axis=2)
    normalized = np.zeros_like(sequence)
    for frame_index, frame in enumerate(sequence):
        valid = visible[frame_index]
        if not np.any(valid):
            continue

        # Neck is the body anchor. Fall back to the shoulder midpoint if it was
        # filtered by MediaPipe visibility on this frame.
        if valid[1]:
            origin = frame[1]
        elif valid[2] and valid[5]:
            origin = (frame[2] + frame[5]) * 0.5
        else:
            origin = np.mean(frame[valid[:13]], axis=0) if np.any(valid[:13]) else np.zeros(2)

        shoulder_scale = np.linalg.norm(frame[2] - frame[5]) if valid[2] and valid[5] else 0.0
        if not np.isfinite(shoulder_scale) or shoulder_scale < 0.04:
            shoulder_scale = 0.5
        shoulder_scale = float(np.clip(shoulder_scale, 0.08, 1.5))
        normalized[frame_index, valid] = (frame[valid] - origin) / shoulder_scale

    return normalized, visible


def _within_class_spread(templates):
    if len(templates) < 2:
        return 0.0
    nearest = []
    for index, template in enumerate(templates):
        distances = [
            _sequence_distance(template, other)
            for other_index, other in enumerate(templates)
            if other_index != index
        ]
        nearest.append(min(distances))
    return float(np.median(nearest))


def _sequence_distance(first, second):
    first_values, first_visible = first
    second_values, second_visible = second
    # Reducing 50 frames to 16 keeps matching responsive while allowing the
    # dynamic-time-warp path to handle faster or slower repetitions.
    first_indices = np.linspace(0, len(first_values) - 1, 16, dtype=int)
    second_indices = np.linspace(0, len(second_values) - 1, 16, dtype=int)
    first_values = first_values[first_indices]
    first_visible = first_visible[first_indices]
    second_values = second_values[second_indices]
    second_visible = second_visible[second_indices]

    delta = first_values[:, None, :, :] - second_values[None, :, :, :]
    joint_distance = np.linalg.norm(delta, axis=3)
    common = first_visible[:, None, :] & second_visible[None, :, :]
    counts = common.sum(axis=2)
    costs = np.divide(
        (joint_distance * common).sum(axis=2),
        np.maximum(counts, 1),
        dtype=np.float32,
    )
    costs[counts == 0] = 1.0

    rows, columns = costs.shape
    cumulative = np.full((rows + 1, columns + 1), np.inf, dtype=np.float32)
    path_length = np.zeros((rows + 1, columns + 1), dtype=np.float32)
    cumulative[0, 0] = 0.0
    band = max(3, int(max(rows, columns) * 0.3))

    for row in range(1, rows + 1):
        for column in range(max(1, row - band), min(columns, row + band) + 1):
            options = (
                (cumulative[row - 1, column - 1], path_length[row - 1, column - 1]),
                (cumulative[row - 1, column], path_length[row - 1, column]),
                (cumulative[row, column - 1], path_length[row, column - 1]),
            )
            previous_cost, previous_length = min(options, key=lambda option: option[0])
            cumulative[row, column] = previous_cost + costs[row - 1, column - 1]
            path_length[row, column] = previous_length + 1.0

    length = max(float(path_length[rows, columns]), 1.0)
    return float(cumulative[rows, columns] / length)
