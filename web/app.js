import {
  PoseLandmarker,
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const canvas = document.getElementById("vj-canvas");
const ctx = canvas.getContext("2d", { alpha: false });
const videoEl = document.getElementById("camera");
const startBtn = document.getElementById("start-btn");
const statusText = document.getElementById("status-text");
const showVideoToggle = document.getElementById("show-video");

const CONFIG = {
  fftSize: 2048,
  smoothing: 0.82,
  barCount: 96,
  motionSmoothing: 0.82,
  fingerTrailLength: 220,
  blinkCooldownMs: 1200,
  levelScale: 320,
  levelExponent: 0.56,
  bandGain: 20,
  bandExponent: 0.7,
  cameraBlend: false,
  hudEnabled: false,
  fluxHistorySize: 128,
  fluxWindow: 24,
  fluxMinWindow: 12,
  fluxSensitivity: 1.65,
  fluxExponent: 1.18,
  fluxDeltaFloor: 0.006,
  fluxPulseGain: 2.6,
  beatDeltaGain: 5.4,
  beatMinIntervalMs: 200,
  beatMaxIntervalMs: 1400,
  beatLevelFloor: 0.16,
  beatEnergyFloor: 0.18,
  beatSilenceResetMs: 2600,
  bpmSmoothing: 0.28,
  bpmWindow: 12,
  paletteMinHoldMs: 2000,
  paletteBeatsPerCycle: 8,
  paletteQuietIntensity: 0.12,
  paletteQuietResetMs: 5200,
  paletteBeatPulseGate: 0.6,
};

const COLOR_THEMES = [
  {
    name: "Glacier Rift",
    baseHue: 210,
    accentHue: 180,
    glowHue: 200,
    bgDark: "#0a1420",
  },
  {
    name: "Infra Noir",
    baseHue: 330,
    accentHue: 0,
    glowHue: 350,
    bgDark: "#120910",
  },
  {
    name: "Electric Alloy",
    baseHue: 260,
    accentHue: 120,
    glowHue: 280,
    bgDark: "#0b0818",
  },
  {
    name: "Chrome Tide",
    baseHue: 190,
    accentHue: 40,
    glowHue: 210,
    bgDark: "#050d18",
  },
];

class AudioAnalyser {
  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.freqData = null;
    this.timeData = null;
    this.rms = 0;
    this.rmsAverage = 0.001;
    this.level = 0;
    this.levelAverage = 0;
    this.beatPulse = 0;
    this.brightness = 0;
    this.lastBeatTime = 0;
    this.beatIntervals = [];
    this.bpm = 0;
    this.bandEnergy = { low: 0, mid: 0, high: 0 };
    this.prevSpectrum = null;
    this.fluxHistory = [];
    this.prevFlux = 0;
    this.lastAudioActive = 0;
    this.beatDetected = false;
  }

  async init(stream) {
    this.audioContext = new AudioContext();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = CONFIG.fftSize;
    this.analyser.smoothingTimeConstant = CONFIG.smoothing;

    const source = this.audioContext.createMediaStreamSource(stream);
    source.connect(this.analyser);

    this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
    this.timeData = new Float32Array(this.analyser.fftSize);
  }

  update() {
    if (!this.analyser) {
      return this._emptyState();
    }

    this.analyser.getByteFrequencyData(this.freqData);
    this.analyser.getFloatTimeDomainData(this.timeData);

    let sumSquares = 0;
    for (let i = 0; i < this.timeData.length; i += 1) {
      const sample = this.timeData[i];
      sumSquares += sample * sample;
    }
    const nextRms = Math.sqrt(sumSquares / this.timeData.length);
    this.rms = lerp(this.rms, nextRms, 0.22);
    this.rmsAverage = lerp(this.rmsAverage, nextRms, 0.05);

    this._updateBands();
    this._updateBrightness();

    const normalized = clamp(Math.pow(nextRms * CONFIG.levelScale, CONFIG.levelExponent), 0, 1);
    this.levelAverage = lerp(this.levelAverage, normalized, 0.06);
    this.level = lerp(this.level, normalized, 0.4);

    const frameNow = performance.now();
    if (normalized > 0.035) {
      this.lastAudioActive = frameNow;
    }

    const flux = this._calculateFlux();
    const { threshold: fluxThreshold, delta: fluxDelta } = this._updateFluxHistory(flux);

    const beatDelta = Math.max(0, normalized - this.levelAverage * 1.01);
    const energyContribution = normalized * (
      this.bandEnergy.low * 0.24 +
      this.bandEnergy.mid * 0.18 +
      this.bandEnergy.high * 0.12
    );
    const targetBeatPulse = clamp(
      beatDelta * CONFIG.beatDeltaGain +
        Math.max(0, fluxDelta) * CONFIG.fluxPulseGain +
        energyContribution,
      0,
      1,
    );
    this.beatPulse = lerp(this.beatPulse, targetBeatPulse, 0.36);

    const beatDetected = this._detectBeat(
      normalized,
      flux,
      fluxThreshold,
      fluxDelta,
      this.bandEnergy,
      frameNow,
    );
    if (beatDetected) {
      this.beatPulse = Math.min(1, this.beatPulse + 0.34);
    }
    this.beatDetected = beatDetected;

    const frequencies = new Float32Array(CONFIG.barCount);
    const step = Math.max(1, Math.floor(this.freqData.length / CONFIG.barCount));
    for (let i = 0; i < CONFIG.barCount; i += 1) {
      const value = this.freqData[i * step] / 255;
      frequencies[i] = Math.pow(value, 0.62);
    }

    return {
      rms: this.rms,
      rmsAverage: this.rmsAverage,
      level: this.level,
      beatPulse: this.beatPulse,
      brightness: this.brightness,
      bpm: this.bpm,
      beatDetected,
      bandEnergy: this.bandEnergy,
      frequencies,
    };
  }

  _updateBands() {
    const len = this.freqData.length;
    const lowEnd = Math.floor(len * 0.12);
    const midEnd = Math.floor(len * 0.45);
    let lowSum = 0;
    let midSum = 0;
    let highSum = 0;

    for (let i = 0; i < len; i += 1) {
      const value = this.freqData[i] / 255;
      if (i < lowEnd) {
        lowSum += value;
      } else if (i < midEnd) {
        midSum += value;
      } else {
        highSum += value;
      }
    }

    const lowCount = Math.max(lowEnd, 1);
    const midCount = Math.max(midEnd - lowEnd, 1);
    const highCount = Math.max(len - midEnd, 1);

    const convert = (sum, count) => {
      const avg = sum / count;
      return Math.pow(clamp(avg * CONFIG.bandGain, 0, 1), CONFIG.bandExponent);
    };

    this.bandEnergy = {
      low: convert(lowSum, lowCount),
      mid: convert(midSum, midCount),
      high: convert(highSum, highCount),
    };
  }

  _updateBrightness() {
    const { low, mid, high } = this.bandEnergy;
    const total = low + mid + high + 1e-6;
    this.brightness = clamp((mid * 0.35 + high * 0.65) / total, 0, 1);
  }

  _calculateFlux() {
    if (!this.prevSpectrum) {
      this.prevSpectrum = new Float32Array(this.freqData.length);
      for (let i = 0; i < this.freqData.length; i += 1) {
        this.prevSpectrum[i] = Math.pow(this.freqData[i] / 255, CONFIG.fluxExponent);
      }
      return 0;
    }

    let flux = 0;
    for (let i = 0; i < this.freqData.length; i += 1) {
      const magnitude = Math.pow(this.freqData[i] / 255, CONFIG.fluxExponent);
      const diff = magnitude - this.prevSpectrum[i];
      if (diff > 0) {
        flux += diff;
      }
      this.prevSpectrum[i] = magnitude;
    }
    return flux / this.freqData.length;
  }

  _updateFluxHistory(flux) {
    const history = this.fluxHistory;
    const windowSize = Math.min(history.length, CONFIG.fluxWindow);
    let threshold = Infinity;

    if (windowSize >= CONFIG.fluxMinWindow) {
      let sum = 0;
      for (let i = history.length - windowSize; i < history.length; i += 1) {
        sum += history[i];
      }
      const mean = sum / windowSize;
      let deviation = 0;
      for (let i = history.length - windowSize; i < history.length; i += 1) {
        deviation += Math.abs(history[i] - mean);
      }
      deviation /= windowSize;
      threshold = mean + deviation * CONFIG.fluxSensitivity;
    }

    history.push(flux);
    if (history.length > CONFIG.fluxHistorySize) {
      history.shift();
    }

    const delta = threshold === Infinity ? 0 : flux - threshold;
    return { threshold, delta };
  }

  _detectBeat(levelSample, flux, threshold, delta, bandEnergy, frameNow) {
    const fluxActive = threshold !== Infinity && delta > CONFIG.fluxDeltaFloor;
    const rising = flux > this.prevFlux;
    const energyScore = bandEnergy.low * 0.5 + bandEnergy.mid * 0.35 + bandEnergy.high * 0.25;
    const levelOk = levelSample > CONFIG.beatLevelFloor;
    const energyOk = energyScore > CONFIG.beatEnergyFloor;
    let detected = false;

    if (fluxActive && rising && (levelOk || energyOk)) {
      const interval = frameNow - this.lastBeatTime;
      if (this.lastBeatTime === 0 || interval > CONFIG.beatMinIntervalMs) {
        detected = true;
        if (this.lastBeatTime > 0 && interval < CONFIG.beatMaxIntervalMs) {
          this.beatIntervals.push(interval);
          if (this.beatIntervals.length > CONFIG.bpmWindow) {
            this.beatIntervals.shift();
          }
          const avg = this.beatIntervals.reduce((a, b) => a + b, 0) / this.beatIntervals.length;
          const targetBpm = clamp(60000 / Math.max(avg, 1), 50, 200);
          this.bpm = this.bpm === 0 ? Math.round(targetBpm) : Math.round(lerp(this.bpm, targetBpm, CONFIG.bpmSmoothing));
        }
        this.lastBeatTime = frameNow;
      }
    }

    if (!detected) {
      const sinceBeat = frameNow - this.lastBeatTime;
      const silence = frameNow - this.lastAudioActive > CONFIG.beatSilenceResetMs;
      if ((this.lastBeatTime > 0 && sinceBeat > CONFIG.beatSilenceResetMs) || silence) {
        this.beatIntervals = [];
        this.lastBeatTime = 0;
        this.bpm = 0;
      }
    }

    this.prevFlux = flux;
    return detected;
  }

  _emptyState() {
    return {
      rms: 0,
      rmsAverage: 0,
      level: 0,
      beatPulse: 0,
      brightness: 0,
      bpm: 0,
      beatDetected: false,
      bandEnergy: { low: 0, mid: 0, high: 0 },
      frequencies: new Float32Array(CONFIG.barCount),
    };
  }
}

class PoseTracker {
  constructor() {
    this.previousEnergy = 0;
  }

  update(result) {
    if (!result || !result.landmarks || result.landmarks.length === 0) {
      this.previousEnergy = lerp(this.previousEnergy, 0, CONFIG.motionSmoothing);
      return {
        motionEnergy: this.previousEnergy,
        centroid: null,
      };
    }

    const landmarks = result.landmarks[0];
    let energy = 0;
    let centroidX = 0;
    let centroidY = 0;

    landmarks.forEach((lm) => {
      centroidX += lm.x;
      centroidY += lm.y;
    });

    const count = landmarks.length || 1;
    centroidX /= count;
    centroidY /= count;

    if (result.worldLandmarks && result.worldLandmarks.length) {
      const indices = [11, 12, 13, 14, 15, 16];
      indices.forEach((idx) => {
        const lm = result.worldLandmarks[0][idx];
        const mag = Math.hypot(lm.x, lm.y, lm.z);
        energy += mag;
      });
      energy /= indices.length;
    }

    this.previousEnergy = lerp(this.previousEnergy, energy, 0.18);
    return {
      motionEnergy: this.previousEnergy,
      centroid: { x: centroidX, y: centroidY },
    };
  }
}

class HandGestureTracker {
  constructor() {
    this.previousHands = new Map();
    this.paths = {
      Left: [],
      Right: [],
    };
  }

  update(result) {
    const hands = [];
    if (!result || !result.landmarks || result.landmarks.length === 0) {
      this._fadeTrails();
      this.previousHands.clear();
      return { hands, paths: this.paths, summary: this._buildSummary(hands) };
    }

    result.landmarks.forEach((landmarks, index) => {
      const handedness = result.handedness?.[index]?.categoryName || "Unknown";
      const handId = `${handedness}-${index}`;
      const prev = this.previousHands.get(handId);

      const normalized = this._computeMetrics(landmarks, handedness, prev);
      hands.push(normalized);
      this.previousHands.set(handId, normalized);
      this._pushTrail(handedness, normalized.indexTip);
    });

    this._fadeTrails();
    return { hands, paths: this.paths, summary: this._buildSummary(hands) };
  }

  _computeMetrics(landmarks, handedness, prev) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const pinkyTip = landmarks[20];
    const wrist = landmarks[0];

    const pinchDistance = distance2D(thumbTip, indexTip);
    const pinch = clamp(1 - pinchDistance * 4, 0, 1);

    const extensionPairs = [
      [8, 6],
      [12, 10],
      [16, 14],
      [20, 18],
    ];
    let extensionSum = 0;
    extensionPairs.forEach(([tipIdx, pipIdx]) => {
      extensionSum += distance2D(landmarks[tipIdx], landmarks[pipIdx]);
    });
    const extension = clamp((extensionSum / extensionPairs.length) * 8, 0, 1);

    const direction = {
      x: indexTip.x - wrist.x,
      y: indexTip.y - wrist.y,
    };

    const velocity = prev
      ? {
          x: (indexTip.x - prev.indexTip.x) * 60,
          y: (indexTip.y - prev.indexTip.y) * 60,
        }
      : { x: 0, y: 0 };
    const speed = Math.hypot(velocity.x, velocity.y);

    const spread = clamp(distance2D(indexTip, pinkyTip) * 3, 0, 1);
    const gesture = spread > 0.45 && extension > 0.45 && pinch < 0.65 ? 'open' : 'closed';

    return {
      handedness,
      pinch,
      spread,
      extension,
      gesture,
      indexTip: { x: indexTip.x, y: indexTip.y },
      thumbTip: { x: thumbTip.x, y: thumbTip.y },
      palmCenter: midpoint(wrist, indexTip),
      direction,
      velocity,
      speed,
      rawLandmarks: landmarks,
    };
  }

  _pushTrail(handedness, point) {
    if (!point) return;
    const list = this.paths[handedness] || (this.paths[handedness] = []);
    list.push({ x: point.x, y: point.y, life: 1 });
    if (list.length > CONFIG.fingerTrailLength) {
      list.shift();
    }
  }

  _fadeTrails() {
    Object.values(this.paths).forEach((trail) => {
      for (let i = trail.length - 1; i >= 0; i -= 1) {
        trail[i].life -= 0.03;
        if (trail[i].life <= 0) {
          trail.splice(i, 1);
        }
      }
    });
  }

  _buildSummary(hands) {
    const left = hands.find((hand) => hand.handedness === "Left");
    const right = hands.find((hand) => hand.handedness === "Right");
    let distance = 0;
    let angle = 0;
    if (left && right) {
      distance = distance2D(left.palmCenter, right.palmCenter);
      angle = Math.atan2(right.palmCenter.y - left.palmCenter.y, right.palmCenter.x - left.palmCenter.x);
    }

    return {
      handCount: hands.length,
      left,
      right,
      distance,
      angle,
    };
  }
}

class FaceExpressionTracker {
  constructor() {
    this.prevBlinkTime = 0;
    this.isBlinking = false;
    this.blinkCount = 0;
  }

  update(result) {
    if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
      this.isBlinking = false;
      return this._empty();
    }

    const landmarks = result.faceLandmarks[0];
    const metrics = this._computeMetrics(landmarks);
    this._updateBlink(metrics.eyeOpenness);
    return {
      ...metrics,
      blinkCount: this.blinkCount,
      blinkTriggered: metrics.eyeOpenness < 0.22 && !this.isBlinking,
    };
  }

  _computeMetrics(landmarks) {
    const faceHeight = distance2D(landmarks[10], landmarks[152]) || 1;

    const leftEyeOpen = this._eyeOpenness(landmarks, 159, 145, 33, 133);
    const rightEyeOpen = this._eyeOpenness(landmarks, 386, 374, 362, 263);
    const eyeOpenness = clamp((leftEyeOpen + rightEyeOpen) * 0.5, 0, 1);

    const leftBrow = distance2D(landmarks[65], landmarks[159]) / faceHeight;
    const rightBrow = distance2D(landmarks[295], landmarks[386]) / faceHeight;
    const browRaise = clamp((leftBrow + rightBrow) * 2.8, 0, 1);

    const mouthWidth = distance2D(landmarks[61], landmarks[291]) / faceHeight;
    const mouthHeight = distance2D(landmarks[13], landmarks[14]) / faceHeight;
    const smile = clamp((mouthWidth - 0.25) * 3, 0, 1);
    const mouthOpen = clamp((mouthHeight - 0.02) * 12, 0, 1);

    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const roll = Math.atan2(rightCheek.y - leftCheek.y, rightCheek.x - leftCheek.x);

    return {
      eyeOpenness,
      browRaise,
      smile,
      mouthOpen,
      headTilt: { roll },
    };
  }

  _eyeOpenness(landmarks, upperIdx, lowerIdx, leftIdx, rightIdx) {
    const vertical = distance2D(landmarks[upperIdx], landmarks[lowerIdx]);
    const horizontal = distance2D(landmarks[leftIdx], landmarks[rightIdx]) + 1e-6;
    return clamp((vertical / horizontal) * 3, 0, 1);
  }

  _updateBlink(openness) {
    const now = performance.now();
    const currentlyBlinking = openness < 0.18;
    if (currentlyBlinking && !this.isBlinking && now - this.prevBlinkTime > CONFIG.blinkCooldownMs) {
      this.blinkCount += 1;
      this.prevBlinkTime = now;
    }
    this.isBlinking = currentlyBlinking;
  }

  _empty() {
    return {
      eyeOpenness: 0,
      browRaise: 0,
      smile: 0,
      mouthOpen: 0,
      headTilt: { roll: 0 },
      blinkCount: this.blinkCount,
      blinkTriggered: false,
    };
  }
}

class PaletteController {
  constructor() {
    this.index = 0;
    this.lastSwitch = performance.now();
    this.beatCount = 0;
    this.lastBeatAt = 0;
  }

  update({ audioIntensity, beatPulse, beatDetected, bpm }) {
    const now = performance.now();

    if (beatDetected) {
      this.lastBeatAt = now;
      this.beatCount += 1;
    }

    const beatDuration = bpm > 0 ? 60000 / bpm : 0;
    const targetHold = beatDuration > 0 ? beatDuration * CONFIG.paletteBeatsPerCycle : CONFIG.paletteMinHoldMs;
    const sinceSwitch = now - this.lastSwitch;
    const holdSatisfied = sinceSwitch >= Math.max(CONFIG.paletteMinHoldMs, targetHold);
    const strongPulse = beatPulse > CONFIG.paletteBeatPulseGate || beatDetected;

    if (holdSatisfied && strongPulse && this.beatCount >= CONFIG.paletteBeatsPerCycle) {
      this.index = (this.index + 1) % COLOR_THEMES.length;
      this.lastSwitch = now;
      this.beatCount = 0;
    }

    const timeSinceBeat = now - this.lastBeatAt;
    const quiet = audioIntensity < CONFIG.paletteQuietIntensity;
    if (quiet && holdSatisfied && timeSinceBeat > CONFIG.paletteQuietResetMs) {
      this.index = 0;
      this.lastSwitch = now;
      this.beatCount = 0;
    }

    return COLOR_THEMES[this.index];
  }
}

class CoolMonoScene {
  constructor() {
    this.glitchSeed = 0;
    this.rings = [];
    this.prevGestures = new Map();
    this.lastEmitTime = new Map();
  }

  render(ctx, shared) {
    const { width, height } = ctx.canvas;
    const { audio, gestures, hands, face, palette, audioIntensity } = shared;
    const intensity = audioIntensity;
    const pulse = audio.beatPulse ?? 0;
    const time = performance.now() * 0.001;

    ctx.fillStyle = palette.bgDark;
    ctx.fillRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, `hsla(${palette.baseHue + intensity * 35}, 70%, ${18 + intensity * 32}%, ${0.45 + intensity * 0.3})`);
    gradient.addColorStop(1, `hsla(${palette.accentHue + audio.bandEnergy.mid * 60}, 70%, ${24 + intensity * 36}%, ${0.38 + audio.bandEnergy.high * 0.35})`);
    ctx.globalCompositeOperation = 'screen';
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    const ringRadius = 140 + intensity * 220 + pulse * 260;
    const ringGradient = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, ringRadius);
    ringGradient.addColorStop(0, `hsla(${palette.glowHue + pulse * 40}, 95%, ${55 + intensity * 25}%, ${0.25 + pulse * 0.4})`);
    ringGradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = ringGradient;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(width / 2, height / 2);
    const scale = 1 + intensity * 0.35 + audio.bandEnergy.low * 0.2 + (face?.browRaise ?? 0) * 0.15;
    ctx.scale(scale, scale);
    ctx.rotate(((face?.headTilt?.roll) ?? 0) * 0.6 + intensity * 0.08);

    const gridSize = Math.max(24, 60 - intensity * 36 - audio.bandEnergy.low * 26);
    ctx.strokeStyle = `hsla(${palette.baseHue}, 18%, 80%, ${0.05 + intensity * 0.28 + audio.bandEnergy.mid * 0.23})`;
    for (let x = -width; x <= width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, -height);
      ctx.lineTo(x + gestures.panX * 70, height);
      ctx.stroke();
    }
    for (let y = -height; y <= height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(-width, y);
      ctx.lineTo(width, y + gestures.panY * 70);
      ctx.stroke();
    }
    ctx.restore();

    this.glitchSeed += 0.012 + intensity * 0.55 + audio.bandEnergy.high * 0.32;
    ctx.save();
    ctx.globalCompositeOperation = 'lighten';
    const slices = 16;
    for (let i = 0; i < slices; i += 1) {
      const sliceHeight = height / slices;
      const y = i * sliceHeight;
      const offset = Math.sin(this.glitchSeed + i * 1.3 + time * 0.3) * (audio.bandEnergy.high + intensity * 0.45) * 90;
      ctx.drawImage(canvas, 0, y, width, sliceHeight, offset, y, width, sliceHeight);
    }
    ctx.restore();

    this._updateRings(width, height, hands, intensity, palette);
    this._drawRings(ctx, palette);

    if (pulse > 0.25) {
      ctx.save();
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = `hsla(${palette.glowHue + pulse * 60}, 100%, 65%, ${0.08 + pulse * 0.35})`;
      ctx.fillRect(0, 0, width, height);
      ctx.restore();
    }

    ctx.globalCompositeOperation = 'source-over';
  }

  _updateRings(width, height, hands, intensity, palette) {
    const now = performance.now();
    const decayBase = 0.016;

    this.rings = this.rings
      .map((ring) => ({
        ...ring,
        radius: ring.radius + ring.speed,
        life: ring.life - ring.decay,
        alpha: Math.max(0, ring.life),
      }))
      .filter((ring) => ring.life > 0);

    const activeHands = new Set();
    hands.hands.forEach((hand) => {
      const id = hand.handedness;
      activeHands.add(id);
      const extension = hand.extension ?? 0;
      const gesture = hand.gesture ?? 'closed';
      const lastEmit = this.lastEmitTime.get(id) ?? 0;
      const emitInterval = gesture === 'open' ? 180 : 240;
      const shouldEmit = now - lastEmit > emitInterval;

      const tip = hand.indexTip;
      if (!tip) return;
      const x = tip.x * width;
      const y = tip.y * height;

      if (gesture === 'open' && shouldEmit) {
        this._spawnRing(x, y, 'open', intensity, palette);
        this.lastEmitTime.set(id, now);
      } else if (gesture === 'closed' && extension < 0.4 && shouldEmit) {
        this._spawnRing(x, y, 'closed', intensity, palette);
        this.lastEmitTime.set(id, now);
      }

      this.prevGestures.set(id, gesture);
    });

    Array.from(this.prevGestures.keys()).forEach((id) => {
      if (!activeHands.has(id)) {
        this.prevGestures.delete(id);
        this.lastEmitTime.delete(id);
      }
    });
  }

  _spawnRing(x, y, type, intensity, palette) {
    const ring = {
      x,
      y,
      radius: type === 'open' ? 30 : 18,
      speed: type === 'open' ? 6 + intensity * 18 : 10 + intensity * 22,
      life: 1,
      decay: type === 'open' ? 0.016 : 0.02,
      alpha: 1,
      type,
      hue: type === 'open' ? palette.glowHue : (palette.baseHue + 180) % 360,
      lineWidth: type === 'open' ? 8 + intensity * 6 : 4 + intensity * 4,
    };
    this.rings.push(ring);
  }

  _drawRings(ctx, palette) {
    ctx.save();
    this.rings.forEach((ring) => {
      if (ring.type === 'open') {
        ctx.globalCompositeOperation = 'lighter';
        ctx.lineWidth = ring.lineWidth;
        ctx.strokeStyle = `hsla(${ring.hue}, 100%, 70%, ${0.12 + ring.alpha * 0.35})`;
        ctx.beginPath();
        ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.globalCompositeOperation = 'screen';
        ctx.setLineDash([14, 12]);
        ctx.lineWidth = ring.lineWidth;
        ctx.strokeStyle = `hsla(${ring.hue}, 90%, 65%, ${0.1 + ring.alpha * 0.4})`;
        ctx.beginPath();
        ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
    ctx.restore();
  }
}
class Renderer {
  constructor(ctx, videoEl) {
    this.ctx = ctx;
    this.videoEl = videoEl;
    this.scene = new CoolMonoScene();
    this.paletteController = new PaletteController();
    this.previousFrame = document.createElement("canvas");
    this.previousFrame.width = ctx.canvas.width;
    this.previousFrame.height = ctx.canvas.height;
    this.prevCtx = this.previousFrame.getContext("2d");
  }

  draw(shared) {
    const ctx = this.ctx;
    const audio = shared.audio ?? {};
    const audioIntensity = clamp(
      (audio.level ?? 0) * 1.5 +
        (audio.bandEnergy?.low ?? 0) * 0.35 +
        (audio.bandEnergy?.mid ?? 0) * 0.25 +
        (audio.bandEnergy?.high ?? 0) * 0.2,
      0,
      1,
    );
    shared.audioIntensity = audioIntensity;

    const palette = this.paletteController.update({
      audioIntensity,
      beatPulse: audio.beatPulse ?? 0,
      beatDetected: audio.beatDetected ?? false,
      bpm: audio.bpm ?? 0,
    });
    shared.palette = palette;

    this.scene.render(ctx, shared);

    if (CONFIG.cameraBlend && this.videoEl && this.videoEl.readyState >= 2) {
      const glowAmount = 0.18 + audioIntensity * 0.22;
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ctx.globalAlpha = glowAmount;
      ctx.filter = `blur(${18 - audioIntensity * 14}px) saturate(${1.4 + audioIntensity * 1.6})`;
      ctx.drawImage(this.videoEl, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    this._drawHUD(shared, palette, audioIntensity);
  }

  _drawHUD(shared, palette, audioIntensity) {
    if (!CONFIG.hudEnabled) {
      return;
    }

    const { width } = this.ctx.canvas;
    const lines = [
      `Energy: ${(audioIntensity * 100).toFixed(0)}%`,
      `Palette: ${palette.name}`,
      `BPM: ${shared.audio?.bpm || "--"}`,
      `Smile: ${((shared.face?.smile ?? 0) * 100).toFixed(0)}%`,
      `Hands: ${shared.hands.summary.handCount}`,
    ];

    this.ctx.save();
    this.ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
    this.ctx.fillRect(width - 230, 24, 206, 140);
    this.ctx.fillStyle = "rgba(235, 240, 250, 0.88)";
    this.ctx.font = "14px 'Segoe UI', sans-serif";
    lines.forEach((line, index) => {
      this.ctx.fillText(line, width - 214, 48 + index * 20);
    });
    this.ctx.restore();
  }
}

function computeGestureSummary(hands, face) {
  const panX = ((hands.summary.left?.direction.x ?? 0) + (hands.summary.right?.direction.x ?? 0)) * 0.5;
  const panY = ((hands.summary.left?.direction.y ?? 0) + (hands.summary.right?.direction.y ?? 0)) * 0.5;
  const zoom = clamp(((hands.summary.distance || 0) - 0.1) * 2, 0, 1);
  const headTilt = face?.headTilt?.roll ?? 0;
  return { panX, panY, zoom, headTilt };
}

function distance2D(a, b) {
  const dx = (a?.x ?? 0) - (b?.x ?? 0);
  const dy = (a?.y ?? 0) - (b?.y ?? 0);
  return Math.hypot(dx, dy);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

const audioAnalyser = new AudioAnalyser();
const poseTracker = new PoseTracker();
const handTracker = new HandGestureTracker();
const faceTracker = new FaceExpressionTracker();
const renderer = new Renderer(ctx, videoEl);

const MODEL_CANDIDATES = {
  pose: [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  ],
  hands: [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker_lite/float16/latest/hand_landmarker_lite.task",
  ],
  face: [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  ],
};

const modelBuffers = {};

let filesetResolver = null;
let poseLandmarker = null;
let handLandmarker = null;
let faceLandmarker = null;
let animationId = null;
let running = false;

async function fetchModelBuffer(key) {
  if (modelBuffers[key]) return modelBuffers[key];
  const urls = MODEL_CANDIDATES[key] || [];
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to fetch ${url}: ${response.status}`);
        continue;
      }
      const buffer = new Uint8Array(await response.arrayBuffer());
      modelBuffers[key] = buffer;
      return buffer;
    } catch (error) {
      console.warn(`Error fetching model ${url}:`, error);
    }
  }
  throw new Error(`Could not download ${key} model from candidates.`);
}

async function initTasks() {
  if (!filesetResolver) {
    filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm",
    );
  }

  try {
    if (!poseLandmarker) {
      const poseModel = await fetchModelBuffer("pose");
      poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetBuffer: poseModel },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }

    if (!handLandmarker) {
      const handModel = await fetchModelBuffer("hands");
      handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetBuffer: handModel },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });
    }

    if (!faceLandmarker) {
      const faceModel = await fetchModelBuffer("face");
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetBuffer: faceModel },
        runningMode: "VIDEO",
        outputFaceBlendshapes: false,
        numFaces: 1,
      });
    }
  } catch (error) {
    console.error(error);
    throw new Error(`MediaPipe model load failed: ${error.message ?? error}`);
  }
}

async function startExperience() {
  startBtn.disabled = true;
  statusText.textContent = "初期化中...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
    });

    await audioAnalyser.init(stream);

    videoEl.srcObject = stream;
    await videoEl.play();

    statusText.textContent = "MediaPipe モデルをロード中...";
    try {
      await initTasks();
    } catch (error) {
      statusText.textContent = error.message || String(error);
      startBtn.disabled = false;
      startBtn.textContent = "Retry";
      return;
    }

    running = true;
    statusText.textContent = "Running";
    startBtn.textContent = "Running";
    startBtn.blur();
    startBtn.disabled = true;

    loop();
  } catch (error) {
    console.error(error);
    statusText.textContent = `Error: ${error.message ?? error}`;
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
  }
}

function loop() {
  if (!running) return;

  const audioState = audioAnalyser.update();
  const frameTime = performance.now();

  let poseState = { motionEnergy: 0, centroid: null };
  if (poseLandmarker && videoEl.readyState >= 2) {
    const poseResult = poseLandmarker.detectForVideo(videoEl, frameTime);
    poseState = poseTracker.update(poseResult);
  }

  let handState = { hands: [], paths: handTracker.paths, summary: handTracker._buildSummary([]) };
  if (handLandmarker && videoEl.readyState >= 2) {
    const handResult = handLandmarker.detectForVideo(videoEl, frameTime);
    handState = handTracker.update(handResult);
  }

  let faceState = faceTracker._empty();
  if (faceLandmarker && videoEl.readyState >= 2) {
    const faceResult = faceLandmarker.detectForVideo(videoEl, frameTime);
    faceState = faceTracker.update(faceResult);
  }

  const gestures = computeGestureSummary(handState, faceState);

  const shared = {
    audio: audioState,
    pose: poseState,
    hands: handState,
    face: faceState,
    gestures,
  };

  renderer.draw(shared);
  animationId = requestAnimationFrame(loop);
}

function stopExperience() {
  running = false;
  if (animationId) {
    cancelAnimationFrame(animationId);
  }
}

startBtn.addEventListener("click", () => {
  if (!running) {
    startExperience();
  }
});

showVideoToggle.addEventListener("change", (event) => {
  if (event.target.checked) {
    videoEl.classList.add("visible");
  } else {
    videoEl.classList.remove("visible");
  }
});

window.addEventListener("beforeunload", () => {
  stopExperience();
  if (videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach((track) => track.stop());
  }
});

statusText.textContent = "Ready. Click Start to begin.";
