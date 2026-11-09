# AVT-4: Lip-Sync and Avatar Refinement Report

**Student Name:** Krishna  
**Assigned Role / Task:** AVT-4 – Lip-sync and Avatar Refinement  
**Project:** Talking to Bridges (TTB)  
**Reference Document:** Talking to Bridges Revised Plan v2 (3 November 2026)

---

## 1. Scope & Objective

As assigned in the project plan:
- **AVT-4: Lip-sync and avatar refinement (Krishna)**:
  *Improve lip synchronization and facial animation, test the avatar locally, and provide a simpler fallback if the full avatar is unstable. Output: working avatar presentation with fallback.*
- **INT-1 Test Checklist Verification**:
  *The avatar speaks the answers, and the avatar-off (voice only) mode works.*

### Constraints Respected
- Implemented **only** the assigned AVT-4 responsibility.
- Preserved existing work and interfaces from teammates (AVT-1, AVT-2, AVT-3 by Nagarjun and Krishna).
- Did **not** modify ML models (`analysis/sensor_pipeline.py`, `analysis/features.py`), sensor processing, or backend endpoints (`app/api/routes.py`).
- Kept all changes tested locally without committing or pushing to git.

---

## 2. Implemented Optimizations & Architecture

### 1. Real-Time Audio-Driven Lip Synchronization
- Connected HTMLAudioElement playback to a Web Audio API `AudioContext` and `AnalyserNode` (`fftSize = 512`, `smoothingTimeConstant = 0.5`).
- Extracted multi-band acoustic energy on every animation frame:
  - **Low Band (80–500 Hz)**: Fundamental vocal frequency driving jaw opening and open vowels (`/a/`, `/o/`, `/u/`).
  - **Mid Band (500–2200 Hz)**: Second formant F2 driving mouth width and smile stretch (`/e/`, `/i/`).
  - **High Band (2200–6000 Hz)**: Sibilants and fricatives (`/s/`, `/sh/`, `/f/`, `/t/`) triggering dental contact and mouth height narrowing.
- Combined overall time-domain RMS power with spectral distribution to compute dynamic viseme targets (`mouthOpen`, `mouthWidth`, `jawOpen`, `lipRound`).

### 2. Natural Mouth and Jaw Movements
- **Asymmetric Physics Damping**:
  - Fast attack rate ($\alpha = 0.45$) for crisp articulation on syllable onsets.
  - Slower decay rate ($\alpha = 0.18$) for smooth closure without robotic fluttering.
- **Anatomical Structure**:
  - Cupid's bow contour on the upper lip.
  - Dynamic jaw and chin displacement following vocal volume.
  - Dark oral cavity with gradient depth, upper teeth row, lower dental contour, and responsive tongue contour.

### 3. Smooth Facial Animation, Blinking, and Gaze
- **Stochastic Natural Blinking**:
  - Randomized natural blink intervals (2.5 to 5.5 seconds).
  - Smooth sinusoidal opening and closing easing curve over 160 ms.
  - 12% probability of a rapid double-blink.
- **Micro-Saccades & Cognitive Gaze**:
  - Subtle eye drift every 1.8 to 4.0 seconds.
  - Cognitive upward-right gaze shift during `thinking` state.
  - Direct forward attentive gaze during `listening`.
- **Dynamic Eyebrows & Respiration**:
  - Eyebrows dynamically elevate with speech emphasis peaks.
  - Contemplative furrow during `thinking`.
  - Gentle sinusoidal breathing cycle (~0.25 Hz) preventing static freezing during idle periods.

### 4. Synchronization with Actual Audio Playback
- **Silence & Breath Detection**: Gated by silence threshold (`rms < 0.025`); mouth smoothly relaxes to neutral rest during speech pauses between words.
- **Instant Interruption Handling**: User stop commands (`"stop"`, `"exit"`, `"quit"`, `"goodbye"`), manual stage clicks, and pause events trigger immediate audio pause and mouth closure decay.
- **Lifecycle Events**: Synchronized listeners for `play`, `ended`, `pause`, and `error` events.

### 5. Smooth Interaction State Transitions
- **Continuous Interpolation**: Linear interpolation (`lerp`) blends state weights across `idle`, `listening`, `thinking`, and `speaking` over 250 ms.
- **Dynamic Holographic Aura**:
  - `idle`: Calm cyan/emerald breathing glow.
  - `listening`: Dynamic halo reacting to incoming user microphone volume.
  - `thinking`: Dual orbiting quantum particle arcs (cyan and purple).
  - `speaking`: Outward acoustic soundwave ripples pulsing with assistant speech volume.

### 6. Animation Performance & Resource Usage
- 60 FPS animation loop using `requestAnimationFrame` with delta-time clamping ($\le 50\text{ ms}$).
- Pre-allocated typed arrays (`Uint8Array`, `Float32Array`) in render loop (zero garbage collection overhead).
- Page Visibility API integration (`visibilitychange`) automatically pauses canvas rendering when the browser tab is hidden to eliminate CPU/GPU consumption.

### 7. Reliable Voice-Only & Basic-Avatar Fallback
- **3 Presentation Modes**:
  1. `Animated Avatar`: Full procedural canvas digital face with lip-sync and expressions.
  2. `Classic Orb`: Preserved AVT-1/2/3 glowing orb baseline with speaking pulse animation.
  3. `Voice Only`: Minimalist animated sound-wave visualizer for low-power operation and INT-1 compliance.
- Mode toggle buttons in the UI with preference persistence in `localStorage`.
- Automatic try-catch fallback: if canvas context or Web Audio initialization fails, automatically downgrades to Classic Orb or Voice-Only mode without disrupting speech playback or conversational flow.

---

## 3. Files Modified and Created

| File | Change | Description |
|---|---|---|
| `frontend/voice_ui.html` | Modified | Added presentation mode toggle, procedural canvas avatar, audio sync controller, viseme mapper, and voice-only stage. Preserved all document/CSV upload and conversational features. |
| `tests/test_avatar_refinement.py` | Created | Automated unit and integration test suite verifying all 7 AVT-4 requirements and feature preservation. |
| `docs/krishna_avt4_avatar_refinement.md` | Created | Contribution report and technical documentation for AVT-4. |

---

## 4. Test Verification Results

### 1. Algorithm & Viseme Unit Tests (`scratch/test_avatar_logic.js`)
- Silence threshold test: **Passed** (mouth closes naturally below threshold).
- Vowel `/A/` opening & jaw test: **Passed** (jaw open $> 0.4$, mouth open $> 0.5$).
- Vowel `/EE/` width stretch test: **Passed** (mouth width $> 0.6$).
- Sibilant `/S/` dental narrowing test: **Passed** (mouth open narrowed relative to open vowels).
- Physics damping attack & decay test: **Passed** (attack $= 0.45$, decay $= 0.18$).
- Blinking controller sinusoidal curve test: **Passed** (timing and easing verified).
- **Overall:** `100% Passed`

### 2. HTML & JavaScript DOM Syntax Check (`scratch/validate_voice_ui.js`)
- JavaScript syntax check: **Passed** (0 syntax errors).
- All 17 required DOM element IDs verified present: **Passed**.
- All 28 AVT-4 optimization features verified in code: **Passed**.
- **Overall:** `100% Passed`

### 3. Regression & Feature Preservation Test Suite (`tests/test_avatar_refinement.py`)
```text
Ran 9 tests in 0.004s

OK
```
- `test_voice_ui_file_structure`: Passed
- `test_requirement_1_realtime_audio_driven_lip_sync`: Passed
- `test_requirement_2_natural_mouth_and_jaw_movements`: Passed
- `test_requirement_3_smooth_facial_animations_and_blinking`: Passed
- `test_requirement_4_synchronization_with_actual_audio_playback`: Passed
- `test_requirement_5_smooth_state_transitions`: Passed
- `test_requirement_6_animation_performance_and_resource_usage`: Passed
- `test_requirement_7_reliable_fallback_modes`: Passed
- `test_preservation_of_existing_features`: Passed

