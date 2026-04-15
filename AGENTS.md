# AGENTS.md

## Project overview
This repository contains a desktop application for drone detection built with Python and Qt.

Target stack:
- Python 3.x
- Qt for Python (prefer PySide6 unless explicitly requested otherwise)
- Ultralytics YOLO
- OpenCV

The goal of this project is to provide a practical and responsive demo application for:
- loading images
- loading video files
- optionally using a camera later
- loading a trained model
- running drone detection
- drawing detections in the UI
- providing a smoother and easier YOLO integration path than the C++ prototype

---

## Main goals
1. Keep the application practical and easy to run.
2. Prioritize working YOLO integration and clean demo behavior.
3. Keep UI logic separated from detection logic.
4. Keep the application modular enough for later extension.
5. Prefer incremental, testable changes.

---

## Preferred stack decisions
- Prefer **PySide6** for Qt integration unless the task explicitly asks for PyQt.
- Prefer **Ultralytics YOLO** as the primary model interface.
- Prefer **OpenCV** for frame/image handling.
- Prefer simple local configuration over overengineered dependency injection.
- Prefer code that is easy to debug and run on Windows.

---

## Recommended repository structure
Suggested structure:

- `main.py` — application entry point
- `ui/` — windows, widgets, dialogs, view helpers
- `controllers/` — application flow and UI coordination
- `detection/` — detector abstraction and YOLO integration
- `sources/` — image/video/camera sources
- `utils/` — helpers (conversion, drawing, logging, config)
- `models/` — optional local model metadata/config, not large trained weights
- `assets/` — icons and small UI assets

---

## Architecture rules
- UI classes should remain focused on presentation and user interaction.
- Detection logic should not live directly inside widgets unless absolutely necessary.
- Video/image source management should be isolated from detector implementation.
- Detector-specific logic should be encapsulated behind a detector class or interface.
- Keep drawing/overlay helpers reusable.
- Preserve clear boundaries between:
  - UI
  - application control flow
  - model inference
  - frame source handling

---

## Model integration expectations
This project is intended to work well with:
- `.pt` models through Ultralytics
- possibly `.onnx` later if useful

Preferred first-class backend:
- Ultralytics YOLO in Python

Detection pipeline should support:
- loading a model path from UI
- switching models at runtime if practical
- image inference
- video inference
- confidence threshold control
- NMS threshold control if supported by implementation
- class display names

---

## Performance expectations
- The project should favor practical demo responsiveness over theoretical purity.
- If video inference is slow, prefer:
  - skipping frames
  - running inference on reduced-resolution frames
  - using the latest-frame-only strategy
  - keeping UI responsive
- Multi-threading or worker-thread inference is acceptable and encouraged for video/camera mode.
- Avoid blocking the Qt UI thread during long-running inference.

---

## Coding constraints
- Keep code readable and pragmatic.
- Avoid large rewrites for small features.
- Prefer adding new focused modules over bloating existing files.
- Keep changes incremental and testable.
- Do not overengineer early.
- Use type hints where helpful, but do not add noise just for style.
- Keep Windows usability in mind.
- Do not assume GPU is always available.

---

## UX expectations
At minimum the app should eventually support:
- Open Image
- Open Video
- Load Model
- Run / Play / Pause / Stop as appropriate
- clear detector/model status
- visible detection overlays
- basic status text or status bar
- practical error messages

Useful future additions:
- confidence threshold control
- performance profile selector
- FPS display
- remembered last model path
- remembered last opened media path

---

## What to avoid
- Do not mix all logic into one file.
- Do not put model inference directly into button handlers if it can be avoided.
- Do not freeze the UI with long inference loops.
- Do not break working image inference while adding video inference.
- Do not hardcode temporary debug behavior into production logic.
- Do not add unnecessary external services or frameworks.

---

## Development style
- Prefer vertical slices.
- Keep each task buildable/runnable.
- Preserve existing working behavior unless intentionally changing it.
- Add concise comments only where they improve understanding.
- Be explicit about tradeoffs when choosing speed vs quality.

---

## Good first milestones
1. Load image and run `.pt` model inference.
2. Display detections correctly in the UI.
3. Load video and process frames without freezing the UI.
4. Add performance controls for video.
5. Add optional camera support.

---

## Deliverable expectations for agent work
When implementing a feature:
- keep changes minimal and targeted
- explain what changed
- explain any threading or performance tradeoffs
- explain any new dependency or runtime requirement
- keep the project runnable after the change