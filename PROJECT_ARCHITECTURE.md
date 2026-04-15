# PROJECT_ARCHITECTURE.md

## 1. Project purpose
This project is a desktop demo application for drone detection built with Python and Qt.

Its purpose is to provide a practical and responsive UI for:
- loading images
- loading video files
- optionally using a camera later
- loading a trained drone detection model
- running inference
- displaying detections clearly in the UI

This project prioritizes practical model integration and demo usability over strict native-code purity.

---

## 2. Technology stack
Primary stack:
- Python 3.x
- PySide6 (preferred)
- Ultralytics YOLO
- OpenCV

Secondary optional tools:
- NumPy
- QSettings for persistent UI/app settings
- threading/Qt worker threads for responsive video inference

---

## 3. Core architectural principles
The application must remain modular and understandable.

Key principles:
1. UI code should stay in UI classes.
2. Model inference should be encapsulated in detector classes.
3. Video/image/camera handling should not be mixed into widget code.
4. The UI must remain responsive during inference.
5. The application should support incremental extension without rewriting the whole project.

---

## 4. High-level module structure
Recommended layout:

```text id="q7o3bc"
project_root/
├── main.py
├── AGENTS.md
├── PROJECT_ARCHITECTURE.md
│
├── ui/
│   ├── main_window.py
│   ├── dialogs.py
│   └── widgets/
│
├── controllers/
│   ├── app_controller.py
│   ├── playback_controller.py
│   └── settings_controller.py
│
├── detection/
│   ├── detector_base.py
│   ├── yolo_detector.py
│   ├── detection_types.py
│   └── postprocess.py
│
├── sources/
│   ├── image_source.py
│   ├── video_source.py
│   └── camera_source.py
│
├── workers/
│   ├── inference_worker.py
│   └── video_worker.py
│
├── utils/
│   ├── image_utils.py
│   ├── draw_utils.py
│   ├── qt_utils.py
│   └── config.py
│
├── assets/
│
└── tests/
5. Responsibilities by layer
5.1 UI layer

Files in ui/ should:

create and manage widgets
handle user actions
update displayed images/video
show status/error messages
expose controls for:
open image
open video
load model
playback control
thresholds/profile selection

UI layer should not:

perform model inference directly
parse detections
manage detector internals
implement video processing loops directly in widget logic
5.2 Controller layer

Files in controllers/ should:

coordinate UI, sources, and detector
manage current mode (image/video/camera)
manage model loading state
manage playback state
manage runtime settings
send processed results back to UI

Controllers should represent the application flow.

5.3 Detection layer

Files in detection/ should:

define detection data structures
provide detector abstraction
implement YOLO inference
handle preprocessing/postprocessing
expose clean detector methods such as:
load_model(...)
detect_image(...)
detect_frame(...)

Detection code should not depend on UI widgets.

5.4 Source layer

Files in sources/ should:

load images
open and read video files
optionally support camera streams later

This layer should only care about frame acquisition, not model inference.

5.5 Worker layer

Files in workers/ should:

move long-running inference work off the main UI thread
support bounded-latency processing for video
drop stale frames if needed
keep playback responsive
5.6 Utility layer

Files in utils/ should provide:

drawing helpers
OpenCV ↔ Qt conversions
common geometry helpers
settings/config helpers
convenience functions that do not belong to business logic classes
6. Main application flow
6.1 Image mode
User clicks Open Image
UI opens file dialog
Controller loads the image
Detector runs inference
Detections are drawn
UI displays the final rendered image
Status text is updated
6.2 Video mode
User clicks Open Video
UI opens file dialog
Controller opens the video source
Playback starts through timer/worker-based processing
Frames are read one-by-one
Detector runs on frames in a worker thread or bounded pipeline
Detections are drawn onto rendered frames
UI displays the latest processed frame
Pause/Stop remains responsive
6.3 Camera mode (future)
User opens camera source
Controller starts capture
Frames are processed in the same bounded inference pipeline as video
UI displays live processed frames
7. Detector design

Primary detector:

YoloDetector

Recommended detector interface:

class DetectorBase:
    def load_model(self, model_path: str) -> bool:
        ...
    def is_ready(self) -> bool:
        ...
    def detector_name(self) -> str:
        ...
    def detect(self, frame):
        ...
Detection data model

Use simple reusable structures:

bounding box
confidence
class id
class label

Prefer a dedicated detection type module.

8. Performance strategy

Because this project prioritizes practical usability:

avoid blocking inference on the UI thread
use frame skipping if needed
use reduced inference resolution for video if needed
preserve full-quality display if practical
keep the latest-frame-only strategy available
show FPS and current performance mode when useful

Performance is allowed to trade off some completeness in video mode in exchange for responsiveness.

9. Settings and configuration

Settings should eventually include:

last selected model path
last opened media path
confidence threshold
optional NMS threshold
performance profile
preferred source mode

Recommended implementation:

QSettings
10. Error handling expectations

The app should fail clearly and safely when:

no model is loaded
model path is invalid
image/video cannot be opened
inference fails
source ends or disconnects

Error handling should:

preserve app stability
avoid crashing the UI
provide visible status text
11. What is explicitly out of scope for early milestones

These may come later, but should not be forced too early:

distributed inference
remote inference services
cloud integration
advanced plugin architecture
overengineered dependency injection
packaging/installer automation
camera tracking and autopilot integration
12. Recommended milestone order
Milestone 1
open image
load model
run inference on image
show bbox
Milestone 2
open video
play/pause/stop
run frame-by-frame inference
keep UI responsive
Milestone 3
thresholds
FPS
performance profiles
settings persistence
Milestone 4
optional camera support
improved UX polish
13. Architecture guardrails

When extending the project:

do not move inference code into UI classes
do not use blocking loops in the main thread
do not rewrite the whole project to add a small feature
do not mix source acquisition and detector internals in one file
do not degrade working image inference while adding video features
