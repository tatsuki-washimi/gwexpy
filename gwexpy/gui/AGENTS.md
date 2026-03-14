# GWexpy GUI (pyaggui) Agent Guidelines

**Summary.**  
This document provides module-specific guidelines for AI Coding Agents working on the GWexpy GUI (`gwexpy/gui/`). This file **extends** the repository-level `AGENTS.md` and MUST be followed in addition to the root guidelines.

**Prerequisites / Links.**  
- Read repository root `AGENTS.md` first.  
- Read `.agent/skills/manage_gui/SKILL.md` for GUI skill conventions.  
- See `.agent/skills/run_tests/reference/gui.md` for GUI test execution details.  
- CI note: GUI tests require an X server (or Xvfb) and proper environment variables; do not assume a desktop.

---

## 1. CRITICAL: PyQt/PySide Thread Safety (must-follow)

- **Never update Qt widgets from a background thread.** All UI mutations must occur on the main (GUI) thread.
- **Always use Signals/Slots for inter-thread communication.** Do not pass direct widget references into worker threads.
- **Long-running tasks must be in worker threads (e.g., `QThread`) or asynchronous workers.** The main thread must remain non-blocking and only perform short UI tasks.
- **Worker lifecycle rules:**
  - Worker must expose clean `start()` / `stop()` / `is_running()` semantics.
  - On stop/teardown, ensure `thread.quit()`, `thread.wait()` (or equivalent) are called and resources freed.
  - Catch and log exceptions inside workers; never let them silently abort.
- **Example (pattern):**
  ```python
  from PyQt5.QtCore import QObject, QThread, pyqtSignal

  class Worker(QObject):
      finished = pyqtSignal()
      progress = pyqtSignal(float)

      def __init__(self):
          super().__init__()
          self._running = True

      def run(self):
          try:
              while self._running:
                  # heavy work / compute
                  value = compute()
                  self.progress.emit(value)
          finally:
              self.finished.emit()

      def stop(self):
          self._running = False

  # Usage on main thread:
  thread = QThread()
  worker = Worker()
  worker.moveToThread(thread)
  thread.started.connect(worker.run)
  worker.finished.connect(thread.quit)
  worker.progress.connect(self.on_progress)  # safe UI update
  thread.start()
````

Use similar pattern and ensure `worker.stop()` and `thread.wait()` in teardown.

---

## 2. Architecture and State Management (guidelines)

* **Separation of concerns**

  * `ui/` : pure view layer (layouts, widgets). No business logic here.
  * `engine.py` / `data_sources.py` : application state, orchestrator, command handlers.
  * `nds/` : NDS2 connections and streaming; treat as long-lived background workers.
  * `plotting/` : plotting adapters and widget wrappers.
  * `loaders/` : asynchronous I/O handlers.
* **Do not put blocking I/O in `ui/`.** File loads and NDS fetches belong to `loaders` or `nds` modules and must be performed in worker threads.
* **State synchronization**

  * Any shared state mutated by background workers must be updated via thread-safe mechanisms (signals, `QMutex` if unavoidable).
  * Keep a single source of truth in `engine.py` and make the UI a read-only consumer of that state.

---

## 3. NDS Connections & Streaming (lifecycle)

* **NDS2 clients must implement:**

  * Robust reconnect logic with exponential backoff.
  * Explicit `start()`, `stop()`, `teardown()` that free sockets and threads.
  * Timeouts and max retry limits as configurable parameters.
* **Testing and mocking.** In unit/CI tests, mock NDS2 responses and network errors. Do not rely on real NDS servers in CI.

---

## 4. Testing and Mocking (CI-friendly)

* **Mock external services**: NDS2, file systems, remote resources.
* **Headless CI**: Use Xvfb or CI-provided headless X for GUI tests. Ensure `DISPLAY` is set in test wrappers.
* **Use `pytest-qt` / `qtbot` for UI assertions**:

  * Use `qtbot.waitSignal` / `qtbot.waitUntil` for asynchronous conditions.
  * Prefer `qtbot.waitSignal(signal, timeout=5000)` over naive `sleep`.
* **Timeout guidance**:

  * Local dev: 5–10s for most UI waits.
  * CI: increase to 20–30s for flaky environments; but target deterministic tests.
* **Scripts**:

  * Run GUI tests via: `../../tests/run_gui_tests.sh`
  * Run NDS GUI tests via: `../../tests/run_gui_nds_tests.sh`
* **Do not write tests that block indefinitely**; always provide sensible timeouts and teardown cleanup.

---

## 5. Resource Management & Memory

* **Always disconnect signals** where appropriate when widgets are destroyed.
* **Stop timers** (`QTimer.stop()`) and background threads in widget `closeEvent` or `teardown` hooks.
* **Avoid circular references** involving QObject / Python closures that prevent garbage collection.

---

## 6. Logging, Metrics, and Diagnostics

* GUI modules must log:

  * Worker start/stop, exception stack traces, NDS connection status, and key lifecycle events.
* Attach compact diagnostic summaries to PRs when changing engine/nds/plotting logic (e.g., sample logs or screenshots).

---

## 7. Accessibility & Style (recommended)

* Prefer high-contrast color sets and scalable fonts.
* Document colors and font sizes used by widgets in a small style guide (optional but recommended).

---

## 8. Agent Skills & Workflows

* Use `.agent/skills/manage_gui/SKILL.md` for GUI skill details.
* For test execution: `.agent/skills/run_tests/reference/gui.md`.
* Before any edit under `gwexpy/gui/`, run the Pre-execution Check (see below).

---

## PRE-EXECUTION CHECK (Agents must confirm)

Before modifying any file in `gwexpy/gui/`, explicitly confirm:

* Will the code execute on the **main UI thread** or a **background worker**? (State which.)
* If background: will it use Signals/Slots and follow the Worker lifecycle pattern? (Yes/No)
* Have you planned for teardown and cleanup on widget close? (Yes/No)
* Will tests be run in headless CI with required mocks? (Yes/No)

---

## PR & Review Rules

* PR titles for agent-created changes: `[AGENT:<skill>] <short description>`.
* Add label `needs-gui-review` for any UI/engine/nds changes.
* Attach diagnostics: test logs, screenshot(s), and a short runtime/teardown checklist.
* Any change altering threading model, NDS lifecycle, or core plotting widgets requires human sign-off.

---

**End of GWexpy GUI Agent Guidelines**
