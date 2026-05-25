from __future__ import annotations

from typing import Callable

TEXTUAL_AVAILABLE = False

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, RichLog, Static

    TEXTUAL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    App = object  # type: ignore
    ComposeResult = object  # type: ignore
    Horizontal = object  # type: ignore
    Vertical = object  # type: ignore
    Footer = object  # type: ignore
    Header = object  # type: ignore
    Input = object  # type: ignore
    RichLog = object  # type: ignore
    Static = object  # type: ignore


if TEXTUAL_AVAILABLE:

    class ShellTUI(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #top {
            height: 1fr;
        }
        #session-pane {
            width: 40%;
            border: round $accent;
            padding: 1;
        }
        #conversation-pane {
            width: 60%;
            border: round $accent;
        }
        #trace-pane {
            height: 1fr;
            border: round $accent;
        }
        #input-box {
            dock: bottom;
        }
        """

        def __init__(self, engine):
            super().__init__()
            self.engine = engine

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical(id="top"):
                with Horizontal():
                    yield Static(id="session-pane")
                    yield RichLog(id="conversation-pane", wrap=True)
                yield RichLog(id="trace-pane", wrap=True)
            yield Input(placeholder="Type command or request", id="input-box")
            yield Footer()

        def on_mount(self) -> None:
            self._refresh_session()
            for line in self.engine.banner_lines():
                self._conversation(line)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            line = str(event.value or "").strip()
            event.input.value = ""
            if not line:
                return

            self._conversation(f"> {line}")
            keep = self.engine.handle_line(line, emit=self._emit)
            self._refresh_session()
            if not keep:
                self.exit()

        def _emit(self, text: str) -> None:
            s = str(text or "")
            if s.startswith(("▶️", "✅", "❌", "🔁")):
                self._trace(s)
            else:
                self._conversation(s)

        def _conversation(self, msg: str) -> None:
            pane = self.query_one("#conversation-pane", RichLog)
            pane.write(msg)

        def _trace(self, msg: str) -> None:
            pane = self.query_one("#trace-pane", RichLog)
            pane.write(msg)

        def _refresh_session(self) -> None:
            pane = self.query_one("#session-pane", Static)
            pane.update("\n".join(self.engine.session.summary_lines()))


def run_tui(engine) -> bool:
    if not TEXTUAL_AVAILABLE:
        return False
    app = ShellTUI(engine)
    app.run()
    return True
