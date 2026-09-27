"""Non-blocking skip control for harness sweeps (press 'c' to skip current run)."""
import select
import sys
import threading
import termios
import tty


class SkipListener:
    def __init__(self, key: str = "c"):
        self._key = key.lower()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._old_term = None

    def start(self):
        if not config_experiment_skip_enabled():
            return
        if not sys.stdin.isatty():
            print("  (skip key disabled — stdin is not a TTY)")
            return
        self._fd = sys.stdin.fileno()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        try:
            self._old_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch.lower() == self._key:
                    self._event.set()
                    return
        except (termios.error, OSError):
            pass
        finally:
            self._restore_term()

    def _restore_term(self):
        if self._old_term is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            self._old_term = None

    def stop(self):
        self._stop.set()
        self._restore_term()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def poll_line(self) -> bool:
        if not sys.stdin.isatty():
            return False
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            if line in ("c", "skip", "s"):
                self._event.set()
                return True
        return False

    def should_stop(self, _step: int) -> bool:
        if self._event.is_set():
            return True
        return self.poll_line()


def config_experiment_skip_enabled() -> bool:
    import config
    return getattr(config, "experiment_skip_enabled", True)
