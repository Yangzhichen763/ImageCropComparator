import os
import sys
import time
import shutil
from typing import Optional


# ANSI color codes (Linux/macOS terminals)
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLINK = "\033[5m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def supports_color(stream) -> bool:
    try:
        return stream.isatty() and os.name != 'nt'
    except Exception:
        return False


def colorize(text: str, color: Optional[str] = None, bold: bool = False) -> str:
    if not color:
        return text
    if bold:
        return f"{Color.BOLD}{color}{text}{Color.RESET}"
    return f"{color}{text}{Color.RESET}"


class Logger:
    LEVELS = {
        'debug': 10,
        'info': 20,
        'warn': 30,
        'error': 40,
    }

    def __init__(self, level: str = 'info', use_color: Optional[bool] = None, name: str = 'LLIE-Results'):
        self.level = self.LEVELS.get(level, 20)
        # auto-enable color if not specified and stdout is a tty
        env_force = os.environ.get('FORCE_COLOR', '')
        self.use_color = (supports_color(sys.stdout) or env_force) if use_color is None else bool(use_color)
        self.name = name

    def set_level(self, level: str):
        self.level = self.LEVELS.get(level, self.level)

    def set_color_enabled(self, enabled: bool):
        self.use_color = bool(enabled)

    def _ts(self):
        return time.strftime('%H:%M:%S')

    def _separator_with_ts(self, c: str = '-') -> str:
        ts_full = time.strftime('%Y-%m-%d %H:%M:%S')
        width = shutil.get_terminal_size(fallback=(80, 20)).columns
        width = max(width, len(ts_full) + 4)
        line = [c] * width
        start = max(0, (width - len(ts_full)) // 2)
        line[start:start + len(ts_full)] = list(ts_full)
        sep = ''.join(line)
        return sep

    def _emit(self, lvl_name: str, msg: str, color: Optional[str] = None):
        prefix = f"[{self._ts()}] {self.name} {lvl_name.upper():>5}: "
        if self.use_color and color:
            prefix = colorize(prefix, color)
        print(prefix + msg)

    def debug(self, msg: str):
        if self.level <= self.LEVELS['debug']:
            self._emit('debug', msg, Color.BRIGHT_BLACK if self.use_color else None)

    def info(self, msg: str):
        if self.level <= self.LEVELS['info']:
            self._emit('info', msg, Color.CYAN if self.use_color else None)

    def success(self, msg: str):
        # success shown at info threshold
        if self.level <= self.LEVELS['info']:
            self._emit('info', msg, Color.GREEN if self.use_color else None)

    def warn(self, msg: str):
        if self.level <= self.LEVELS['warn']:
            self._emit('warn', msg, Color.YELLOW if self.use_color else None)

    def error(self, msg: str):
        self._emit('error', msg, Color.RED if self.use_color else None)

    def note(self, msg: str):
        # neutral note, cyan
        self._emit('info', msg, Color.BRIGHT_CYAN if self.use_color else None)

    def banner(self, title: str):
        sep1 = self._separator_with_ts('-')
        sep2 = self._separator_with_ts('=')
        width = len(sep1)
        title_raw = f"  {title}"
        padding = max(0, width - len(title_raw))
        left = padding // 2
        right = padding - left
        title_line = ' ' * left + title_raw + ' ' * right

        print()
        if self.use_color:
            print(colorize(sep2, Color.BRIGHT_BLUE))
            print()
            print(colorize(title_line, Color.BRIGHT_WHITE, bold=True))
            print()
            print(colorize(sep1, Color.BRIGHT_BLUE))
        else:
            print(sep2)
            print()
            print(title_line)
            print()
            print(sep1)
        print()

    # Styling helpers for tokens/values
    def style_key(self, token: str) -> str:
        return colorize(token, Color.BRIGHT_MAGENTA, bold=True) if self.use_color else token

    def style_num(self, token: str) -> str:
        return colorize(token, Color.BRIGHT_YELLOW, bold=True) if self.use_color else token

    def style_result(self, token: str) -> str:
        return colorize(token, Color.GREEN, bold=True) if self.use_color else token

    def style_mode(self, token: str) -> str:
        return colorize(token, Color.BRIGHT_WHITE, bold=True) if self.use_color else token

    def style_path(self, token: str) -> str:
        return colorize(token, Color.BLUE) if self.use_color else token

    def style_keyward(self, token: str) -> str:
        return colorize(token, Color.MAGENTA) if self.use_color else token

    def style_cmd(self, token: str) -> str:
        return colorize(token, Color.BRIGHT_CYAN, bold=True) if self.use_color else token
