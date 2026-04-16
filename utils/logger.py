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


class Font:
    # 前景色（字体颜色）
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    DEFAULT = "\033[39m"  # 默认颜色

    # 背景色
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    BG_DEFAULT = "\033[49m"  # 默认背景色

    # 样式
    BOLD = "\033[1m"  # 加粗
    DIM = "\033[2m"  # 暗淡
    ITALIC = "\033[3m"  # 斜体（部分终端不支持）
    UNDERLINE = "\033[4m"  # 下划线
    BLINK = "\033[5m"  # 闪烁
    REVERSE = "\033[7m"  # 反色（前景色和背景色互换）
    HIDDEN = "\033[8m"  # 隐藏（文字不可见）

    # 重置所有样式和颜色
    RESET = "\033[0m"

    # 亮色（高亮前景色）
    LIGHT_BLACK = "\033[90m"
    LIGHT_RED = "\033[91m"
    LIGHT_GREEN = "\033[92m"
    LIGHT_YELLOW = "\033[93m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_MAGENTA = "\033[95m"
    LIGHT_CYAN = "\033[96m"
    LIGHT_WHITE = "\033[97m"

    # 亮色背景
    BG_LIGHT_BLACK = "\033[100m"
    BG_LIGHT_RED = "\033[101m"
    BG_LIGHT_GREEN = "\033[102m"
    BG_LIGHT_YELLOW = "\033[103m"
    BG_LIGHT_BLUE = "\033[104m"
    BG_LIGHT_MAGENTA = "\033[105m"
    BG_LIGHT_CYAN = "\033[106m"
    BG_LIGHT_WHITE = "\033[107m"

    # 暗色
    DIM_BLACK = "\033[2m\033[30m"
    DIM_RED = "\033[2m\033[31m"
    DIM_GREEN = "\033[2m\033[32m"
    DIM_YELLOW = "\033[2m\033[33m"
    DIM_BLUE = "\033[2m\033[34m"
    DIM_MAGENTA = "\033[2m\033[35m"
    DIM_CYAN = "\033[2m\033[36m"
    DIM_WHITE = "\033[2m\033[37m"

    # 暗色背景
    BG_DIM_BLACK = "\033[2m\033[40m"
    BG_DIM_RED = "\033[2m\033[41m"
    BG_DIM_GREEN = "\033[2m\033[42m"
    BG_DIM_YELLOW = "\033[2m\033[43m"
    BG_DIM_BLUE = "\033[2m\033[44m"
    BG_DIM_MAGENTA = "\033[2m\033[45m"
    BG_DIM_CYAN = "\033[2m\033[46m"
    BG_DIM_WHITE = "\033[2m\033[47m"

    @staticmethod
    def _debug_color():
        print(f"{Font.BLACK}BLACK{Font.RESET} "
              f"{Font.DEFAULT}DEFAULT{Font.RESET} "
              f"{Font.MAGENTA}MAGENTA{Font.RESET} "
              f"{Font.RED}RED{Font.RESET} "
              f"{Font.YELLOW}YELLOW{Font.RESET} "
              f"{Font.GREEN}GREEN{Font.RESET} "
              f"{Font.BLUE}BLUE{Font.RESET} "
              f"{Font.CYAN}CYAN{Font.RESET} "
              f"{Font.WHITE}WHITE{Font.RESET} "
              f"{Font.LIGHT_BLACK}LIGHT_BLACK{Font.RESET} "
              f"{Font.LIGHT_RED}LIGHT_RED{Font.RESET} "
              f"{Font.LIGHT_GREEN}LIGHT_GREEN{Font.RESET} "
              f"{Font.LIGHT_YELLOW}LIGHT_YELLOW{Font.RESET} "
              f"{Font.LIGHT_BLUE}LIGHT_BLUE{Font.RESET} "
              f"{Font.LIGHT_MAGENTA}LIGHT_MAGENTA{Font.RESET} "
              f"{Font.LIGHT_CYAN}LIGHT_CYAN{Font.RESET} "
              f"{Font.LIGHT_WHITE}LIGHT_WHITE{Font.RESET} "
              f"{Font.DIM_BLACK}DIM_BLACK{Font.RESET} "
              f"{Font.DIM_RED}DIM_RED{Font.RESET} "
              f"{Font.DIM_GREEN}DIM_GREEN{Font.RESET} "
              f"{Font.DIM_YELLOW}DIM_YELLOW{Font.RESET} "
              f"{Font.DIM_BLUE}DIM_BLUE{Font.RESET} "
              f"{Font.DIM_MAGENTA}DIM_MAGENTA{Font.RESET} "
              f"{Font.DIM_CYAN}DIM_CYAN{Font.RESET} "
              f"{Font.DIM_WHITE}DIM_WHITE{Font.RESET} "

              f"{Font.BOLD}BOLD{Font.RESET} "
              f"{Font.DIM}DIM{Font.RESET} "
              f"{Font.ITALIC}ITALIC{Font.RESET} "
              f"{Font.UNDERLINE}UNDERLINE{Font.RESET} "
              f"{Font.BLINK}BLINK{Font.RESET} "
              f"{Font.REVERSE}REVERSE{Font.RESET} "
              f"{Font.HIDDEN}HIDDEN{Font.RESET} "

              f"{Font.BG_BLACK}BG_BLACK{Font.RESET} "
              f"{Font.BG_RED}BG_RED{Font.RESET} "
              f"{Font.BG_GREEN}BG_GREEN{Font.RESET} "
              f"{Font.BG_YELLOW}BG_YELLOW{Font.RESET} "
              f"{Font.BG_BLUE}BG_BLUE{Font.RESET} "
              f"{Font.BG_MAGENTA}BG_MAGENTA{Font.RESET} "
              f"{Font.BG_CYAN}BG_CYAN{Font.RESET} "
              f"{Font.BG_WHITE}BG_WHITE{Font.RESET} "
              f"{Font.BG_DEFAULT}BG_DEFAULT{Font.RESET} "
              f"{Font.BG_LIGHT_BLACK}BG_LIGHT_BLACK{Font.RESET} "
              f"{Font.BG_LIGHT_RED}BG_LIGHT_RED{Font.RESET} "
              f"{Font.BG_LIGHT_GREEN}BG_LIGHT_GREEN{Font.RESET} "
              f"{Font.BG_LIGHT_YELLOW}BG_LIGHT_YELLOW{Font.RESET} "
              f"{Font.BG_LIGHT_BLUE}BG_LIGHT_BLUE{Font.RESET} "
              f"{Font.BG_LIGHT_MAGENTA}BG_LIGHT_MAGENTA{Font.RESET} "
              f"{Font.BG_LIGHT_CYAN}BG_LIGHT_CYAN{Font.RESET} "
              f"{Font.BG_LIGHT_WHITE}BG_LIGHT_WHITE{Font.RESET} "
              f"{Font.RESET}RESET{Font.RESET}")

    @staticmethod
    def get_256_color(code):
        return f"\033[38;5;{code}m"


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


def make_font(text: str, style_code: Optional[str] = None) -> str:
    if not style_code:
        return text
    if style_code:
        return f"{style_code}{text}{Font.RESET}"
    return text


class Logger:
    LEVELS = {
        'debug': 10,
        'info': 20,
        'warn': 30,
        'error': 40,
    }

    def __init__(self, level: str = 'info', use_color: Optional[bool] = None, name: str = ''):
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

    def banner(self, title: str, level: int=0):
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
            if level == 0:
                print(colorize(sep2, Color.BRIGHT_BLUE))
                # print()
            elif level == 1:
                print(colorize(sep1, Color.BRIGHT_BLUE))
            print(colorize(title_line, Color.BRIGHT_WHITE, bold=True))
            # if level == 0:
            #     print()
            #     print(colorize(sep1, Color.BRIGHT_BLUE))
        else:
            if level == 0:
                print(sep2)
                # print()
            elif level == 1:
                print(sep1)
            print(title_line)
            # if level == 0:
            #     print()
            #     print(sep1)
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

    def style_important(self, token: str) -> str:
        return colorize(token, Color.GREEN, bold=True) if self.use_color else token

    def style_underline(self, token: str) -> str:
        return make_font(token, Font.UNDERLINE) if self.use_color else token
