
from utils.utils.general import try_fill_default_dict
from .log import Font, color_text


'''
进度条工具函数
'''

__all__ = ['ProcessBar', 'PbarContext', 'try_get_pbar']


# 用来包装 tqdm 的进度条，使得没有导入 tqdm 时，不会报错（也不会显示），进度条还可以手动关闭（不会引起大批量代码的修改）
# noinspection SpellCheckingInspection
class ProcessBar:
    def __init__(self, **kwargs):
        if not kwargs.pop('init', True):
            self.pbar = None
        else:
            self.pbar = try_get_pbar(**kwargs)

    def update(self, *args, **kwargs):
        if self.pbar is None:
            return

        progress = (self.pbar.n + 1) / self.pbar.total
        # set bar_format is more flexible than:
        # `self.pbar.colour = get_custom_progress_color(progress)` to directly set color
        self.pbar.bar_format = get_custom_progress_bar_format(progress)
        self.pbar.update(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'pbar':
            return self.pbar

        pbar = self.pbar
        if pbar is not None:
            if hasattr(pbar, name):
                attr = getattr(pbar, name)
                if callable(attr):
                    def method(*args, **kwargs):
                        return getattr(pbar, name)(*args, **kwargs)
                    return method
                else:
                    return attr
            raise AttributeError(f"'ProcessBar' object has no attribute '{name}'")
        else:
            return self.do_nothing

    def __setattr__(self, name, value):
        if name == 'pbar':
            object.__setattr__(self, name, value)
            return

        pbar = self.pbar
        if pbar is not None:
            if hasattr(pbar, name):
                setattr(pbar, name, value)
            else:
                AttributeError(f"'ProcessBar' object has no attribute '{name}'")
        else:
            object.__setattr__(self, name, value)

    def do_nothing(self, *args, **kwargs):
        pass

    @staticmethod
    def get_empty_pbar():
        return ProcessBar(init=False)



class PbarContext:
    """

    e.g.
    with pbar_context(show_if=image is not None, total=100, desc='Process') as pbar:
        for i in range(100):
            pbar.update(1)
    """
    def __init__(self, show_if=True, **kwargs):
        self.show_if = show_if
        self.kwargs = kwargs

    def __enter__(self):
        if self.show_if:
            self.pbar = ProcessBar(**self.kwargs)
            return self.pbar
        else:
            return ProcessBar.get_empty_pbar()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show_if:
            self.pbar.close()


def try_get_pbar(header=None, description=None, start=0, **tqdm_kwargs):
    if 'ascii' in tqdm_kwargs and tqdm_kwargs['ascii'] in ascii_chars:
        tqdm_kwargs['ascii'] = ascii_chars[tqdm_kwargs['ascii']]
    tqdm_kwargs = try_fill_default_dict(
        tqdm_kwargs,
        bar_format=f'{color_text("{l_bar}", Font.LIGHT_BLACK)}{{bar}}{color_text("{r_bar}", Font.LIGHT_BLACK)}',
        ascii=ascii_chars['horizontal'],
        ncols=80,
    )
    if 'ascii' in tqdm_kwargs and tqdm_kwargs['ascii'] in ascii_chars:
        tqdm_kwargs['ascii'] = ascii_chars[tqdm_kwargs['ascii']]
    try:
        import tqdm
        p_bar = tqdm.tqdm(**tqdm_kwargs)
        if header is not None:
            p_bar.set_description_str(header)
        if description is not None:
            p_bar.set_postfix_str(description)
        p_bar.update(start)
        return p_bar
    except ImportError as _:
        print('tqdm not installed, progress bar will not be shown.')
        return None
        # import pip
        # pip.main(['install', '--user', 'tqdm'])



#region ==[Color]==
ascii_chars = dict(
    arrow = r"->" ,
    split = r" ->=" ,
    fade = r" ░▒▓█",
    vertical = r" ▁▂▃▄▅▆▇█",
    horizontal = r" ▏▎▍▌▋▊▉█",
    circle = r" •○◔◑◕●",
    linux_apt = r".#",
    numbers1 = r" 0123456789-",
    numbers2 = r" 0123456789=",
    numbers3 = r" 0123456789#",
    letters = r" ABCDEFGHIJKLMNOPQRSTUVWXYZ#",
    square = "□■",
    square_block = "◻◼",
    energy = "▯▮",
    bullet = " ▯▮",
    experi = "▭▬",
    star = "☆★",
    nosy = "☷☳☵☱☶☲☴☰",
    line1 = " ⋅‐–—",
    line2 = " ⋅‑‒―",
    point = " ⋅╌┄┈─",
    point_heavy = " ∙╍┅┉━",
)
default_colors = ["#440000", "#880000", "#888800", "#888800", "#888800", "#888800", "#008800", "#008888", "#ffffff"]


def lerp_color(color1, color2, factor):
    """
    Args:
        color1 (str): hex color string, e.g. "#ff0000"
        color2 (str): hex color string, e.g. "#00ff00"
        factor (float): interpolation factor, e.g. 0.5
    """
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_gradient_color(progress, colors=None):
    if colors is None:
        colors = default_colors

    if progress <= 0:
        return colors[0]
    if progress >= 1:
        return colors[-1]

    segment = progress * (len(colors) - 1)
    idx = int(segment)
    factor = segment - idx
    return lerp_color(colors[idx], colors[idx + 1], factor)


def get_custom_progress_color(progress):
    if progress <= 0.2:
        return "#ff0000"
    elif progress <= 0.6:
        return "#ffff00"
    elif progress <= 0.8:
        return "#00ff00"
    elif progress <= 0.90:
        return "#0000ff"
    elif progress <= 0.95:
        return "#00ffff"
    else:
        return "#ffffff"


def get_custom_progress_bar_format(progress):
    bar_str = "{bar}"
    if progress <= 0.05:
        bar_str = color_text(bar_str, Font.DIM_RED)
    if progress <= 0.1:
        bar_str = color_text(bar_str, Font.RED)
    if progress <= 0.2:
        bar_str = color_text(bar_str, Font.LIGHT_RED)
    elif progress <= 0.6:
        bar_str = color_text(bar_str, Font.YELLOW)
    elif progress <= 0.8:
        bar_str = color_text(bar_str, Font.GREEN)
    elif progress <= 0.85:
        bar_str = color_text(bar_str, Font.BLUE)
    elif progress <= 0.90:
        bar_str = color_text(bar_str, Font.LIGHT_BLUE)
    elif progress <= 0.95:
        bar_str = color_text(bar_str, Font.CYAN)
    elif progress <= 0.98:
        bar_str = color_text(bar_str, Font.LIGHT_CYAN)
    else:
        bar_str = color_text(bar_str, Font.WHITE)

    bar_format = f'{color_text("{l_bar}", Font.LIGHT_BLACK)}{bar_str}{color_text("{r_bar}", Font.LIGHT_BLACK)}'
    return bar_format


def get_custom_progress_bar_format_256(progress):
    colors = [
        52, 88, 124, 160, 196, 1, 202, 208, 214, 220, # red
        184, # yellow
        148, 112, 76, 40, 41, 36, # green
        31, 32, 39, 45, # blue
        51, 87, 123, 159, 195, # cyan
        231, 255  # white
    ]
    bar_str = "{bar}"
    for i in range(len(colors)):
        if progress <= i / (len(colors) - 1):
            bar_str = color_text(bar_str, Font.get_256_color(colors[i]))
            break
    bar_format = f'{color_text("{l_bar}", Font.LIGHT_BLACK)}{bar_str}{color_text("{r_bar}", Font.LIGHT_BLACK)}'
    return bar_format
#endregion