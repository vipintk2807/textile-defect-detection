"""Visual design tokens from the app theme specification, plus a WCAG contrast calculator.

Only presentation constants live here; nothing in this module touches model inputs/outputs.
"""

from __future__ import annotations

NAVY = "#17324D"
TEAL = "#167D8D"
PAGE_BACKGROUND = "#F4F7FA"
SURFACE = "#FFFFFF"
BODY_TEXT = "#263746"
MUTED_TEXT = "#667788"
BORDER = "#D7DEE7"

SUCCESS = "#237A57"
WARNING = "#946200"
ERROR = "#B42318"
INFO = "#2457A7"

FONT_STACK = "Inter, 'Segoe UI', Arial, sans-serif"


def _channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio between two ``#RRGGBB`` colours."""
    a, b = relative_luminance(foreground), relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def app_css() -> str:
    """Small stylesheet for the app's *own* HTML blocks (no reliance on Streamlit internals)."""
    return f"""
<style>
.tdd-title {{ color:{NAVY}; font-family:{FONT_STACK}; font-size:30px; font-weight:700; margin:0; }}
.tdd-subtitle {{ color:{BODY_TEXT}; font-family:{FONT_STACK}; font-size:16px; margin:4px 0 16px 0; }}
.tdd-panel {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
  padding:16px; margin-bottom:16px; font-family:{FONT_STACK}; color:{BODY_TEXT}; }}
.tdd-label {{ color:{BODY_TEXT}; font-size:13px; font-weight:500; margin:0 0 4px 0; }}
.tdd-prediction {{ color:{NAVY}; font-size:26px; font-weight:700; margin:0 0 8px 0;
  overflow-wrap:anywhere; word-break:break-word; }}
.tdd-status {{ font-weight:600; margin:0 0 8px 0; }}
.tdd-status.ok {{ color:{SUCCESS}; }}
.tdd-status.warn {{ color:{WARNING}; }}
.tdd-status.err {{ color:{ERROR}; }}
.tdd-status.info {{ color:{INFO}; }}
.tdd-score-row {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
.tdd-score-name {{ flex:0 0 40%; overflow-wrap:anywhere; word-break:break-word; font-size:14px; }}
.tdd-bar {{ flex:1 1 auto; background:{PAGE_BACKGROUND}; border:1px solid {BORDER};
  border-radius:4px; height:14px; }}
.tdd-bar > div {{ background:{TEAL}; height:100%; border-radius:3px; }}
.tdd-score-value {{ flex:0 0 56px; text-align:right; font-variant-numeric:tabular-nums;
  font-size:14px; }}
.tdd-notice {{ color:{BODY_TEXT}; border-left:4px solid {WARNING}; padding:8px 12px;
  background:{SURFACE}; font-size:14px; }}
</style>
"""
