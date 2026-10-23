import re
from pathlib import Path

import pytest
from PIL import Image

from textile_defect_detection import theme
from textile_defect_detection.synthetic import SYNTHETIC_CLASSES, generate_synthetic_dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("fg,bg", [
    (theme.BODY_TEXT, theme.PAGE_BACKGROUND), (theme.BODY_TEXT, theme.SURFACE),
    (theme.NAVY, theme.PAGE_BACKGROUND), ("#FFFFFF", theme.TEAL), (theme.TEAL, theme.SURFACE),
    (theme.SUCCESS, theme.SURFACE), (theme.WARNING, theme.SURFACE), (theme.ERROR, theme.SURFACE),
    (theme.INFO, theme.SURFACE), ("#FFFFFF", theme.NAVY), (theme.MUTED_TEXT, theme.SURFACE),
])
def test_palette_pairs_meet_wcag_aa_for_normal_text(fg, bg):
    assert theme.contrast_ratio(fg, bg) >= 4.5


def test_muted_text_must_not_be_used_on_the_page_background():
    # Measured: #667788 on #F4F7FA is ~4.28:1 (< 4.5). It passes only on white surfaces.
    assert theme.contrast_ratio(theme.MUTED_TEXT, theme.PAGE_BACKGROUND) < 4.5
    assert "MUTED_TEXT" not in theme.app_css() and theme.MUTED_TEXT.lower() not in theme.app_css().lower()


def test_contrast_calculator_known_values():
    assert abs(theme.contrast_ratio("#000000", "#FFFFFF") - 21.0) < 1e-9
    assert abs(theme.contrast_ratio("#777777", "#777777") - 1.0) < 1e-9


def test_streamlit_theme_file_matches_design_tokens():
    text = (ROOT / ".streamlit" / "config.toml").read_text()
    def value(key):
        return re.search(rf'^{key}\s*=\s*"([^"]+)"', text, re.M).group(1).lower()
    assert value("primaryColor") == theme.TEAL.lower()
    assert value("backgroundColor") == theme.PAGE_BACKGROUND.lower()
    assert value("secondaryBackgroundColor") == theme.SURFACE.lower()
    assert value("textColor") == theme.BODY_TEXT.lower()
    assert value("base") == "light"
    assert "gatherUsageStats = false" in text


def test_synthetic_generator_is_labelled_deterministic_and_decodable(tmp_path):
    a = generate_synthetic_dataset(tmp_path / "a", images_per_class=6, seed=3)
    b = generate_synthetic_dataset(tmp_path / "b", images_per_class=6, seed=3)
    names_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*.png"))
    assert names_a == sorted(p.relative_to(b).as_posix() for p in b.rglob("*.png"))
    assert len(names_a) == 6 * len(SYNTHETIC_CLASSES)
    assert all(c.startswith("synthetic_") for c in SYNTHETIC_CLASSES)
    assert (a / names_a[0]).read_bytes() == (b / names_a[0]).read_bytes()
    Image.open(a / names_a[0]).load()
    assert '"synthetic": true' in (a / "dataset_info.json").read_text()
    assert "SYNTHETIC" in (a / "README_SYNTHETIC.txt").read_text()


def test_synthetic_generator_refuses_nonempty_directory(tmp_path):
    generate_synthetic_dataset(tmp_path / "x", images_per_class=3)
    with pytest.raises(FileExistsError):
        generate_synthetic_dataset(tmp_path / "x", images_per_class=3)
