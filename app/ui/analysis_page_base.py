from pathlib import Path

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QWidget,
    QMessageBox,
    QComboBox,
    QLabel,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class AnalysisPageBase(QWidget):
    """Общая база для страниц анализа колебаний."""

    LEFT_PANEL_WIDTH = 452
    CANVAS_MIN_WIDTH = 680
    CANVAS_MIN_HEIGHT = 460
    RESULT_MIN_HEIGHT = 170

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.figure = Figure(facecolor="#ffffff", constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("plotCanvas")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(self.CANVAS_MIN_WIDTH, self.CANVAS_MIN_HEIGHT)
        self.preset_combo = None
        self.presets = {}
        self.results_label = None
        self.results_scroll = None

    def _show_error(self, text: str):
        QMessageBox.critical(self, "Ошибка параметров", text)

    def _get_current_parameters(self) -> dict:
        return self.get_parameters()

    def _make_card(self, title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setMinimumWidth(0)
        layout = QVBoxLayout(frame)
        # справа оставляем чуть больше места, чтобы контент не прятался под полосой прокрутки
        layout.setContentsMargins(18, 18, 24, 18)
        layout.setSpacing(10)
        if title:
            label = QLabel(title)
            label.setObjectName("sectionTitle")
            layout.addWidget(label)
        return frame, layout

    def _make_result_card(self, initial_text: str = ""):
        card, layout = self._make_card("Результаты расчёта")

        label = QLabel(initial_text)
        label.setObjectName("resultsLabel")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(label)
        scroll.setMinimumHeight(self.RESULT_MIN_HEIGHT)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(scroll)

        self.results_label = label
        self.results_scroll = scroll
        return card, label

    def _set_results_text(self, text: str):
        if self.results_label is None:
            return
        self.results_label.setText(text)
        self.results_label.adjustSize()
        if self.results_scroll is not None:
            self.results_scroll.ensureVisible(0, 0)

    def _style_plot_axes(self, ax, title: str, xlabel: str, ylabel: str, *, equal: bool = False):
        ax.set_facecolor("#fbfcfe")
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if equal:
            ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.set_aspect("auto")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.margins(x=0.06, y=0.10)
        for spine in ax.spines.values():
            spine.set_color("#c9d2e3")
            spine.set_linewidth(0.9)
        if ax.legend_ is not None:
            leg = ax.legend_
            leg.get_frame().set_facecolor("#ffffff")
            leg.get_frame().set_edgecolor("#d7dfeb")
            leg.get_frame().set_alpha(0.95)

    def _finalize_plot(self):
        self.figure.canvas.draw_idle()

    def _setup_preset_selector(self, parent_layout, presets: dict, slot, label_text: str = "Типовые режимы:"):
        self.presets = dict(presets)
        self.preset_combo = QComboBox()
        self.preset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.preset_combo.addItem("Выберите пресет")
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentTextChanged.connect(slot)

        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        parent_layout.addWidget(label)
        parent_layout.addWidget(self.preset_combo)

    def _add_frequency_controls(self, parent_layout, start_input, end_input, step_input):
        for text, widget in [
            ("Начальная частота ω₀ (рад/с)", start_input),
            ("Конечная частота ω (рад/с)", end_input),
            ("Шаг Δω", step_input),
        ]:
            label = QLabel(text)
            label.setObjectName("fieldLabel")
            parent_layout.addWidget(label)
            parent_layout.addWidget(widget)

    def _make_left_controls_scroll(self, left_card: QFrame) -> QScrollArea:
        left_card.setMinimumWidth(0)
        scroll = QScrollArea()
        scroll.setObjectName("leftPanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left_card)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(self.LEFT_PANEL_WIDTH)
        scroll.setMaximumWidth(self.LEFT_PANEL_WIDTH + 56)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return scroll

    def _build_analysis_layout(self, left_card: QFrame, result_card: QFrame):
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        left_scroll = self._make_left_controls_scroll(left_card)

        plot_card, plot_layout = self._make_card("График")
        plot_layout.addWidget(self.canvas, 1)
        plot_card.setMinimumSize(self.CANVAS_MIN_WIDTH + 36, self.CANVAS_MIN_HEIGHT + 72)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(plot_card)
        right_splitter.addWidget(result_card)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([560, 190])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([self.LEFT_PANEL_WIDTH + 20, 920])

        root.addWidget(main_splitter)

        self._main_splitter = main_splitter
        self._right_splitter = right_splitter
        self._left_scroll = left_scroll
        self._plot_card = plot_card
        self._result_card = result_card


    def _project_root_dir(self) -> Path:
        candidates = []
        try:
            here = Path(__file__).resolve()
            candidates.extend([here.parent, *here.parents])
        except Exception:
            pass
        candidates.append(Path.cwd())

        for candidate in candidates:
            if (candidate / "main.py").exists():
                return candidate
            if (candidate / "app" / "main_window.py").exists() or (candidate / "main_window.py").exists():
                return candidate
        return Path.cwd()

    def _ensure_export_dir(self) -> Path:
        export_dir = self._project_root_dir() / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir

    def _default_export_path(self, filename: str) -> str:
        return str(self._ensure_export_dir() / filename)

    def sizeHint(self) -> QSize:
        return QSize(1310, 860)

    def minimumSizeHint(self) -> QSize:
        return QSize(1160, 760)
