from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple


class Mode(Enum):
    IDLE = auto()
    DRAWING = auto()
    ERASING = auto()


Point = Tuple[float, float]
Color = Tuple[int, int, int]  # (B, G, R)

MIN_PEN_SIZE = 5
MAX_PEN_SIZE = 25


@dataclass
class Segment:
    points: List[Point]
    color: Color
    size: int


@dataclass
class DrawingState:
    drawing_segments: List[Segment] = field(default_factory=list)
    current_segment: List[Point] = field(default_factory=list)

    mode: Mode = Mode.IDLE
    drawing_stable_counter: int = 0
    not_drawing_stable_counter: int = 0

    pen_color: Color = (0, 0, 255)
    pen_size: int = 5
    eraser_radius: int = 20

    def __post_init__(self) -> None:
        self.pen_size = self.clamp_pen_size(self.pen_size)

    @staticmethod
    def clamp_pen_size(value: int) -> int:
        return max(MIN_PEN_SIZE, min(MAX_PEN_SIZE, value))

    def set_pen_size(self, value: int) -> None:
        self.pen_size = self.clamp_pen_size(value)

    def start_stroke(self) -> None:
        if not self.current_segment:
            self.current_segment = []

    def add_point(self, point: Point) -> None:
        if self.mode == Mode.DRAWING:
            self.current_segment.append(point)

    def end_stroke(self) -> None:
        if len(self.current_segment) > 1:
            self.drawing_segments.append(
                Segment(
                    points=self.current_segment.copy(),
                    color=self.pen_color,
                    size=self.pen_size,
                )
            )
        self.current_segment.clear()

    def commit_if_drawing(self) -> None:
        if self.mode == Mode.DRAWING or self.current_segment:
            self.end_stroke()

    def on_hand_lost(self) -> None:
        self.commit_if_drawing()
        self.mode = Mode.IDLE
        self.reset_stability_counters()

    def clear_canvas(self) -> None:
        self.drawing_segments.clear()
        self.current_segment.clear()

    def reset_stability_counters(self) -> None:
        self.drawing_stable_counter = 0
        self.not_drawing_stable_counter = 0
