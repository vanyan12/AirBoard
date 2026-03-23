from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Any


class Mode(Enum):
    IDLE = auto()
    DRAWING = auto()
    ERASING = auto()

Point = Tuple[int, int]
Segment = Dict[str, Any]  # {"points": List[Point], "color":
Color = Tuple[int, int, int]  # (B, G, R)

@dataclass
class DrawingState:
    drawing_segments: List[Dict[str, Any]] = field(default_factory=list)
    current_segment: List[Point] = field(default_factory=list)

    mode: Mode = Mode.IDLE
    drawing_stable_counter: int = 0
    not_drawing_stable_counter: int = 0

    pen_color: Color = (0, 0, 255)  # Red
    pen_size: int = 5
    eraser_radius: int = 20

    def start_stroke(self) -> None:
        if not self.current_segment:
            self.current_segment = []

    def add_point(self, point: Point) -> None:
        if self.mode == Mode.DRAWING:
            self.current_segment.append(point)

    def end_stroke(self) -> None:
        if len(self.current_segment) > 1:
            # Store a copy so future point appends/clears cannot mutate saved strokes.
            self.drawing_segments.append({
                "points": self.current_segment.copy(),
                "color": self.pen_color,
                "size": self.pen_size
            })
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


    def reset_stability_counters(self):
        self.drawing_stable_counter = 0
        self.not_drawing_stable_counter = 0