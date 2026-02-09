import math


class Vector:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    # --- Constructors ---

    @classmethod
    def cartesian(cls, x: float, y: float) -> "Vector":
        return cls(x, y)

    @classmethod
    def polar(cls, magnitude: float, direction_deg: float) -> "Vector":
        rad = math.radians(direction_deg)
        return cls(
            magnitude * math.cos(rad),
            magnitude * math.sin(rad)
        )

    # --- Basic operations (return new vectors) ---

    def add(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def sub(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def mul(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar)

    def div(self, scalar: float) -> "Vector":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector(self.x / scalar, self.y / scalar)

    # --- Magnitude / direction ---

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def direction(self) -> float:
        """Direction in degrees, CCW from +X axis."""
        return math.degrees(math.atan2(self.y, self.x))

    # --- Normalisation ---

    def normalised(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            return Vector(0.0, 0.0)
        return self.div(mag)

    def normalise(self) -> None:
        mag = self.magnitude()
        if mag == 0:
            self.x = 0.0
            self.y = 0.0
        else:
            self.x /= mag
            self.y /= mag

    # --- Convenience ---

    def __repr__(self) -> str:
        return f"Vector(x={self.x:.4f}, y={self.y:.4f})"
