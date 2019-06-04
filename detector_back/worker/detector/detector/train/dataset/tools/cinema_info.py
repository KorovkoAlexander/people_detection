import json
from typing import NamedTuple, List, Optional, Dict

import numpy as np


class Head(object):
    def __init__(self, x: int, y: int, r: int = 0):
        self.x = x
        self.y = y
        self.r = r

    @property
    def top(self) -> int:
        return self.y - self.r

    @property
    def left(self) -> int:
        return self.x - self.r

    @property
    def bottom(self) -> int:
        return self.y + self.r

    @property
    def right(self) -> int:
        return self.x + self.r


class Body(object):
    def __init__(self, top: int, left: int, bottom: int, right: int):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def prune(self, dim_x, dim_y):
        self.top = max(0, self.top)
        self.left = max(0, self.left)
        self.bottom = min(dim_y, self.bottom)
        self.right = min(dim_x, self.right)

        return self


class HallCoefficients(NamedTuple):
    a: float
    b: float


class Coordinates(NamedTuple):
    x: int
    y: int


class Seat(NamedTuple):
    pos: Coordinates
    num_row: int
    num_seat: int


class HallSeatsInfo(NamedTuple):
    num_seats: int
    num_rows: int
    num_seats_by_row: Dict[str, int]
    seat_width_by_row: Dict[str, float]
    seats: List[Seat]


class Hall(NamedTuple):
    cinema: Optional[str]
    hall: Optional[str]
    coefficients: Optional[HallCoefficients]
    seats_info: Optional[HallSeatsInfo]

    def seat_width(self, y: int) -> float:
        return y * self.coefficients.a + self.coefficients.b

    def head_radius(self, y: int) -> float:
        return self.seat_width(y) * 0.22

    def head_size(self, y: int) -> float:
        return self.head_radius(y) * 2.0

    def head_square(self, y: int) -> float:
        return np.pi * self.head_size(y) ** 2 / 4.0

    def head_distance_to_seats(
            self, head: Head, coeff: float = 3.0
    ) -> List[float]:
        acc = []
        for seat in self.seats_info.seats:
            coeff = coeff if head.y < seat.pos.y else 1.0
            acc.append(
                coeff * np.linalg.norm(
                    (head.x - seat.pos.x, head.y - seat.pos.y)
                )
            )
        return acc

    def has_coefficients(self):
        return self.coefficients is not None

    def has_seats_info(self):
        return self.seats_info is not None

    def has_any_info(self):
        has_name = (self.cinema is not None) and (self.hall is not None)

        return has_name or self.has_coefficients() or self.has_seats_info()


class CinemaInfo(object):
    def __init__(self, coefficients_path, seats_info_path):
        self.coefficients = self.load_coefficients(coefficients_path)
        self.seats_info = self.load_seats_info(seats_info_path)

    def get_hall(self, cinema: Optional[str], hall: Optional[str]) -> Hall:
        if cinema in self.coefficients and hall in self.coefficients[cinema]:
            coefficients = self.coefficients[cinema][hall]
        else:
            coefficients = None

        if cinema in self.seats_info and hall in self.seats_info[cinema]:
            seats_info = self.seats_info[cinema][hall]
        else:
            seats_info = None

        return Hall(cinema, hall, coefficients, seats_info)

    def load_coefficients(
            self, path
    ) -> Dict[str, Dict[str, HallCoefficients]]:
        with open(path, 'r') as f:
            hall_coeffs = json.load(f)

        for cinema in hall_coeffs:
            for hall in hall_coeffs[cinema]:
                hall_coeffs[cinema][hall] = (
                    HallCoefficients(**hall_coeffs[cinema][hall])
                )

        return hall_coeffs

    def load_seats_info(self, path) -> Dict[str, Dict[str, HallSeatsInfo]]:
        with open(path, 'r') as f:
            hall_seats_json = json.load(f)

        hall_seats = {}
        for cinema in hall_seats_json:
            hall_seats[cinema] = dict()
            for hall in hall_seats_json[cinema]:
                seats = hall_seats_json[cinema][hall]["seats"]
                seats = [
                    Seat(
                        Coordinates(*seat["loc"]),
                        seat["num_row"],
                        seat["num_seat"],
                    ) for seat in seats
                ]
                hall_seats_info = hall_seats_json[cinema][hall]
                hall_seats[cinema][hall] = HallSeatsInfo(
                    num_seats=hall_seats_info["num_seats"],
                    num_rows=hall_seats_info["num_rows"],
                    num_seats_by_row=hall_seats_info["num_seats_by_row"],
                    seat_width_by_row=hall_seats_info["seat_width_by_row"],
                    seats=seats
                )

        return hall_seats
