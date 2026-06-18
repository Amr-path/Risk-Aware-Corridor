"""Tiny dependency-free progress bar (prints to stderr).

Usage:
    pb = Progress(total, desc="parity ")
    for ...:
        ...
        pb.tick()
    pb.close()
"""
import sys, time


class Progress:
    def __init__(self, total, desc="", width=30, stream=sys.stderr):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.stream = stream
        self.done = 0
        self.t0 = time.time()
        self._draw()

    def _fmt(self, s):
        s = int(s)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s//60}m{s%60:02d}s"
        return f"{s//3600}h{(s%3600)//60:02d}m"

    def _draw(self):
        frac = self.done / self.total
        el = time.time() - self.t0
        eta = (el / frac - el) if frac > 0 else 0.0
        rate = self.done / el if el > 0 else 0.0
        fill = int(self.width * frac)
        bar = "#" * fill + "-" * (self.width - fill)
        self.stream.write(
            f"\r{self.desc}[{bar}] {self.done}/{self.total} "
            f"{100*frac:4.1f}%  elapsed {self._fmt(el)}  eta {self._fmt(eta)}  "
            f"{rate:.1f}/s   "
        )
        self.stream.flush()

    def tick(self, n=1):
        self.done = min(self.done + n, self.total)
        self._draw()

    def close(self):
        self.done = self.total
        self._draw()
        self.stream.write("\n")
        self.stream.flush()
