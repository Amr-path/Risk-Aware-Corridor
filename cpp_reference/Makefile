CXX ?= g++
# -ffp-contract=off keeps IEEE rounding bit-identical to Python (no FMA),
# which is required for exact node-count parity. Drop -march=native for the
# same reason. Use -O2; still ~50-100x faster than the Python reference.
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -Wall
pathfind: pathfind.cpp
	$(CXX) $(CXXFLAGS) -o pathfind pathfind.cpp
clean:
	rm -f pathfind
.PHONY: clean
