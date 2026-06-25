"""
charging_layout_generator.py
─────────────────────────────────────────────────────────────────────────────
Four strategic pipelines for placing Kiva-robot charging stations in an
RMFS warehouse simulation.

Grid cell legend (matches data_matrix convention)
─────────────────────────────────────────────────
  0  Navigable aisle / floor
  1  Storage pod (obstacle)
  2  Charging station  ← written by this module
  3  Picking / delivery station
  4  Replenishment station

Any other positive integer is treated as a non-traversable obstacle during
BFS pathfinding unless ``TRAVERSABLE`` is overridden at the module level.

Pipelines
─────────
  1  Set Cover Approximation  — worst-case reachability guarantee
  2  Affinity Propagation     — data-driven traffic clustering (requires scikit-learn)
  3  Picking-Station Heuristic — opportunity charging in queue cells
  4  Perimeter Wall Strategy  — isolation / control baseline

Usage
─────
    from model.charging_layout_generator import ChargingLayoutGenerator
    import numpy as np

    grid = np.array(...)   # 2-D int array using the legend above

    # Pipeline 1 — every navigable cell is ≤ 8 hops from a charger
    gen = ChargingLayoutGenerator(grid, {"pipeline": 1, "d": 8})
    new_grid = gen.generate()

    # Pipeline 2 — traffic-aware clustering
    traffic = np.random.rand(*grid.shape)
    gen = ChargingLayoutGenerator(
        grid,
        {"pipeline": 2, "traffic_matrix": traffic, "c": 1.0,
         "alpha": 1.0, "beta": 1.5, "gamma": 0.5},
    )
    new_grid = gen.generate()

    # Pipeline 3 — opportunity charging near picking stations
    gen = ChargingLayoutGenerator(grid, {"pipeline": 3, "num_chargers": 12})
    new_grid = gen.generate()

    # Pipeline 4 — perimeter isolation
    gen = ChargingLayoutGenerator(grid, {"pipeline": 4, "num_chargers": 10})
    new_grid = gen.generate()
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── cell-value constants ──────────────────────────────────────────────────────
POD: int = 1
CHARGER: int = 2

# Di sistem Rika, Picking station = 11, Replenishment = 21
PICKING_STATIONS: FrozenSet[int] = frozenset({11, 21})

# Di sistem Rika, jalanan/rel Kiva punya banyak angka (3-7, 12-29, 99)
TRAVERSABLE: FrozenSet[int] = frozenset({
    3, 4, 5, 6, 7,                # Aisle & intersections
    12, 13, 14, 16, 17, 18, 19,   # Rails & corners (kiri/picking)
    22, 23, 24, 26, 27, 28, 29,   # Rails & corners (kanan/replenishment)
    99                            # Blank space / safe zone
})

# Extended traversable for BFS: includes floor (0) and pod (1) cells
# because robots physically traverse through these in the simulation.
BFS_TRAVERSABLE: FrozenSet[int] = TRAVERSABLE | frozenset({0, 1})

# Cells eligible for charger placement (Pipeline 1):
# floor (deactivated pod) and active pod positions only.
# Stations (11, 21) excluded — those are human staff positions.
CHARGER_CANDIDATE_VALUES: FrozenSet[int] = frozenset({0, 1})

# Cells eligible for Pipeline 2 (AP) charger placement — stricter than
# CHARGER_CANDIDATE_VALUES.  Only value 0 (warehouse floor / deactivated
# pod slot) is allowed because:
#   • Pods (1) are excluded — AP tends to bunch chargers along pod
#     columns, and stacked chargers in a pod lane break Dijkstra routing.
#   • Blank-space (99) cells are excluded — netlogo.py adds routing-graph
#     nodes only for values {0, 1, 2}, so a charger on a 99 cell is
#     unreachable by the pathfinder at runtime.
AP_SAFE_CANDIDATE_VALUES: FrozenSet[int] = frozenset({0})

# ── type aliases ──────────────────────────────────────────────────────────────
Cell = Tuple[int, int]
Matrix = np.ndarray


# ═════════════════════════════════════════════════════════════════════════════
#  Module-level helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    """Manhattan (L¹) distance between two grid cells."""
    return abs(r1 - r2) + abs(c1 - c2)


def _evenly_spaced_sample(items: list, n: int) -> list:
    """
    Select *n* items from *items* at approximately equal index spacing.

    Fully deterministic — no randomness involved.  Useful for distributing
    chargers evenly along a list of candidate cells without clustering them
    at one end.

    Examples
    --------
    >>> _evenly_spaced_sample(list(range(10)), 3)
    [0, 3, 6]
    >>> _evenly_spaced_sample(list(range(10)), 10)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def bfs_on_grid(
    matrix: Matrix,
    start: Cell,
    max_depth: int,
    traversable: FrozenSet[int] = TRAVERSABLE,
) -> Set[Cell]:
    """
    General-purpose 4-connected BFS over a 2-D integer grid.

    Explores neighbours whose cell value is in *traversable*, up to
    *max_depth* hops from *start*.  The start cell itself is included in
    the returned set only if its value is also in *traversable*.

    Parameters
    ----------
    matrix : np.ndarray of int
    start : (row, col)
    max_depth : int
        Maximum number of hops (edges) from start.
    traversable : frozenset of int
        Cell values a robot can enter.

    Returns
    -------
    Set of (row, col) cells reachable from *start* within *max_depth* hops.
    """
    rows, cols = matrix.shape
    visited: Set[Cell] = {start}
    queue: deque[Tuple[Cell, int]] = deque([(start, 0)])
    reachable: Set[Cell] = set()

    while queue:
        (r, c), depth = queue.popleft()
        if matrix[r][c] in traversable:
            reachable.add((r, c))
        if depth < max_depth:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and matrix[nr][nc] in traversable
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), depth + 1))

    return reachable


# ═════════════════════════════════════════════════════════════════════════════
#  Main class
# ═════════════════════════════════════════════════════════════════════════════

class ChargingLayoutGenerator:
    """
    Generate charging-station layouts for an RMFS warehouse grid.

    The class wraps four strategic placement pipelines behind a single
    ``generate()`` call.  The original *data_matrix* is never mutated;
    each call to ``generate()`` works on a fresh deep copy.

    Parameters
    ----------
    data_matrix : array-like of int, shape (R, C)
        Warehouse grid using the cell-value legend at the top of this module.
    config : dict
        Must contain ``'pipeline'`` (int 1–4) plus pipeline-specific keys
        described below.

        Pipeline 1 — Set Cover Approximation
            ``'d'`` (int, default 10)
                Maximum BFS hop-distance from any navigable cell to its
                nearest charger.  Smaller *d* guarantees shorter travel
                but produces more chargers.

        Pipeline 2 — Affinity Propagation
            ``'traffic_matrix'`` (array-like of float, **required**)
                Same shape as *data_matrix*.  Higher value means more
                robot traffic at that cell.
            ``'c'`` (float, default 1.0)
                Regularisation constant in self-similarity; prevents log(0).
            ``'alpha'`` (float, default 1.0)
                Weight of traffic term in placement score.
            ``'beta'`` (float, default 1.0)
                Weight of shelf-proximity penalty in placement score.
            ``'gamma'`` (float, default 1.0)
                Weight of distance-from-hotspot penalty in placement score.
            ``'max_ap_cells'`` (int, default 2 000)
                If the grid has more navigable cells than this, a random
                subsample of this size is used to keep the N×N similarity
                matrix tractable.  Set to 0 to disable subsampling.

        Pipeline 3 — Picking-Station Heuristic
            ``'num_chargers'`` (int, default 10)

        Pipeline 4 — Perimeter Wall Strategy
            ``'num_chargers'`` (int, default 10)

    Raises
    ------
    ValueError
        Unknown pipeline id, or required config key is missing / mismatched.
    ImportError
        scikit-learn not installed when Pipeline 2 is requested.
    """

    def __init__(self, data_matrix, config: dict) -> None:
        self.matrix: Matrix = np.array(data_matrix, dtype=int)
        self.config: dict = config
        self.rows, self.cols = self.matrix.shape

    def __repr__(self) -> str:
        return (
            f"ChargingLayoutGenerator("
            f"shape={self.matrix.shape}, "
            f"pipeline={self.config.get('pipeline', '?')})"
        )

    # ── public entry-point ────────────────────────────────────────────────────

    def generate(self) -> Matrix:
        """
        Execute the configured pipeline and return the modified grid.

        Returns
        -------
        np.ndarray of int, same shape as the input, with CHARGER (2) values
        stamped at positions chosen by the selected pipeline.
        """
        pipeline = int(self.config.get("pipeline", 1))
        work: Matrix = self.matrix.copy()

        dispatch = {
            1: self._pipeline_set_cover,
            2: self._pipeline_affinity,
            3: self._pipeline_picking_station,
            4: self._pipeline_perimeter,
        }

        handler = dispatch.get(pipeline)
        if handler is None:
            raise ValueError(
                f"Unknown pipeline '{pipeline}'.  Valid options are 1, 2, 3, 4."
            )
        return handler(work)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 1 — Set Cover Approximation (Worst-Case Reachability)
    # ═════════════════════════════════════════════════════════════════════════

    def get_reachability_subsets(
        self,
        matrix: Matrix,
        candidate_locations: List[Cell],
        d: int,
        traversable: FrozenSet[int] = TRAVERSABLE,
    ) -> Dict[Cell, Set[Cell]]:
        """
        Run BFS from every candidate location up to depth *d*.

        Parameters
        ----------
        matrix : Matrix
            Current warehouse grid (read-only inside this method).
        candidate_locations : list of (row, col)
            Cells under consideration for charger placement.
        d : int
            Maximum battery-distance in grid hops.
        traversable : frozenset of int
            Cell values the BFS may enter.

        Returns
        -------
        dict mapping each candidate (row, col) → set of navigable cells
        reachable within *d* hops from that candidate.
        """
        return {
            loc: bfs_on_grid(matrix, loc, d, traversable)
            for loc in candidate_locations
        }

    def greedy_set_cover(
        self,
        universe: Set[Cell],
        subsets: Dict[Cell, Set[Cell]],
    ) -> List[Cell]:
        """
        Greedy approximation of minimum set cover (Kundu & Saha 2012).

        At each iteration the candidate whose coverage set intersects the
        most currently uncovered navigable cells is selected.  The algorithm
        terminates when all cells are covered, or when no candidate covers
        any remaining uncovered cell (e.g., disconnected pocket — a warning
        is logged).

        The greedy approach guarantees a solution within O(log n) of the
        optimal minimum-cardinality cover.

        Parameters
        ----------
        universe : set of Cell
            All navigable cells that must be within *d* hops of a charger.
        subsets : dict of Cell → set of Cell
            Reachability subsets from :meth:`get_reachability_subsets`.

        Returns
        -------
        Ordered list of selected charger positions (greedy insertion order).
        """
        uncovered: Set[Cell] = set(universe)

        # Intersect each subset with uncovered upfront for O(|candidates|·|subset|)
        remaining: Dict[Cell, Set[Cell]] = {
            loc: s & uncovered for loc, s in subsets.items()
        }
        selected: List[Cell] = []

        while uncovered and remaining:
            best = max(remaining, key=lambda loc: len(remaining[loc]))
            gain: Set[Cell] = remaining[best]

            if not gain:
                logger.warning(
                    "Set cover: %d navigable cell(s) remain unreachable "
                    "within the maximum BFS distance. "
                    "The grid may contain disconnected pockets.",
                    len(uncovered),
                )
                break

            selected.append(best)
            uncovered -= gain

            # Remove selected candidate and trim every remaining intersection
            del remaining[best]
            remaining = {loc: s - gain for loc, s in remaining.items()}

        return selected

    def apply_set_cover_layout(self, matrix: Matrix, d: int) -> Matrix:
        """
        Run the full set-cover pipeline and record charger positions.

        Candidate charger locations are restricted to warehouse floor (0),
        pod/storage positions (1), and station cells (11, 21) — as defined
        by ``CHARGER_CANDIDATE_VALUES``.  Aisles, rails, intersections, and
        corners are excluded so chargers never block traffic infrastructure.

        BFS uses ``BFS_TRAVERSABLE`` which extends ``TRAVERSABLE`` with
        values 0 and 1, matching the cells robots physically traverse in
        the simulation.

        Selected positions are stored in
        ``self.config["charger_positions"]`` as a list of [row, col] pairs.
        netlogo.py reads this list and registers the cells as chargers
        *without* altering their graph connectivity.

        Steps
        -----
        1. Collect all BFS-traversable cells as the *universe* to cover.
        2. Filter candidates to CHARGER_CANDIDATE_VALUES only.
        3. BFS from each candidate over BFS_TRAVERSABLE to build subsets.
        4. Greedily pick candidates until every reachable cell is covered.
        5. Save selected positions to ``self.config["charger_positions"]``.

        Parameters
        ----------
        matrix : Matrix
            Working copy of the warehouse grid (returned unmodified).
        d : int
            BFS depth limit (maximum robot travel distance to a charger).

        Returns
        -------
        The unmodified matrix.
        """
        # If the caller already supplied an explicit charger_positions list
        # (e.g. for DoE runs that pin a fixed layout from a prior experiment),
        # respect it verbatim instead of re-running the set-cover algorithm.
        # Mirrors the same convention used by the picking-station pipeline
        # (selective_picker_chargers + explicit) at line 881.
        explicit = self.config.get("charger_positions") or []
        if explicit:
            self.config["num_chargers"] = len(explicit)
            logger.info(
                "Set Cover: respecting %d explicit charger position(s) "
                "supplied by caller; skipping greedy set-cover computation.",
                len(explicit),
            )
            return matrix

        # Universe: all cells robots can physically visit.
        universe: Set[Cell] = {
            (int(r), int(c))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if matrix[r][c] in BFS_TRAVERSABLE
        }

        # Candidates: only floor, pod, and station cells.
        candidates: List[Cell] = [
            (int(r), int(c))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if matrix[r][c] in CHARGER_CANDIDATE_VALUES
        ]

        logger.info(
            "Set Cover: %d universe cells, %d candidates, BFS depth d=%d.",
            len(universe), len(candidates), d,
        )

        subsets = self.get_reachability_subsets(matrix, candidates, d, BFS_TRAVERSABLE)
        selected = self.greedy_set_cover(universe, subsets)

        logger.info("Set Cover: %d charger(s) selected.", len(selected))

        # Store positions as JSON-serialisable list (do NOT stamp the grid).
        self.config["charger_positions"] = [[r, c] for r, c in selected]
        self.config["num_chargers"] = len(selected)

        return matrix

    def _pipeline_set_cover(self, work: Matrix) -> Matrix:
        d: int = int(self.config.get("d", 10))
        return self.apply_set_cover_layout(work, d)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 2 — Affinity Propagation (Data-Driven Traffic Clustering)
    # ═════════════════════════════════════════════════════════════════════════

    def run_affinity_propagation(
        self,
        traffic_matrix: Matrix,
        c: float = 1.0,
    ) -> Dict[int, List[Cell]]:
        """
        Cluster candidate cells via Affinity Propagation (Baras et al. 2023).

        Candidate set (Pipeline 2 restriction)
        ──────────────────────────────────────
        Only floor (0) cells are clustered — Pipeline 2 uses the stricter
        ``AP_SAFE_CANDIDATE_VALUES`` filter rather than Pipeline 1's
        ``CHARGER_CANDIDATE_VALUES``.  Pods (1) are excluded because AP
        tends to bunch chargers along pod columns, producing stacks that
        break Dijkstra routing.  Blank-space (99) cells are excluded too
        because netlogo.py only adds routing-graph nodes for values
        {0, 1, 2}, so a charger placed on 99 is unreachable at runtime.

        Similarity matrix construction (Baras et al. 2023, Eq. 1)
        ─────────────────────────────────────────────────────────
        Off-diagonal  S[i, j] = −Manhattan(i, j)²
            Nearby cells in the grid are more similar; the squared penalty
            strongly discourages merging distant cells into one cluster.

        Diagonal (preference)  S[i, i] = +log(c + traffic[i])
            The preference controls how willing a cell is to become an
            exemplar (cluster centre).  High-traffic cells get a HIGHER
            self-similarity and therefore attract more clusters, exactly
            as the paper's text describes.  The sign was flipped relative
            to the earlier implementation, which had the opposite behaviour.

        Requires
        --------
        scikit-learn (``pip install scikit-learn``).

        Parameters
        ----------
        traffic_matrix : Matrix
            Robot-traffic density; same shape as the warehouse grid.
        c : float
            Regularisation constant to avoid log(0).  Default 1.0.

        Returns
        -------
        dict mapping integer cluster label → list of (row, col) cells.
        """
        try:
            from sklearn.cluster import AffinityPropagation as _AP
        except ImportError as exc:
            raise ImportError(
                "Pipeline 2 requires scikit-learn.\n"
                "Install it with:  pip install scikit-learn"
            ) from exc

        # Candidates: floor (0) cells only — pods and blank-space excluded.
        nav_cells: List[Cell] = sorted(
            (int(r), int(c))
            for r in range(self.rows)
            for c in range(self.cols)
            if self.matrix[r][c] in AP_SAFE_CANDIDATE_VALUES
        )
        n = len(nav_cells)
        if n == 0:
            logger.warning(
                "AP: no safe candidate cells (floor) found in the grid."
            )
            return {}

        # ── optional subsampling to keep the N×N matrix tractable ──────────
        max_cells: int = int(self.config.get("max_ap_cells", 2_000))
        if max_cells > 0 and n > max_cells:
            logger.warning(
                "AP: %d candidate cells exceeds max_ap_cells=%d; "
                "deterministically subsampling to keep computation tractable.",
                n, max_cells,
            )
            rng = np.random.default_rng(42)
            idxs = sorted(rng.choice(n, size=max_cells, replace=False).tolist())
            nav_cells = [nav_cells[i] for i in idxs]
            n = max_cells

        # ── build N×N similarity matrix ─────────────────────────────────────
        S = np.empty((n, n), dtype=np.float64)
        coords = np.array(nav_cells, dtype=np.float64)  # shape (N, 2)

        # Vectorised off-diagonal: S[i,j] = −Manhattan(i,j)²
        for i in range(n):
            diff = np.abs(coords - coords[i])      # (N, 2)  row/col deltas
            mdist = diff[:, 0] + diff[:, 1]        # (N,)    Manhattan distance
            S[i] = -(mdist ** 2)

        # Diagonal (Baras et al. 2023, Eq. 1): +log(c + traffic[i])
        for i, (r, col_idx) in enumerate(nav_cells):
            traffic_val = float(traffic_matrix[r][col_idx])
            S[i, i] = math.log(c + traffic_val)

        # ── fit AP with precomputed similarity ──────────────────────────────
        ap = _AP(
            affinity="precomputed",
            random_state=42,
            max_iter=400,
            convergence_iter=25,
            damping=0.9,
        )
        ap.fit(S)

        # ── group cells by cluster label ─────────────────────────────────────
        clusters: Dict[int, List[Cell]] = {}
        for idx, label in enumerate(ap.labels_):
            clusters.setdefault(int(label), []).append(nav_cells[idx])

        logger.info("AP: produced %d cluster(s) from %d cells.", len(clusters), n)
        return clusters

    def calculate_placement_score(
        self,
        cluster_cells: List[Cell],
        traffic_matrix: Matrix,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> Dict[Cell, float]:
        """
        Score every candidate cell in a cluster for charger placement.

        Scoring function (Baras et al. 2023, Eq. 2)
        ──────────────────────────────────────────
        Score(c) = α · TrafficDesirability(c)
                 − β · ProximityToShelves(c)
                 − γ · DistanceFromHighTrafficCell(c)

        where:
          ``TrafficDesirability(c) = 1 − (Traffic(c) − mean(Traffic))²``
              Inverted bell around the cluster mean.  Rewards cells whose
              traffic is close to the typical robot flow through the cluster
              and penalises extremes (idle pockets AND congested hotspots).

          ``ProximityToShelves(c)``
              Binary penalty: 1 if any 4-connected neighbour in the original
              warehouse grid (``self.matrix``) is a storage pod (value 1),
              else 0.  Keeps chargers out of pod retrieval lanes.

          ``DistanceFromHighTrafficCell(c)``
              Manhattan distance from *c* to the cluster's hottest cell
              (max traffic).  The paper includes this term to keep chargers
              close to peak demand and minimise robot downtime when they
              need to recharge.

        Parameters
        ----------
        cluster_cells : list of (row, col)
        traffic_matrix : Matrix
        alpha, beta, gamma : float
            Relative importance weights.

        Returns
        -------
        dict mapping each cell to its float score.
        """
        if not cluster_cells:
            return {}

        # Cluster-local mean traffic drives the inverted bell.
        vals = np.array(
            [float(traffic_matrix[r][c]) for r, c in cluster_cells],
            dtype=np.float64,
        )
        mean_traffic = float(vals.mean())

        # Hotspot: cell with maximum traffic in this cluster.
        hot_cell: Cell = max(
            cluster_cells, key=lambda rc: traffic_matrix[rc[0]][rc[1]]
        )

        scores: Dict[Cell, float] = {}
        for r, c in cluster_cells:
            traffic_val = float(traffic_matrix[r][c])

            # Binary penalty: direct neighbour of a storage pod?
            adj_to_pod = any(
                0 <= r + dr < self.rows
                and 0 <= c + dc < self.cols
                and self.matrix[r + dr][c + dc] == POD
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            prox_penalty = 1.0 if adj_to_pod else 0.0

            td = 1.0 - (traffic_val - mean_traffic) ** 2
            dist_from_hot = manhattan(r, c, hot_cell[0], hot_cell[1])

            scores[(r, c)] = (
                alpha * td
                - beta  * prox_penalty
                - gamma * dist_from_hot
            )

        return scores

    def apply_traffic_layout(
        self, matrix: Matrix, traffic_matrix: Matrix
    ) -> Matrix:
        """
        AP-cluster the grid, score candidates per cluster, record chargers.

        One charger is placed per AP cluster, at the cell with the highest
        placement score.  Positions are stored in
        ``self.config["charger_positions"]`` (overlay approach used by
        Pipeline 1); the grid itself is NOT mutated so the directed graph
        rail/intersection cells remain intact.

        Parameters
        ----------
        matrix : Matrix
            Working copy of the warehouse grid (returned unmodified).
        traffic_matrix : Matrix
            Robot-traffic density grid (same shape as *matrix*).

        Returns
        -------
        The unmodified matrix.
        """
        c     = float(self.config.get("c",     1.0))
        alpha = float(self.config.get("alpha", 1.0))
        beta  = float(self.config.get("beta",  1.0))
        gamma = float(self.config.get("gamma", 1.0))
        # Optional hard cap: if AP yields more clusters than we want chargers,
        # keep only the top-`max_chargers` by placement score.  Default None
        # means "one charger per AP cluster" (paper-faithful behaviour).
        max_chargers = self.config.get("num_chargers")
        if max_chargers is not None:
            max_chargers = int(max_chargers)

        clusters = self.run_affinity_propagation(traffic_matrix, c)
        ranked: List[Tuple[float, Cell, int]] = []  # (score, cell, cluster_label)
        all_cell_scores: Dict[Cell, float] = {}     # union across clusters

        for label, cells in clusters.items():
            scores = self.calculate_placement_score(
                cells, traffic_matrix, alpha, beta, gamma
            )
            if not scores:
                continue
            all_cell_scores.update(scores)
            best_cell = max(scores, key=scores.__getitem__)
            ranked.append((scores[best_cell], best_cell, label))

        # Rank cluster winners globally (highest score first) and apply cap.
        ranked.sort(key=lambda t: t[0], reverse=True)
        if max_chargers is not None and len(ranked) > max_chargers:
            logger.info(
                "AP traffic layout: %d cluster(s) produced, capping to top %d "
                "by placement score.",
                len(ranked), max_chargers,
            )
            ranked = ranked[:max_chargers]

        selected: List[Cell] = [cell for _, cell, _ in ranked]
        for score, cell, label in ranked:
            logger.debug(
                "AP cluster %d → charger at (%d, %d)  score=%.3f",
                label, cell[0], cell[1], score,
            )

        # Top-up: if AP produced fewer clusters than num_chargers requested,
        # fill remaining slots from top-scored candidates enforcing a
        # minimum Manhattan separation from already-selected chargers to
        # prevent stacking in a single corridor.
        min_sep = int(self.config.get("ap_min_separation", 3))
        if max_chargers is not None and len(selected) < max_chargers:
            remaining = max_chargers - len(selected)
            pool = [
                (s, cell) for cell, s in all_cell_scores.items()
                if cell not in set(selected)
            ]
            pool.sort(key=lambda t: t[0], reverse=True)
            added = 0
            for score, cell in pool:
                if added >= remaining:
                    break
                if all(manhattan(cell[0], cell[1], r, c) >= min_sep
                       for r, c in selected):
                    selected.append(cell)
                    added += 1
                    logger.debug(
                        "AP top-up → charger at (%d, %d)  score=%.3f",
                        cell[0], cell[1], score,
                    )
            logger.info(
                "AP traffic layout: topped-up with %d extra charger(s) "
                "(min_separation=%d).", added, min_sep,
            )

        logger.info("AP traffic layout: %d charger(s) selected.", len(selected))

        self.config["charger_positions"] = [[int(r), int(c)] for r, c in selected]
        self.config["num_chargers"] = len(selected)

        return matrix

    def _build_station_proximity_traffic(self, tau: float = 8.0) -> Matrix:
        """
        Heuristic traffic density when no measured traffic_matrix is given.

        Multi-source BFS from every picking (11) and replenishment (21) cell
        over BFS_TRAVERSABLE; each reachable cell gets
            traffic(c) = exp(-hop_distance / tau).

        Robots converge on stations, so cells closer to stations see more
        pod-delivery flows.  tau controls decay: smaller tau = more
        concentrated hotspots.
        """
        rows, cols = self.matrix.shape
        dist = np.full((rows, cols), 10_000, dtype=np.int32)
        q: deque[Cell] = deque()
        for r in range(rows):
            for c in range(cols):
                if self.matrix[r][c] in (11, 21):
                    dist[r, c] = 0
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows and 0 <= nc < cols
                    and self.matrix[nr, nc] in BFS_TRAVERSABLE
                    and dist[nr, nc] > dist[r, c] + 1
                ):
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        traffic = np.where(
            dist < 10_000,
            np.exp(-dist.astype(np.float64) / tau),
            0.0,
        )
        return traffic

    def _pipeline_affinity(self, work: Matrix) -> Matrix:
        raw_traffic = self.config.get("traffic_matrix")
        if raw_traffic is None:
            tau = float(self.config.get("traffic_tau", 8.0))
            logger.info(
                "Pipeline 2: no traffic_matrix supplied, using station-proximity "
                "heuristic with tau=%.2f.", tau,
            )
            traffic: Matrix = self._build_station_proximity_traffic(tau=tau)
        else:
            traffic = np.array(raw_traffic, dtype=np.float64)
            if traffic.shape != self.matrix.shape:
                raise ValueError(
                    f"traffic_matrix shape {traffic.shape} does not match "
                    f"data_matrix shape {self.matrix.shape}."
                )
        return self.apply_traffic_layout(work, traffic)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 3 — Picking-Station Heuristic (Opportunity Charging)
    # ═════════════════════════════════════════════════════════════════════════

    def find_picking_stations(self, matrix: Matrix) -> List[Cell]:
        """
        Return the coordinates of picking-station cells (value 11) themselves.

        These are the exact cells where chargers should be co-located.
        Replenishment stations (value 21) are excluded — chargers belong
        only at the picking side of the warehouse.

        Parameters
        ----------
        matrix : Matrix

        Returns
        -------
        Sorted list of (row, col) picking-station cells, in row-major order.
        """
        rows, cols = matrix.shape
        station_cells: List[Cell] = []

        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 11:  # picking station only
                    station_cells.append((r, c))

        return sorted(station_cells)

    def apply_picking_station_layout(
        self, matrix: Matrix, num_chargers: int
    ) -> Matrix:
        """
        Co-locate chargers at picking-station cells (value 11).

        The grid values are NOT overwritten because netlogo.py's station-
        creation logic (value 14) checks ``obj_left_value == 11``.
        Instead, netlogo.py registers value-11 cells as charger_cells
        directly.  This method selects which stations receive chargers
        and logs the result for traceability.

        Parameters
        ----------
        matrix : Matrix
            Working copy (returned unmodified).
        num_chargers : int
            Target number of chargers to place.

        Returns
        -------
        The matrix (unchanged — charger registration happens in netlogo.py).
        """
        station_cells = self.find_picking_stations(matrix)
        if not station_cells:
            logger.warning(
                "Pipeline 3: no picking-station cells (value 11) found."
            )
            return matrix

        n_place = min(num_chargers, len(station_cells))
        selected_pickers = _evenly_spaced_sample(station_cells, n_place)

        # Selective mode: only K out of 5 pickers get chargers.  Write the
        # picker's value-14 entry cells to charger_positions so netlogo.py
        # registers them via the overlay mechanism (bypassing the blanket
        # pipeline-3 special case).  A picker at (r, c) is paired with two
        # value-14 cells immediately to its right — at (r, c+1) and
        # (r+1, c+1) — so we scan a small box rather than just 4-neighbours.
        selective = bool(self.config.get("selective_picker_chargers", False))
        # If the caller already provided an explicit charger_positions list
        # (e.g. the per-station gate+non-gate layout in run_one.py), keep it
        # verbatim and skip auto-computation.
        explicit = self.config.get("charger_positions") or []
        if selective and explicit:
            self.config["num_chargers"] = len(explicit)
            logger.info(
                "Picking-station layout (selective, explicit): "
                "%d charger cell(s) supplied by caller: %s",
                len(explicit), explicit,
            )
        elif selective:
            overlay: List[Cell] = []
            for r, c in selected_pickers:
                for dr in (-1, 0, 1, 2):
                    for dc in (1, 2):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self.rows
                            and 0 <= nc < self.cols
                            and self.matrix[nr][nc] == 14
                        ):
                            overlay.append((int(nr), int(nc)))
            # Deduplicate while preserving order.
            seen: set = set()
            overlay = [cell for cell in overlay
                       if not (cell in seen or seen.add(cell))]
            self.config["charger_positions"] = [[r, c] for r, c in overlay]
            self.config["num_chargers"] = len(overlay)
            logger.info(
                "Picking-station layout (selective): %d picker(s) chosen → "
                "%d value-14 charger cell(s): %s",
                len(selected_pickers), len(overlay), overlay,
            )
        else:
            logger.info(
                "Picking-station layout: %d charger(s) co-located with "
                "picking stations from %d available station cell(s): %s",
                len(selected_pickers), len(station_cells), selected_pickers,
            )
        return matrix

    def _pipeline_picking_station(self, work: Matrix) -> Matrix:
        num_chargers: int = int(self.config.get("num_chargers", 10))
        return self.apply_picking_station_layout(work, num_chargers)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 4 — Perimeter Wall Strategy (Isolation / Control)
    # ═════════════════════════════════════════════════════════════════════════

    def find_perimeter_cells(self, matrix: Matrix) -> List[Cell]:
        """
        Mencari aisle vertikal (garis hijau) tepat sebelum Replenishment Station,
        dan MENGHINDARI pintu masuk stasiun agar robot tidak terhalang.
        """
        rows, cols = matrix.shape
        perimeter: List[Cell] = []

        # 1. Cari batas paling KIRI dari stasiun Replenishment (angka 21 - 29)
        min_station_col = cols
        for r in range(rows):
            for c in range(cols):
                if 21 <= matrix[r][c] <= 29:
                    if c < min_station_col:
                        min_station_col = c

        # 2. Garis hijau berada tepat 1 atau 2 kolom di sebelah kiri stasiun tersebut
        target_col = min_station_col - 1

        if target_col <= 0 or target_col >= cols:
            target_col = cols - 2 # Fallback aman jika stasiun tidak ditemukan

        # 3. Kumpulkan titik yang aman untuk diletakkan charger
        for r in range(rows):
            if matrix[r][target_col] in TRAVERSABLE:
                # -- ATURAN ANTI-BLOKIR (DO NOT BLOCK ENTRANCE) --
                # Kita cek area di sebelah kanan titik ini. 
                # Jika ada struktur stasiun (21-29) atau persimpangan (3) di baris yang sama,
                # berarti titik ini persis berada di depan pintu masuk. KITA LEWATI!
                is_blocking = False
                for c in range(target_col + 1, cols):
                    val = matrix[r][c]
                    if (21 <= val <= 29) or val == 3:
                        is_blocking = True
                        break
                
                # Jika baris ini aman (tidak sejajar dengan pintu stasiun), 
                # tambahkan sebagai kandidat lokasi charger
                if not is_blocking:
                    perimeter.append((r, target_col))

        return perimeter
    # ═════════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _navigable_cells(self, matrix: Matrix) -> Set[Cell]:
        """Return the set of all (row, col) cells that Kiva can drive on."""
        return {
            (int(r), int(c))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if matrix[r][c] in TRAVERSABLE
        }
    
    def apply_perimeter_layout(
        self, matrix: Matrix, num_chargers: int
    ) -> Matrix:
        """
        Spread *num_chargers* chargers evenly along the perimeter cells.

        Placement is deterministic (evenly spaced by index).  If
        *num_chargers* exceeds the number of perimeter cells, every
        perimeter cell receives a charger.

        Selected positions are stored in
        ``self.config["charger_positions"]`` as a list of [row, col] pairs
        (same convention as P1/P2/P3).  The grid itself is not mutated —
        netlogo.py reads positions from the config overlay.

        Parameters
        ----------
        matrix : Matrix
            Working copy (returned unmodified).
        num_chargers : int

        Returns
        -------
        The unmodified matrix.
        """
        perimeter = self.find_perimeter_cells(matrix)
        if not perimeter:
            logger.warning(
                "Pipeline 4: no navigable cells found on the top or bottom rows."
            )
            self.config["charger_positions"] = []
            self.config["num_chargers"] = 0
            return matrix

        n_place = min(num_chargers, len(perimeter))
        selected = _evenly_spaced_sample(perimeter, n_place)

        self.config["charger_positions"] = [[int(r), int(c)] for r, c in selected]
        self.config["num_chargers"] = len(selected)

        logger.info(
            "Perimeter layout: %d charger(s) spread across %d perimeter cell(s).",
            len(selected), len(perimeter),
        )
        return matrix

    def _pipeline_perimeter(self, work: Matrix) -> Matrix:
        num_chargers: int = int(self.config.get("num_chargers", 10))
        return self.apply_perimeter_layout(work, num_chargers)

