"use strict";

// Minimum-cost bipartite assignment — the Kuhn-Munkres (Hungarian) algorithm on
// a RECTANGULAR cost matrix. Pure: a cost matrix in, a list of (row, col) pairs
// out. No I/O, no mutation of the input.
//
// Self-contained (no external dependency) ON PURPOSE: the build environment must
// not run `npm install`, and the per-cluster assignment is the one piece of the
// v2 algorithm that MUST be offline-unit-tested (spec §4.5 / refinement D). A
// small vetted implementation we own and test outright is the poka-yoke choice.
//
// Spec contract (§4.5, §5, §6):
//   - Returns the assignment that MINIMISES the total cost (distance), NOT a
//     greedy lowest-first pick. This is the property greedy lacks: an early
//     lowest pick can steal the best target from a better-suited later row.
//   - Rectangular: rows = set side, cols = target side; the smaller side is fully
//     matched, the larger side has leftovers (returned unmatched by the caller).
//   - The caller has ALREADY capped costs at τ and partitioned into components,
//     so matrices reaching here are tiny (a handful per side). This implementation
//     is O(n³) in the larger dimension — correct precisely because n is small.
//
// Algorithm: the O(n³) Jonker-style augmenting-path Hungarian on a padded square
// matrix (pad the short side with sentinel BIG cost so padded cells are never
// chosen when a real assignment exists). We use a finite BIG (not Infinity) so
// potentials stay finite; the caller never feeds a cost anywhere near BIG.

const BIG = 1e9;

// Solve the square assignment problem for an n×n cost matrix using the
// O(n³) potentials/augmenting-path method. Returns `colForRow`: an Int array
// where colForRow[r] is the column assigned to row r. Pure.
//
// @allowloop: a numeric matrix solver has no map/filter/reduce equivalent; the
// nested loops below ARE the algorithm. Contained in this single low-level
// module (Coding-best-practices §4) and exhaustively unit-tested offline.
function solveSquare(cost) {
  const n = cost.length;
  // Potentials and the augmenting-path bookkeeping use 1-based indexing with a
  // virtual row/column 0, the classic e-maxx formulation.
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0); // p[col] = row assigned to col (0 = none)
  const way = new Array(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const colForRow = new Array(n).fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j] >= 1) colForRow[p[j] - 1] = j - 1;
  }
  return colForRow;
}

// Pad a rectangular cost matrix (rows × cols) to square with BIG sentinels.
function padToSquare(cost, rows, cols) {
  const n = Math.max(rows, cols);
  const square = [];
  for (let r = 0; r < n; r++) {
    const rowVals = [];
    for (let c = 0; c < n; c++) {
      rowVals.push(r < rows && c < cols ? cost[r][c] : BIG);
    }
    square.push(rowVals);
  }
  return square;
}

// Minimum-cost assignment over a rectangular cost matrix.
//
// `cost` is a rows×cols array of finite non-negative numbers. Returns an array of
// { row, col, cost } for every REAL cell that was assigned (padded sentinel cells
// are dropped). The smaller side is fully matched; unmatched rows/cols are simply
// absent from the result. An empty matrix yields [].
function minCostAssignment(cost) {
  const rows = cost.length;
  if (rows === 0) return [];
  const cols = cost[0].length;
  if (cols === 0) return [];

  const square = padToSquare(cost, rows, cols);
  const colForRow = solveSquare(square);

  const assignments = [];
  for (let r = 0; r < rows; r++) {
    const c = colForRow[r];
    if (c >= 0 && c < cols) {
      assignments.push({ row: r, col: c, cost: cost[r][c] });
    }
  }
  return assignments;
}

module.exports = { minCostAssignment, BIG };
