"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const { minCostAssignment } = require("../src/hungarian");

function totalCost(assignment) {
  return assignment.reduce((sum, a) => sum + a.cost, 0);
}

test("empty matrix -> no assignments", () => {
  assert.deepEqual(minCostAssignment([]), []);
  assert.deepEqual(minCostAssignment([[]]), []);
});

test("1x1 picks the only cell", () => {
  const a = minCostAssignment([[0.3]]);
  assert.deepEqual(a, [{ row: 0, col: 0, cost: 0.3 }]);
});

test("2x2 chooses the minimum-total assignment, not the greedy one", () => {
  // Greedy picks the global min 0.10 (row0,col0), forcing row1->col1 at 0.90,
  // total 1.00. Optimal picks row0->col1 (0.20) + row1->col0 (0.30) = 0.50.
  const cost = [
    [0.1, 0.2],
    [0.3, 0.9],
  ];
  const a = minCostAssignment(cost);
  assert.equal(a.length, 2);
  assert.equal(totalCost(a).toFixed(4), "0.5000");
  const byRow = new Map(a.map((x) => [x.row, x.col]));
  assert.equal(byRow.get(0), 1);
  assert.equal(byRow.get(1), 0);
});

test("classic 3x3 has known optimal cost", () => {
  // Standard textbook matrix; optimal assignment total = 5.
  const cost = [
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2],
  ];
  const a = minCostAssignment(cost);
  assert.equal(a.length, 3);
  assert.equal(totalCost(a), 5);
});

test("rectangular: more rows than cols — only cols-many matched, no double-use", () => {
  const cost = [
    [0.1, 0.8],
    [0.7, 0.2],
    [0.3, 0.3],
  ];
  const a = minCostAssignment(cost);
  assert.equal(a.length, 2); // min(3,2)
  const cols = a.map((x) => x.col).sort();
  assert.deepEqual(cols, [0, 1]); // each column used once
  const rows = new Set(a.map((x) => x.row));
  assert.equal(rows.size, a.length); // no row reused
});

test("rectangular: more cols than rows — only rows-many matched", () => {
  const cost = [[0.1, 0.4, 0.9]];
  const a = minCostAssignment(cost);
  assert.equal(a.length, 1);
  assert.equal(a[0].col, 0); // cheapest column
});

test("all-equal costs still produce a valid 1:1 assignment", () => {
  const cost = [
    [0.5, 0.5],
    [0.5, 0.5],
  ];
  const a = minCostAssignment(cost);
  assert.equal(a.length, 2);
  assert.deepEqual(a.map((x) => x.col).sort(), [0, 1]);
});
