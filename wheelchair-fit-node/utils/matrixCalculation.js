/**
 * Dot product of column vector by row vector.
 */
function dotProduct(column, row) {
  if (column.length !== row.length) {
    throw new Error("Column and row must be the same length");
  }
  return column.reduce((sum, value, index) => sum + value * row[index], 0);
}

export { dotProduct };
