const inToMFactor = 0.0254;
const mToInFactor = 39.3701;

function inToM(num) {
  return num * inToMFactor;
}

function mToIn(num) {
  return num * mToInFactor;
}

export { inToM, mToIn };
