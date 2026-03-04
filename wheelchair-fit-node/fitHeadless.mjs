#!/usr/bin/env node
/**
 * Headless wheelchair fit: same math as the web tool. No browser.
 * Reads JSON array from stdin: [{ id, gender, age, heightCm, bmi, wheelchairType }, ...]
 * Outputs JSON array to stdout: [{ id, seatWidth, seatDepth, seatPanHeight }, ...]
 *
 * Run: node fitHeadless.mjs < input.json
 * Requires: model/mean_model_tri.ply and model/Anth2Data.csv in this directory.
 */

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import * as THREE from "three";
import { dotProduct } from "./utils/matrixCalculation.js";
import { inToM, mToIn } from "./utils/unitConverter.js";
import {
  getHumanModelWorldCoordinates,
  calculateDistanceBetweenPoints,
} from "./utils/meshUtils.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "model");
const MEAN_PLY = path.join(MODEL_DIR, "mean_model_tri.ply");
const ANTH_CSV = path.join(MODEL_DIR, "Anth2Data.csv");

const PRED_ANTH_NUM = 0;
const PRED_LANDMARK_NUM = 0;
const HUMAN_ROTATION = new THREE.Vector3(
  (-73 * Math.PI) / 180,
  0,
  (90 * Math.PI) / 180
);

const LEFT_THIGH_INDEX = 1359;
const RIGHT_THIGH_INDEX = 3264;
const BEHIND_KNEE_INDEX = 3491;
const FARTHEST_BACK_INDEX = 2242;
const SCAPULA_INDEX = 2249;
const SPINAL_INDEX = 212;
const HUMAN_BOTTOM_INDEX = 12976;

function parsePlyVertices(filePath) {
  const buf = fs.readFileSync(filePath);
  const headerEnd = buf.indexOf(Buffer.from("end_header\n"));
  if (headerEnd < 0) throw new Error("PLY: end_header not found");
  const header = buf.subarray(0, headerEnd).toString("utf8");
  const vertexMatch = header.match(/element\s+vertex\s+(\d+)/i);
  if (!vertexMatch) throw new Error("PLY: element vertex not found");
  const vertexCount = parseInt(vertexMatch[1], 10);
  const isBinary = /format\s+binary/i.test(header);
  const geometryZero = [];
  let offset = headerEnd + "end_header\n".length;
  if (isBinary) {
    const littleEndian = /binary_little_endian/i.test(header);
    for (let i = 0; i < vertexCount; i++) {
      const x = littleEndian ? buf.readFloatLE(offset) : buf.readFloatBE(offset);
      const y = littleEndian ? buf.readFloatLE(offset + 4) : buf.readFloatBE(offset + 4);
      const z = littleEndian ? buf.readFloatLE(offset + 8) : buf.readFloatBE(offset + 8);
      geometryZero.push({ x, y, z });
      offset += 12;
    }
  } else {
    const text = buf.subarray(offset).toString("utf8");
    const lines = text.split(/\r?\n/);
    for (let k = 0; k < vertexCount && k < lines.length; k++) {
      const parts = lines[k].trim().split(/\s+/);
      geometryZero.push({
        x: parseFloat(parts[0]),
        y: parseFloat(parts[1]),
        z: parseFloat(parts[2]),
      });
    }
  }
  return geometryZero;
}

function parseCsvNumbers(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  const rows = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const row = trimmed.split(",").map((s) => parseFloat(s.trim()));
    if (row.some((n) => Number.isNaN(n))) continue;
    rows.push(row);
  }
  return rows;
}

function buildGeometryFromZero(geometryZero) {
  const positions = new Float32Array(geometryZero.length * 3);
  for (let i = 0; i < geometryZero.length; i++) {
    positions[i * 3] = geometryZero[i].x;
    positions[i * 3 + 1] = geometryZero[i].y;
    positions[i * 3 + 2] = geometryZero[i].z;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return geom;
}

function updateHumanGeometry(anth, geometry, geometryZero, PCAdata) {
  const positions = geometry.attributes.position;
  const Anths = [
    anth.STUDY,
    anth.GENDER,
    anth.STATURE,
    anth.SHS,
    anth.BMI,
    anth.AGE,
    1,
  ];
  const skipNum = PRED_ANTH_NUM + PRED_LANDMARK_NUM * 3;
  for (let i = 0; i < positions.count; i++) {
    const diffx = dotProduct(Anths, PCAdata[skipNum + i * 3 + 0]);
    const diffy = dotProduct(Anths, PCAdata[skipNum + i * 3 + 1]);
    const diffz = dotProduct(Anths, PCAdata[skipNum + i * 3 + 2]);
    positions.setXYZ(
      i,
      geometryZero[i].x + diffx,
      geometryZero[i].y + diffy,
      geometryZero[i].z + diffz
    );
  }
  positions.needsUpdate = true;
}

function createHumanMesh(geometry) {
  const material = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    specular: 0xaaaaaa,
    shininess: 20,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.set(HUMAN_ROTATION.x, HUMAN_ROTATION.y, HUMAN_ROTATION.z);
  mesh.scale.set(0.001, 0.001, 0.001);
  mesh.position.set(0, 0, 0);
  mesh.updateMatrixWorld(true);
  return mesh;
}

function calculateOptimalSeatWidth(humanMesh) {
  const thighWidth = calculateDistanceBetweenPoints(
    humanMesh,
    LEFT_THIGH_INDEX,
    RIGHT_THIGH_INDEX,
    true,
    false,
    false
  );
  const padding = 1;
  return mToIn(thighWidth) + padding * 2;
}

function calculateOptimalSeatDepth(humanMesh, wheelchairType) {
  const thighLength = calculateDistanceBetweenPoints(
    humanMesh,
    BEHIND_KNEE_INDEX,
    FARTHEST_BACK_INDEX,
    false,
    false,
    true
  );
  let padding = wheelchairType === "powered" ? -1 : 0;
  return mToIn(thighLength) + padding;
}

function calculateOptimalBackHeight(humanMesh, seatWidth, wheelchairType, wheelchairParams) {
  const scapulaWorld = getHumanModelWorldCoordinates(humanMesh, SCAPULA_INDEX);
  const spinalWorld = getHumanModelWorldCoordinates(humanMesh, SPINAL_INDEX);
  const seatWidthM = inToM(seatWidth);
  if (wheelchairType === "manual") {
    return mToIn(scapulaWorld.y - seatWidthM);
  }
  const seatBottom = inToM(
    wheelchairParams.seatPanHeight + wheelchairParams.seatCushThick
  );
  return mToIn(spinalWorld.y - seatBottom);
}

function calculateOptimalSeatPanHeight(humanMesh, wheelchairType, wheelchairParams) {
  if (wheelchairType === "powered") {
    const seatPanHeightTop = humanMesh.position.clone();
    const humanBottom = getHumanModelWorldCoordinates(humanMesh, HUMAN_BOTTOM_INDEX);
    const clearance = 4.5;
    const padding = 1;
    return (
      mToIn(seatPanHeightTop.y) -
      mToIn(humanBottom.y) +
      clearance -
      wheelchairParams.seatCushThick +
      padding
    );
  }
  return wheelchairParams.seatPanHeight;
}

function validateWheelchairParams(params) {
  const RANGES = {
    seatWidth: { min: 14, max: 20 },
    seatBackHeight: { min: 16, max: 26 },
    seatDepth: { min: 15, max: 25 },
  };
  for (const [key, range] of Object.entries(RANGES)) {
    if (params[key] != null) {
      if (params[key] < range.min) params[key] = range.min;
      else if (params[key] > range.max) params[key] = range.max;
    }
  }
}

function runFit(anth, wheelchairType) {
  const baseParams = {
    seatWidth: 17,
    seatDepth: 19,
    seatPanHeight: 16,
    seatCushThick: 4,
  };
  const geometryZero = runFit.geometryZero;
  const PCAdata = runFit.PCAdata;
  const geom = buildGeometryFromZero(geometryZero);
  updateHumanGeometry(anth, geom, geometryZero, PCAdata);
  geom.computeVertexNormals();
  const humanMesh = createHumanMesh(geom);

  const optimalParams = { ...baseParams };
  optimalParams.seatWidth = calculateOptimalSeatWidth(humanMesh);
  optimalParams.seatBackHeight = calculateOptimalBackHeight(
    humanMesh,
    optimalParams.seatWidth,
    wheelchairType,
    baseParams
  );
  optimalParams.seatDepth = calculateOptimalSeatDepth(humanMesh, wheelchairType);
  optimalParams.seatPanHeight = calculateOptimalSeatPanHeight(
    humanMesh,
    wheelchairType,
    baseParams
  );
  validateWheelchairParams(optimalParams);
  return {
    seatWidth: optimalParams.seatWidth,
    seatDepth: optimalParams.seatDepth,
    seatPanHeight: optimalParams.seatPanHeight,
  };
}

function mapGender(g) {
  const u = String(g ?? "").toUpperCase();
  if (u === "M" || u === "MALE") return 1;
  return -1;
}

function mapWheelchairType(t) {
  const u = String(t ?? "").toLowerCase();
  if (u === "power" || u === "powered" || u === "stroller") return "powered";
  return "manual";
}

async function main() {
  if (!fs.existsSync(MEAN_PLY)) {
    console.error(`Missing ${MEAN_PLY}. Put model files in wheelchair-fit-node/model/`);
    process.exit(1);
  }
  if (!fs.existsSync(ANTH_CSV)) {
    console.error(`Missing ${ANTH_CSV}`);
    process.exit(1);
  }

  runFit.geometryZero = parsePlyVertices(MEAN_PLY);
  runFit.PCAdata = parseCsvNumbers(ANTH_CSV);

  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const input = chunks.join("");
  let rows = [];
  try {
    rows = JSON.parse(input || "[]");
  } catch (e) {
    console.error("Invalid JSON on stdin:", e.message);
    process.exit(1);
  }
  if (!Array.isArray(rows) || rows.length === 0) {
    console.log(JSON.stringify([]));
    process.exit(0);
  }

  const results = [];
  for (const row of rows) {
    const id = row.id ?? row.ID ?? "";
    const heightCm = row.heightCm ?? row["Height (cm)"] ?? 170;
    const statureMm = Math.round(Number(heightCm) * 10);
    const anth = {
      STUDY: 1,
      GENDER: mapGender(row.gender ?? row.Gender),
      STATURE: statureMm,
      SHS: 0.52,
      BMI: Number(row.bmi ?? row.BMI ?? 26),
      AGE: Number(row.age ?? row.Age ?? 40),
    };
    const wheelchairType = mapWheelchairType(
      row.wheelchairType ?? row["WC Type"] ?? "manual"
    );
    try {
      const fitted = runFit(anth, wheelchairType);
      results.push({ id, ...fitted });
    } catch (err) {
      console.error(`Fit failed for id=${id}:`, err.message);
      results.push({
        id,
        seatWidth: null,
        seatDepth: null,
        seatPanHeight: null,
      });
    }
  }
  console.log(JSON.stringify(results));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
