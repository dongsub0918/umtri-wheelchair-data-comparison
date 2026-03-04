import * as THREE from "three";

export function getHumanModelWorldCoordinates(humanMesh, index) {
  const positions = humanMesh.geometry.attributes.position;
  const local = [
    positions.getX(index),
    positions.getY(index),
    positions.getZ(index),
  ];
  return new THREE.Vector3(...local).applyMatrix4(humanMesh.matrixWorld);
}

export function calculateDistanceBetweenPoints(
  humanMesh,
  index1,
  index2,
  useX = true,
  useY = true,
  useZ = true
) {
  const point1World = getHumanModelWorldCoordinates(humanMesh, index1);
  const point2World = getHumanModelWorldCoordinates(humanMesh, index2);
  let distance = 0;
  if (useX) distance += (point1World.x - point2World.x) ** 2;
  if (useY) distance += (point1World.y - point2World.y) ** 2;
  if (useZ) distance += (point1World.z - point2World.z) ** 2;
  return Math.sqrt(distance);
}
