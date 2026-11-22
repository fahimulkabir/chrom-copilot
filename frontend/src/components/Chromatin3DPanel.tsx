// Renders smooth 3D chromatin polymer splines and clickable beads for every active cohort (GM12878 in blue, K562 in red), supports Point-to-Query raycast picking (clicking any bead highlights that exact bin in the 2D Hi-C heatmap and updates ViewportState), and performs smooth Query-to-View camera flights when camera_3d updates.

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Bead3D } from "../services/api";
import { useViewportStore } from "../store/useViewportStore";

interface PolymerProps {
  beads: Bead3D[];
  colorHex: string;
  selectedBins: number[];
  highlightedMap: Map<number, string>;
  hoveredBin: number | null;
  onSelectBin: (binIndex: number) => void;
  onHoverBin: (binIndex: number | null) => void;
}

function ChromatinPolymer({
  beads,
  colorHex,
  selectedBins,
  highlightedMap,
  hoveredBin,
  onSelectBin,
  onHoverBin,
}: PolymerProps) {
  const tubeGeo = useMemo(() => {
    if (beads.length < 2) return null;
    const pts = beads.map((b) => new THREE.Vector3(b.x, b.y, b.z));
    const curve = new THREE.CatmullRomCurve3(pts);
    return new THREE.TubeGeometry(curve, beads.length * 4, 0.08, 8, false);
  }, [beads]);

  useEffect(() => {
    return () => {
      tubeGeo?.dispose();
    };
  }, [tubeGeo]);

  if (!tubeGeo) return null;

  return (
    <group>
      <mesh geometry={tubeGeo}>
        <meshStandardMaterial
          color={colorHex}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>

      {beads.map((b) => {
        const isSelected = selectedBins.includes(b.bin_index);
        const isHovered = hoveredBin === b.bin_index;
        const role = highlightedMap.get(b.bin_index);

        let beadColor = colorHex;
        let radius = 0.14;
        if (role === "enhancer") {
          beadColor = "#facc15";
          radius = 0.22;
        } else if (role === "promoter" || role === "anchor") {
          beadColor = "#38bdf8";
          radius = 0.22;
        }
        if (isSelected) {
          beadColor = "#22d3ee";
          radius = 0.28;
        } else if (isHovered) {
          beadColor = "#ffffff";
          radius = 0.24;
        }

        return (
          <mesh
            key={b.bin_index}
            position={[b.x, b.y, b.z]}
            onClick={(e) => {
              e.stopPropagation();
              onSelectBin(b.bin_index);
            }}
            onPointerOver={(e) => {
              e.stopPropagation();
              onHoverBin(b.bin_index);
            }}
            onPointerOut={() => onHoverBin(null)}
          >
            <sphereGeometry args={[radius, 14, 14]} />
            <meshStandardMaterial
              color={beadColor}
              emissive={isSelected || isHovered ? beadColor : "#000000"}
              emissiveIntensity={isSelected ? 0.6 : isHovered ? 0.35 : 0}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function CameraController() {
  const { camera } = useThree();
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const { viewport, cameraVersion } = useViewportStore();
  const animatingRef = useRef(false);

  const targetPos = useMemo(
    () => new THREE.Vector3(...viewport.camera_3d.position),
    [viewport.camera_3d.position],
  );
  const targetLookAt = useMemo(
    () => new THREE.Vector3(...viewport.camera_3d.target),
    [viewport.camera_3d.target],
  );

  // Trigger smooth flight only when cameraVersion increments (Query-to-View patch)
  useEffect(() => {
    if (cameraVersion > 0) {
      animatingRef.current = true;
    }
  }, [cameraVersion]);

  useFrame((_, delta) => {
    if (!animatingRef.current || !controlsRef.current) return;
    const step = Math.min(1, delta * 5);
    camera.position.lerp(targetPos, step);
    controlsRef.current.target.lerp(targetLookAt, step);
    controlsRef.current.update();

    if (
      camera.position.distanceTo(targetPos) < 0.05 &&
      controlsRef.current.target.distanceTo(targetLookAt) < 0.05
    ) {
      animatingRef.current = false;
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      onStart={() => {
        // Yield control immediately when user drags manually
        animatingRef.current = false;
      }}
    />
  );
}

export function Chromatin3DPanel() {
  const {
    viewport,
    structures3D,
    hoveredBin,
    setHoveredBin,
    toggleSelectedBin,
    clearSelectedBins,
  } = useViewportStore();

  const highlightedMap = useMemo(() => {
    const m = new Map<number, string>();
    for (const h of viewport.highlighted_bins) {
      m.set(h.bin_index, h.role);
    }
    return m;
  }, [viewport.highlighted_bins]);

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            3D Chromatin Conformation Viewer
          </h2>
          <p className="text-xs text-slate-400">
            Drag to rotate · Scroll to zoom · Click any bead to pin for
            Point-to-Query
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {viewport.active_cohorts.map((c) => (
            <span
              key={c.cohort_id}
              className="flex items-center gap-1 text-slate-300"
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: c.color_hex }}
              />
              {c.cohort_id}
            </span>
          ))}
          {viewport.spatial_selection_3d.selected_bins.length > 0 && (
            <button
              onClick={clearSelectedBins}
              className="rounded bg-slate-800 px-2 py-0.5 text-[11px] text-cyan-300 hover:bg-slate-700"
            >
              Clear ({viewport.spatial_selection_3d.selected_bins.length})
            </button>
          )}
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden rounded border border-slate-800 bg-slate-950">
        <Canvas camera={{ position: [12, 9, 16], fov: 45 }}>
          <ambientLight intensity={0.65} />
          <directionalLight position={[10, 15, 10]} intensity={0.9} />
          <directionalLight position={[-10, -10, -8]} intensity={0.35} />

          {viewport.active_cohorts.map((c) => {
            const beads = structures3D[c.cohort_id] ?? [];
            return (
              <ChromatinPolymer
                key={c.cohort_id}
                beads={beads}
                colorHex={c.color_hex}
                selectedBins={viewport.spatial_selection_3d.selected_bins}
                highlightedMap={highlightedMap}
                hoveredBin={hoveredBin}
                onSelectBin={toggleSelectedBin}
                onHoverBin={setHoveredBin}
              />
            );
          })}

          <CameraController />
        </Canvas>

        {/* Legend & Hover Overlay */}
        <div className="pointer-events-none absolute bottom-2 left-2 flex gap-3 rounded bg-slate-900/85 px-2.5 py-1 text-[11px] text-slate-300">
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-full bg-sky-400" />
            Promoter
          </span>
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400" />
            Enhancer
          </span>
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-full bg-cyan-400" />
            Selected ({viewport.spatial_selection_3d.selected_bins.length})
          </span>
        </div>
      </div>
    </div>
  );
}
