"use client";

import * as tf from "@tensorflow/tfjs";
import React, { useEffect, useMemo, useRef, useState } from "react";

type Pred = { ch: string; p: number };

const CANVAS_SIZE = 280;
const MODEL_INPUT = 28;
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function argmax(arr: Float32Array) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewRef = useRef<HTMLCanvasElement | null>(null); // shows what model sees (debug)
  const [isDrawing, setIsDrawing] = useState(false);

  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [status, setStatus] = useState("Loading model…");
  const [top, setTop] = useState<Pred[]>([]);
  const [main, setMain] = useState<Pred | null>(null);

  const hiddenCanvas = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = CANVAS_SIZE;
    c.height = CANVAS_SIZE;
    return c;
  }, []);

  // init drawing canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // White background (important)
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Black ink
    ctx.strokeStyle = "black";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  // load model
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await tf.ready();
        const m = await tf.loadGraphModel("/model/model.json");
        if (!mounted) return;

        // warmup
        tf.tidy(() => {
          const x = tf.zeros([1, 28, 28, 1]);
          m.predict(x) as tf.Tensor;
        });

        setModel(m);
        setStatus("Model loaded ✅ Draw a letter and click Predict.");
      } catch (e) {
        console.error(e);
        setStatus("Failed to load model. Check /public/model/* files.");
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  function getPos(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    return { x, y };
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    canvas.setPointerCapture(e.pointerId);
    setIsDrawing(true);

    const ctx = canvas.getContext("2d")!;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawing) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function onPointerUp() {
    setIsDrawing(false);
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.closePath();
  }

  function clearCanvas() {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setTop([]);
    setMain(null);
    setStatus(model ? "Cleared. Draw a letter and Predict." : status);
  }

  // --- 핵심: center + resize + invert + EMNIST orientation fix ---
  function build28x28FromCanvas(): ImageData {
    const src = canvasRef.current!;
    const sctx = hiddenCanvas.getContext("2d")!;
    sctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    sctx.drawImage(src, 0, 0);

    const img = sctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const d = img.data;

    // Find bounding box of "ink" (anything not near-white)
    let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1;
    for (let y = 0; y < CANVAS_SIZE; y++) {
      for (let x = 0; x < CANVAS_SIZE; x++) {
        const i = (y * CANVAS_SIZE + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        // ink if not white-ish
        if (r < 245 || g < 245 || b < 245) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    // If nothing drawn, return blank 28x28
    const outCanvas = document.createElement("canvas");
    outCanvas.width = MODEL_INPUT;
    outCanvas.height = MODEL_INPUT;
    const octx = outCanvas.getContext("2d")!;
    octx.fillStyle = "white";
    octx.fillRect(0, 0, MODEL_INPUT, MODEL_INPUT);

    if (maxX < 0 || maxY < 0) {
      return octx.getImageData(0, 0, MODEL_INPUT, MODEL_INPUT);
    }

    // Add padding around bbox
    const pad = 12;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(CANVAS_SIZE - 1, maxX + pad);
    maxY = Math.min(CANVAS_SIZE - 1, maxY + pad);

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;

    // Draw cropped region into 28x28 with preserved aspect ratio
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = w;
    cropCanvas.height = h;
    const cctx = cropCanvas.getContext("2d")!;
    cctx.putImageData(img, -minX, -minY);

    const scale = Math.min(MODEL_INPUT / w, MODEL_INPUT / h);
    const nw = Math.max(1, Math.round(w * scale));
    const nh = Math.max(1, Math.round(h * scale));
    const dx = Math.floor((MODEL_INPUT - nw) / 2);
    const dy = Math.floor((MODEL_INPUT - nh) / 2);

    octx.drawImage(cropCanvas, 0, 0, w, h, dx, dy, nw, nh);

    // Preview what model sees (optional but super useful)
    if (previewRef.current) {
      const pctx = previewRef.current.getContext("2d")!;
      pctx.imageSmoothingEnabled = false;
      pctx.clearRect(0, 0, 280, 280);
      pctx.drawImage(outCanvas, 0, 0, 280, 280);
    }

    return octx.getImageData(0, 0, MODEL_INPUT, MODEL_INPUT);
  }

  function imageDataToTensor28(img28: ImageData): tf.Tensor4D {
    const data = img28.data;
    const arr = new Float32Array(MODEL_INPUT * MODEL_INPUT);

    // grayscale + normalize
    // We invert so black ink becomes high value (often matches MNIST/EMNIST-style)
    for (let i = 0; i < MODEL_INPUT * MODEL_INPUT; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const gray = (r + g + b) / (3 * 255); // 0..1 (white=1)
      arr[i] = 1.0 - gray; // ink high
    }

    // [1,28,28,1]
    let x = tf.tensor4d(arr, [1, MODEL_INPUT, MODEL_INPUT, 1]);

    // EMNIST orientation fix:
    // Many EMNIST variants appear rotated/transposed; this aligns user drawings with what the model learned.
    // If predictions are still rotated, we can tweak (swap/flip).
    x = tf.transpose(x, [0, 2, 1, 3]);      // transpose H<->W
    x = tf.reverse(x, [2]);                 // flip width axis

    return x;
  }

  async function predict() {
    if (!model) return;
    setStatus("Predicting…");

    // 1) Build input + run model inside tidy (SYNC)
    const outTensor = tf.tidy(() => {
      const img28 = build28x28FromCanvas();
      const x = imageDataToTensor28(img28);      // [1,28,28,1]
      const y = model.predict(x) as tf.Tensor;   // [1,26]
      return y.squeeze();                        // [26]  <-- returned tensor (kept alive)
    });

    // 2) Read probabilities OUTSIDE tidy (ASYNC OK)
    const probs = (await outTensor.data()) as Float32Array;
    outTensor.dispose();

    // 3) Compute top-3
    const indexed = Array.from(probs).map((p, i) => ({ i, p }));
    indexed.sort((a, b) => b.p - a.p);

    const top3 = indexed.slice(0, 3).map(({ i, p }) => ({
      ch: LETTERS[i],
      p,
    }));

    setMain(top3[0]);
    setTop(top3);
    setStatus("Done ✅");
  }
  return (
    <main style={styles.page}>
      <div style={styles.shell}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.h1}>Handwritten Alphabet Predictor</h1>
            <p style={styles.sub}>{status}</p>
          </div>
          <div style={styles.badge}>A–Z (case-insensitive)</div>
        </header>

        <section style={styles.grid}>
          <div style={styles.card}>
            <div style={styles.canvasWrap}>
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                style={styles.canvas}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
              />
            </div>

            <div style={styles.actions}>
              <button style={styles.btn} onClick={clearCanvas}>Clear</button>
              <button style={styles.btnPrimary} onClick={predict} disabled={!model}>
                Predict
              </button>
            </div>

            <p style={styles.tip}>
              Tip: write big and centered. One letter at a time.
            </p>
          </div>

          <div style={styles.card}>
            <h2 style={styles.h2}>Result</h2>

            {main ? (
              <>
                <div style={styles.bigLetter}>{main.ch}</div>
                <div style={styles.conf}>Confidence: {(main.p * 100).toFixed(1)}%</div>

                <div style={styles.divider} />

                <div style={styles.h3}>Top 3</div>
                <div style={{ display: "grid", gap: 10 }}>
                  {top.map((p) => (
                    <div key={p.ch} style={styles.row}>
                      <span style={styles.rowLeft}>{p.ch}</span>
                      <span style={styles.rowRight}>{(p.p * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ opacity: 0.75 }}>Draw a letter and press Predict.</div>
            )}

            <div style={styles.divider} />

            <div style={styles.h3}>Model input (debug)</div>
            <p style={{ marginTop: 6, opacity: 0.75, fontSize: 13 }}>
              This shows the 28×28 image after centering/resizing. If this looks rotated, we’ll tweak the transform.
            </p>
            <canvas
              ref={previewRef}
              width={280}
              height={280}
              style={{ ...styles.canvas, width: 220, height: 220 }}
            />
          </div>
        </section>
      </div>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #0b1220 0%, #070a12 100%)",
    color: "white",
    padding: 24,
    display: "grid",
    placeItems: "center",
  },
  shell: { width: "min(1040px, 100%)" },
  header: {
    display: "flex",
    justifyContent: "space-between",
    gap: 16,
    alignItems: "center",
    marginBottom: 18,
  },
  h1: { margin: 0, fontSize: 30, letterSpacing: -0.3 },
  sub: { margin: "6px 0 0", opacity: 0.8 },
  badge: {
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(255,255,255,0.06)",
    padding: "8px 12px",
    borderRadius: 999,
    fontWeight: 700,
    fontSize: 13,
    whiteSpace: "nowrap",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 16,
  },
  card: {
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 18,
    padding: 16,
    boxShadow: "0 12px 40px rgba(0,0,0,0.35)",
  },
  canvasWrap: { display: "grid", placeItems: "center" },
  canvas: {
    borderRadius: 14,
    border: "1px solid rgba(255,255,255,0.25)",
    background: "white",
    touchAction: "none",
  },
  actions: { display: "flex", gap: 10, marginTop: 12 },
  btn: {
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.25)",
    background: "rgba(255,255,255,0.06)",
    color: "white",
    fontWeight: 800,
    cursor: "pointer",
  },
  btnPrimary: {
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.25)",
    background: "rgba(99, 102, 241, 0.85)",
    color: "white",
    fontWeight: 900,
    cursor: "pointer",
    flex: 1,
  },
  tip: { marginTop: 10, opacity: 0.75, fontSize: 13 },
  h2: { margin: 0, fontSize: 18 },
  bigLetter: { fontSize: 70, fontWeight: 950, lineHeight: 1, marginTop: 10 },
  conf: { opacity: 0.8, marginTop: 6 },
  divider: { height: 1, background: "rgba(255,255,255,0.12)", margin: "14px 0" },
  h3: { fontWeight: 900, opacity: 0.95 },
  row: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  rowLeft: { fontSize: 18, fontWeight: 900 },
  rowRight: { opacity: 0.8, fontWeight: 800 },
};