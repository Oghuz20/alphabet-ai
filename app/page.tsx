"use client";

import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [result, setResult] = useState("");

  useEffect(() => {
    const loadModel = async () => {
      const m = await tf.loadGraphModel("/model/model.json");
      setModel(m);
    };

    loadModel();
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setResult("");
  };

  const predict = async () => {
    if (!model) return;

    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;

    const sctx = small.getContext("2d")!;
    sctx.drawImage(canvas, 0, 0, 28, 28);

    const img = sctx.getImageData(0, 0, 28, 28);
    const data = img.data;

    const arr = new Float32Array(28 * 28);

    for (let i = 0; i < 28 * 28; i++) {
      const gray =
        (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3 / 255;

      arr[i] = 1 - gray;
    }

    const input = tf.tensor4d(arr, [1, 28, 28, 1]);

    const prediction = model.predict(input) as tf.Tensor;

    const probs = await prediction.data();

    let maxIndex = 0;

    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIndex]) maxIndex = i;
    }

    const letter = letters[maxIndex];
    const confidence = (probs[maxIndex] * 100).toFixed(1);

    setResult(`${letter} (${confidence}%)`);
  };

  const startDrawing = (e: any) => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    canvasRef.current!.onmousemove = draw;
  };

  const draw = (e: any) => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    canvasRef.current!.onmousemove = null;
  };

  useEffect(() => {
    const ctx = canvasRef.current!.getContext("2d")!;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 280, 280);

    ctx.strokeStyle = "black";
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
  }, []);

  return (
    <main style={{ textAlign: "center", marginTop: 40 }}>
      <h1>Handwritten Alphabet Predictor</h1>

      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: "2px solid black", background: "white" }}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />

      <div style={{ marginTop: 20 }}>
        <button onClick={predict}>Predict</button>
        <button onClick={clearCanvas} style={{ marginLeft: 10 }}>
          Clear
        </button>
      </div>

      <h2 style={{ marginTop: 20 }}>{result}</h2>
    </main>
  );
}