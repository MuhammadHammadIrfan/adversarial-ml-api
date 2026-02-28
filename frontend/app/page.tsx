"use client";

import { useState, useRef, useCallback, type DragEvent, type ChangeEvent } from "react";

// ── Config ──────────────────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// ── Types ───────────────────────────────────────────────────────────────
interface AttackResult {
  clean_prediction: number;
  adversarial_prediction: number;
  attack_success: boolean;
  adversarial_image: string; // base64 PNG
}

// ── Page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [epsilon, setEpsilon] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AttackResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── File handling ───────────────────────────────────────────────────
  const handleFile = useCallback((f: File) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const onFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      if (e.target.files?.[0]) handleFile(e.target.files[0]);
    },
    [handleFile]
  );

  const removeFile = useCallback(() => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  // ── API call ────────────────────────────────────────────────────────
  const runAttack = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("epsilon", epsilon.toString());

      const res = await fetch(`${API_URL}/attack`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `Server error (${res.status})`);
      }

      const data: AttackResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, [file, epsilon]);

  // ── Render ──────────────────────────────────────────────────────────
  return (
    <div className="app-wrapper">
      {/* Header */}
      <header className="header">
        <span className="header__badge">⚡ Adversarial ML Demo</span>
        <h1 className="header__title">FGSM Attack Playground</h1>
        <p className="header__subtitle">
          Upload a handwritten digit image and explore how the Fast Gradient
          Sign Method can fool a neural network.
        </p>
      </header>

      {/* Controls */}
      <div className="grid-controls">
        {/* Upload Card */}
        <div className="card">
          <h2 className="card__title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
            Upload Image
          </h2>

          <div
            className={`upload-zone ${dragActive ? "upload-zone--active" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <div className="upload-zone__icon">🖼️</div>
            <p className="upload-zone__text">
              Drag & drop or <strong>browse</strong>
            </p>
            <p className="upload-zone__hint">
              PNG or JPEG · Handwritten digit (0-9)
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg"
              onChange={onFileChange}
              hidden
            />
          </div>

          {file && previewUrl && (
            <div className="preview-thumb">
              <img src={previewUrl} alt="preview" />
              <div className="preview-thumb__info">
                <div className="preview-thumb__name">{file.name}</div>
                <div className="preview-thumb__size">
                  {(file.size / 1024).toFixed(1)} KB
                </div>
              </div>
              <button
                className="preview-thumb__remove"
                onClick={removeFile}
                title="Remove"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* Epsilon Card */}
        <div className="card">
          <h2 className="card__title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06c.5.5 1.21.71 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c.26.6.77 1.02 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
            Epsilon (ε)
          </h2>

          <div className="epsilon-control">
            <div className="epsilon-label">
              <span className="epsilon-label__text">Perturbation Strength</span>
              <span className="epsilon-label__value">{epsilon.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="epsilon-slider"
              min="0"
              max="0.5"
              step="0.01"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            />
            <p className="epsilon-hint">
              Higher ε → stronger perturbation → more likely to fool the model
            </p>
          </div>
        </div>
      </div>

      {/* Attack Button */}
      <button
        className="btn-attack"
        onClick={runAttack}
        disabled={!file || loading}
      >
        {loading ? (
          <>
            <span className="spinner" /> Running Attack…
          </>
        ) : (
          <>⚡ Run FGSM Attack</>
        )}
      </button>

      {/* Error */}
      {error && <div className="error-banner">❌ {error}</div>}

      {/* Results */}
      {result && (
        <div className="results-section">
          {/* Attack status */}
          <div
            className={`status-badge ${result.attack_success
                ? "status-badge--success"
                : "status-badge--fail"
              }`}
          >
            {result.attack_success
              ? "🎯 Attack Succeeded — Model was fooled!"
              : "🛡️ Attack Failed — Model held firm."}
          </div>

          {/* Predictions side-by-side */}
          <div className="grid-2">
            <div className="card pred-card">
              <div className="pred-card__label">Clean Prediction</div>
              <div className="pred-card__digit pred-card__digit--clean">
                {result.clean_prediction}
              </div>
              <div className="pred-card__subtext">Original image</div>
            </div>
            <div className="card pred-card">
              <div className="pred-card__label">Adversarial Prediction</div>
              <div className="pred-card__digit pred-card__digit--adv">
                {result.adversarial_prediction}
              </div>
              <div className="pred-card__subtext">ε = {epsilon.toFixed(2)}</div>
            </div>
          </div>

          {/* Image comparison */}
          <div className="image-compare">
            <div className="image-compare__grid">
              <div className="image-panel">
                <div className="image-panel__label">Original Image</div>
                <div className="image-panel__frame">
                  {previewUrl && (
                    <img src={previewUrl} alt="Original uploaded digit" />
                  )}
                </div>
              </div>
              <div className="image-panel">
                <div className="image-panel__label">Adversarial Image</div>
                <div className="image-panel__frame">
                  <img
                    src={`data:image/png;base64,${result.adversarial_image}`}
                    alt="Adversarial digit"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        FGSM Adversarial Attack Demo · Built with{" "}
        <a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener noreferrer">FastAPI</a>,{" "}
        <a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer">PyTorch</a> &{" "}
        <a href="https://nextjs.org/" target="_blank" rel="noopener noreferrer">Next.js</a>
      </footer>
    </div>
  );
}
