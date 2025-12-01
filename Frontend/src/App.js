import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");

  const API = "http://127.0.0.1:5000";

  const steps = [
    "Calculating MFCC...",
    "Analyzing Chroma...",
    "Computing Centroid...",
    "Measuring Rolloff...",
    "Comparing fingerprints...",
    "Finalizing result..."
  ];

  const order = [
    ["mfcc_mean", "MFCC Mean"],
    ["mfcc_std", "MFCC Std"],
    ["chroma_mean", "Chroma Mean"],
    ["chroma_std", "Chroma Std"],
    ["centroid_mean", "Centroid Mean"],
    ["centroid_std", "Centroid Std"],
    ["rolloff_mean", "Rolloff Mean"],
    ["rolloff_std", "Rolloff Std"]
  ];

  const analyze = async () => {
    if (!file) return alert("Upload a file first.");

    setLoading(true);
    setResult(null);

    let i = 0;
    const loop = setInterval(() => {
      setStatus(steps[i]);
      i = (i + 1) % steps.length;
    }, 800);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await axios.post(`${API}/analyze`, form);
      clearInterval(loop);

      setTimeout(() => {
        setStatus("");
        setResult(res.data);
      }, 1000);
    } catch {
      clearInterval(loop);
      setStatus("");
      alert("Error analyzing file");
    }

    setLoading(false);
  };

  return (
    <>
      {/* Background Video */}
      <video autoPlay loop muted playsInline className="bg-video">
        <source src="/background3.mp4" type="video/mp4" />
      </video>

      {/* Content Overlay */}
      <div className="overlay">
        <div className="page">
          <h1 className="title">Audio Instrument Analyzer</h1>

          <div className="upload-section">
            <input
              type="file"
              id="fileInput"
              style={{ display: "none" }}
              accept="audio/*"
              onChange={e => setFile(e.target.files[0])}
            />
            <label htmlFor="fileInput" className="upload-box">
              {file ? file.name : "Click to upload audio"}
            </label>

            <button className="analyze-btn" disabled={loading} onClick={analyze}>
              {loading ? "Analyzing..." : "Analyze File"}
            </button>
          </div>

          {status && <p className="status">{status}</p>}

          {result && (
            <div className="card">
              <h2>
                Detected Instrument:{" "}
                <span className="highlight">{result.predicted_instrument}</span>
              </h2>

              <p className="confidence">
                Confidence: <span>{result.accuracy_percent.toFixed(2)}%</span>
              </p>

              <h3>Extracted Audio Features</h3>
              <table className="score-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {order.map(([k, name]) => (
                    <tr key={k}>
                      <td>{name}</td>
                      <td>{result.features[k].toFixed(5)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h3>All Similarity Scores</h3>
              <table className="score-table">
                <thead>
                  <tr>
                    <th>Instrument</th>
                    <th>Score</th>
                  </tr>
                </thead>

                <tbody>
                  {Object.entries(result.similarity_scores)
                    .sort((a, b) => b[1] - a[1])
                    .map(([inst, val]) => (
                      <tr key={inst}>
                        <td>{inst}</td>
                        <td>{(val * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
