const $ = id => document.getElementById(id);
const charts = {};

function makeChart(id, type, labels, datasets, options = {}) {
  const c = $(id);
  if (!c) return;
  charts[id] = new Chart(c, {
    type,
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: datasets.length > 1 } },
      scales: {
        x: { grid: { color: "#202b36" } },
        y: { grid: { color: "#202b36" }, beginAtZero: true }
      },
      ...options
    }
  });
}

// 1. Diagnostics Charts
makeChart("classChart", "bar", ["No pain", "Low pain", "Moderate pain", "High pain"], [
  { label: "True frames", data: [], backgroundColor: "#65d9c8", borderRadius: 5 }
]);

makeChart("metricChart", "bar", ["Accuracy", "Balanced Accuracy", "Macro F1"], [
  { label: "Validation", data: [78.43, 56.59, 59.44], backgroundColor: "#65d9c8", borderRadius: 5 }
]);

makeChart("ordinalChart", "bar", ["Exact", "1-level", "2-level", "3-level"], [
  { label: "Validation", data: [], backgroundColor: "#65d9c8", borderRadius: 5 }
]);

makeChart("pspiChart", "line", [], [
  { label: "Accuracy", data: [], borderColor: "#65d9c8", backgroundColor: "#65d9c8", tension: 0.35 }
], {
  scales: {
    x: { grid: { color: "#202b36" } },
    y: { grid: { color: "#202b36" }, beginAtZero: true, max: 100 }
  }
});

makeChart("confidenceChart", "bar", ["Correct", "Incorrect"], [
  { label: "Mean confidence", data: [], backgroundColor: "#65d9c8", borderRadius: 5 }
], {
  scales: {
    x: { grid: { color: "#202b36" } },
    y: { grid: { color: "#202b36" }, beginAtZero: true, max: 1 }
  }
});

makeChart("subjectChart", "bar", [], [
  { label: "Accuracy", data: [], backgroundColor: "#65d9c8", borderRadius: 5 }
], {
  scales: {
    x: { grid: { color: "#202b36" } },
    y: { grid: { color: "#202b36" }, beginAtZero: true, max: 100 }
  }
});

makeChart("historyChart", "line", [], [
  { label: "Train loss", data: [], borderColor: "#65d9c8", backgroundColor: "#65d9c8", tension: 0.25 },
  { label: "Validation loss", data: [], borderColor: "#8290a2", backgroundColor: "#8290a2", tension: 0.25 }
]);

// 2. Dual-Axis Trajectory Chart: PSPI Pain Score (Left Y) + Confidence % (Right Y1)
makeChart("trajectoryChart", "line", [], [
  {
    label: "Facial Pain (PSPI 0–16)",
    data: [],
    borderColor: "#ff6565",
    backgroundColor: "rgba(255, 101, 101, 0.15)",
    yAxisID: "y",
    fill: true,
    tension: 0.3,
    borderWidth: 2
  },
  {
    label: "Model Confidence (%)",
    data: [],
    borderColor: "#65d9c8",
    backgroundColor: "rgba(101, 217, 200, 0.08)",
    yAxisID: "y1",
    fill: false,
    tension: 0.3,
    borderWidth: 2
  }
], {
  plugins: {
    legend: {
      display: true,
      labels: { color: "#8d99a9", font: { size: 11 } }
    }
  },
  scales: {
    x: {
      grid: { color: "#202b36" },
      title: { display: true, text: "Timeline (Seconds)", color: "#728094", font: { size: 10 } }
    },
    y: {
      type: "linear",
      display: true,
      position: "left",
      beginAtZero: true,
      min: 0,
      max: 16,
      grid: { color: "#202b36" },
      title: { display: true, text: "Continuous PSPI (0–16)", color: "#ff6565", font: { size: 11, weight: "bold" } },
      ticks: { color: "#ff6565" }
    },
    y1: {
      type: "linear",
      display: true,
      position: "right",
      beginAtZero: true,
      min: 0,
      max: 100,
      grid: { drawOnChartArea: false },
      title: { display: true, text: "Confidence (%)", color: "#65d9c8", font: { size: 11, weight: "bold" } },
      ticks: { color: "#65d9c8" }
    }
  }
});

function matrix(m) {
  const root = $("matrix");
  if (!root) return;
  root.innerHTML = "";
  if (!m || !m.length) {
    root.innerHTML = "<div class='empty'>Confusion matrix data unavailable.</div>";
    return;
  }
  const labels = ["N", "L", "M", "H"].slice(0, m.length), max = Math.max(...m.flat(), 1);
  root.style.display = "grid";
  root.style.gridTemplateColumns = `35px repeat(${m.length},1fr)`;
  root.style.gap = "4px";
  ["", ...labels].forEach(x => {
    const d = document.createElement("div");
    d.textContent = x;
    d.style.padding = "10px";
    d.style.color = "#718094";
    root.appendChild(d);
  });
  m.forEach((row, i) => {
    const h = document.createElement("div");
    h.textContent = labels[i];
    h.style.padding = "10px";
    h.style.color = "#718094";
    root.appendChild(h);
    row.forEach(v => {
      const d = document.createElement("div");
      d.textContent = v;
      d.style.padding = "12px";
      d.style.textAlign = "center";
      d.style.background = `rgba(101,217,200,${Math.min(0.7, 0.06 + v / max)})`;
      root.appendChild(d);
    });
  });
}

function setText(id, v) {
  const e = $(id);
  if (e) e.textContent = v;
}

function pct(v) {
  return (Number(v) * 100).toFixed(2) + "%";
}

// 3. Mini HTML5 Oscilloscope for rPPG Pulse Waveform
function drawPulseWaveform(wave) {
  const c = $("pulseCanvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.width = c.clientWidth || 340;
  const h = c.height = c.clientHeight || 95;
  ctx.clearRect(0, 0, w, h);

  // Background center grid line
  ctx.strokeStyle = "#1b2632";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  if (!wave || !wave.length) {
    ctx.fillStyle = "#556577";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Pulse waveform synchronizing...", w / 2, h / 2 + 3);
    return;
  }

  // Draw pulse line
  ctx.strokeStyle = "#65d9c8";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  const step = w / (wave.length - 1);
  const midY = h / 2;
  const scale = h * 0.40;

  wave.forEach((v, i) => {
    const x = i * step;
    const y = midY - v * scale;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Gradient fill under pulse wave
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(101, 217, 200, 0.22)");
  grad.addColorStop(1, "rgba(101, 217, 200, 0.0)");
  ctx.fillStyle = grad;
  ctx.fill();
}

async function loadResults() {
  try {
    const r = await fetch("/api/results?ts=" + Date.now(), { cache: "no-store" });
    const d = await r.json();
    console.log("PAIN-AI analytics loaded:", d);

    const metrics = d.metrics || d.validation || {};
    if (metrics.precision != null) setText("precision", pct(metrics.precision ?? metrics.macro_precision));
    if (metrics.recall != null) setText("recall", pct(metrics.recall ?? metrics.macro_recall));
    if (metrics.mcc != null) setText("mcc", Number(metrics.mcc).toFixed(4));

    const cd = d.class_distribution;
    if (cd) {
      charts.classChart.data.labels = cd.labels || charts.classChart.data.labels;
      charts.classChart.data.datasets[0].data = cd.true || [];
      charts.classChart.update();
    }

    const cm = d.confusion_matrix;
    matrix(cm?.values || cm);

    if (d.ordinal_error) {
      charts.ordinalChart.data.labels = d.ordinal_error.labels || [];
      charts.ordinalChart.data.datasets[0].data = (d.ordinal_error.values || []).map(v => Number(v) * 100);
      charts.ordinalChart.update();
    }

    if (Array.isArray(d.pspi_analysis)) {
      charts.pspiChart.data.labels = d.pspi_analysis.map(x => String(x.pspi));
      charts.pspiChart.data.datasets[0].data = d.pspi_analysis.map(x => Number(x.accuracy) * 100);
      charts.pspiChart.update();
    }

    if (d.confidence_summary) {
      charts.confidenceChart.data.labels = d.confidence_summary.labels || [];
      charts.confidenceChart.data.datasets[0].data = d.confidence_summary.mean || [];
      charts.confidenceChart.update();
    }

    if (Array.isArray(d.subject_analysis)) {
      charts.subjectChart.data.labels = d.subject_analysis.map(x => x.subject || x.Subject);
      charts.subjectChart.data.datasets[0].data = d.subject_analysis.map(x => Number(x.accuracy ?? x.Accuracy) * 100);
      charts.subjectChart.update();
    }

    if (Array.isArray(d.training_history)) {
      charts.historyChart.data.labels = d.training_history.map(x => String(x.epoch));
      charts.historyChart.data.datasets[0].data = d.training_history.map(x => x.train_loss);
      charts.historyChart.data.datasets[1].data = d.training_history.map(x => x.val_loss);
      charts.historyChart.update();
    }
  } catch (e) {
    console.error("Analytics load failed:", e);
  }
}
loadResults();

// File upload binding
$("upload").onclick = () => $("file").click();
$("file").onchange = () => {
  const f = $("file").files[0];
  if (!f) return;
  $("preview").src = URL.createObjectURL(f);
  predict(f);
};

async function predict(file) {
  $("label").textContent = "Multimodal Analysing…";
  $("conf").textContent = "";
  setText("clinicalBadge", "PROCESSING...");
  setText("clinicalInterpretation", "Extracting MediaPipe FaceMesh & POS rPPG pulse traces...");

  const fd = new FormData();
  fd.append("file", file);

  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "Inference failed");
    renderPrediction(d);
  } catch (e) {
    $("label").textContent = "Inference unavailable";
    $("pspi").textContent = e.message;
    setText("clinicalBadge", "ERROR");
    setText("clinicalInterpretation", e.message);
  }
}

function renderPrediction(d) {
  // 1. Primary Discrete Classification
  $("label").textContent = d.prediction.label;
  $("conf").textContent = d.prediction.confidence + "%";
  $("bar").style.width = d.prediction.confidence + "%";
  $("frames").textContent = d.frames_analyzed;
  $("duration").textContent = d.duration_seconds + " s";
  $("fps").textContent = d.sampled_fps;
  $("face").textContent = (d.face_detection_rate != null ? d.face_detection_rate + "%" : "100%");
  $("pspi").textContent = d.pspi_message;

  const p = d.prediction.probabilities, names = ["No Pain", "Low Pain", "Moderate", "High Pain"];
  $("probs").innerHTML = names.map((n, i) =>
    `<div class="prob"><span>${n}</span><i><b style="width:${p[i]}%"></b></i><em>${p[i]}%</em></div>`
  ).join("");

  // 2. Clinical Pain Status Card (Tri-Scale)
  const pspi = d.pspi_score != null ? d.pspi_score : 0.0;
  setText("scorePspi", pspi.toFixed(2));
  setText("scoreCpot", (d.cpot_score != null ? d.cpot_score : 0) + " / 8");
  setText("scoreNrs", (d.nrs_score != null ? d.nrs_score : 0) + " / 10");

  const badge = $("clinicalBadge");
  if (badge) {
    badge.className = "status-pill";
    if (pspi < 1.2) {
      badge.textContent = "NO PAIN (RELAXED)";
    } else if (pspi < 3.5) {
      badge.textContent = "MILD PAIN";
      badge.classList.add("mild");
    } else if (pspi < 6.5) {
      badge.textContent = "MODERATE PAIN";
      badge.classList.add("moderate");
    } else {
      badge.textContent = "SEVERE PAIN (CRITICAL)";
      badge.classList.add("severe");
    }
  }

  const interp = $("clinicalInterpretation");
  if (interp) {
    if (pspi < 1.2) {
      interp.textContent = "Patient displays relaxed baseline facial tone and stable autonomic vitals. No analgesic intervention indicated.";
    } else if (pspi < 3.5) {
      interp.textContent = "Mild brow/orbital contraction observed with minor autonomic arousal. Continue non-pharmacological monitoring.";
    } else if (pspi < 6.5) {
      interp.textContent = "Significant facial Action Unit activation (AU4/AU6/AU9) combined with elevated autonomic stress. CPOT intervention recommended.";
    } else {
      interp.textContent = "Critical acute pain spike: intense grimacing with elevated heart rate elevation. Urgent clinical review advised.";
    }
  }

  // 3. Contactless Vitals Card
  const v = d.vitals || {};
  setText("vitalHr", v.heart_rate_bpm ? v.heart_rate_bpm.toFixed(1) + " BPM" : "Estimating...");
  setText("vitalResp", v.respiration_rate ? v.respiration_rate.toFixed(1) + " BrPM" : "Stabilizing...");
  const sqiVal = Math.round((v.sqi || 0) * 100);
  setText("vitalSqi", sqiVal + "%");
  setText("sqiBadge", "SQI: " + sqiVal + "%");
  setText("pulseRateLabel", v.heart_rate_bpm ? `${v.heart_rate_bpm.toFixed(1)} BPM` : "POS rPPG");

  drawPulseWaveform(v.pulse_waveform || []);

  // 4. Dual-Axis Trajectory Chart
  const tr = d.trajectory || [];
  charts.trajectoryChart.data.labels = tr.map(x => x.time + "s");
  charts.trajectoryChart.data.datasets[0].data = tr.map(x => (x.pspi !== undefined ? x.pspi : 0));
  charts.trajectoryChart.data.datasets[1].data = tr.map(x => x.confidence);
  charts.trajectoryChart.update();
}

// 4. Webcam Recording Handler
let stream = null, mediaRecorder = null, recordedChunks = [], recordTimer = null;

$("camera").onclick = async () => {
  try {
    if (stream) {
      stopWebcam();
      return;
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    const v = document.createElement("video");
    v.autoplay = true;
    v.muted = true;
    v.playsInline = true;
    v.style.width = "100%";
    v.style.height = "210px";
    v.style.objectFit = "contain";
    v.srcObject = stream;
    $("webcam").innerHTML = "";
    $("webcam").appendChild(v);
    $("camera").textContent = "Stop Webcam";
    $("label").textContent = "Recording 5-second video clip for multimodal analysis...";

    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" :
                 (MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "");
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) recordedChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      clearTimeout(recordTimer);
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
      if (blob.size < 1000) {
        $("label").textContent = "No usable recording captured";
        return;
      }
      $("label").textContent = "Analysing multimodal webcam recording…";
      try {
        await predict(new File([blob], "webcam_assessment.webm", { type: blob.type || "video/webm" }));
      } finally {
        recordedChunks = [];
      }
    };
    mediaRecorder.start(250);
    recordTimer = setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
    }, 5000);
  } catch (e) {
    $("label").textContent = "Camera unavailable";
    if ($("pspi")) $("pspi").textContent = e.name === "NotAllowedError" ? "Camera permission was denied." : e.message;
    console.error(e);
  }
};

function stopTracksOnly() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

function stopWebcam() {
  clearTimeout(recordTimer);
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  } else {
    stopTracksOnly();
    $("camera").textContent = "Start Webcam";
    $("label").textContent = "Webcam stopped";
  }
}
