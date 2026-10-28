/**
 * Smart Classroom Attention Analytics - Frontend JavaScript Engine
 */

let selectedInputMode = "samples"; // "samples" | "upload"
let selectedSampleId = "high_view1";
let selectedUploadFile = null;
let currentAnalysisData = null;
let currentVideoView = "output"; // "output" | "input"

// Tab Switching (Pre-loaded vs Upload)
function switchInputTab(tab) {
  selectedInputMode = tab;
  document.getElementById("tab-samples-btn").classList.toggle("active", tab === "samples");
  document.getElementById("tab-upload-btn").classList.toggle("active", tab === "upload");
  document.getElementById("panel-samples").style.display = tab === "samples" ? "block" : "none";
  document.getElementById("panel-upload").style.display = tab === "upload" ? "block" : "none";
}

// Sample Video Selection
function selectSample(id, element) {
  selectedSampleId = id;
  document.querySelectorAll(".sample-card").forEach(el => el.classList.remove("selected"));
  if (element) {
    element.classList.add("selected");
  }
}

// File Input & Drag-and-Drop Handling
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    setUploadFile(file);
  }
}

function setUploadFile(file) {
  selectedUploadFile = file;
  const nameEl = document.getElementById("selected-file-name");
  nameEl.style.display = "block";
  nameEl.innerText = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
}

// Setup Dropzone
const dropzone = document.getElementById("video-dropzone");
if (dropzone) {
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setUploadFile(e.dataTransfer.files[0]);
    }
  });
}

// Video View Toggling (Output vs Input)
function switchVideoView(view) {
  currentVideoView = view;
  document.getElementById("vtab-output").classList.toggle("active", view === "output");
  document.getElementById("vtab-input").classList.toggle("active", view === "input");

  if (!currentAnalysisData) return;

  const player = document.getElementById("main-video-player");
  const targetUrl = view === "output" ? currentAnalysisData.output_video_url : currentAnalysisData.input_video_url;
  
  if (targetUrl) {
    const curTime = player.currentTime;
    player.src = targetUrl;
    player.load();
    player.currentTime = curTime;
    player.play().catch(() => {});
  }
}

// Main Execution Function
async function startAnalysis() {
  const btn = document.getElementById("btn-analyze");
  const progressCard = document.getElementById("progress-card");
  const progressBar = document.getElementById("progress-bar-fill");
  const progressStatus = document.getElementById("progress-status");
  const progressPercent = document.getElementById("progress-percent");

  // Prepare FormData
  const formData = new FormData();
  const maxDuration = parseFloat(document.getElementById("duration-slider").value) || 10.0;
  formData.append("max_duration_sec", maxDuration);
  formData.append("sample_fps", 10.0);
  formData.append("drop_threshold", 20.0);

  if (selectedInputMode === "samples") {
    formData.append("sample_id", selectedSampleId);
  } else {
    if (!selectedUploadFile) {
      alert("Please select or drop a classroom video file first.");
      return;
    }
    formData.append("video_file", selectedUploadFile);
  }

  // UI state: Running
  btn.disabled = true;
  progressCard.style.display = "block";
  progressBar.style.width = "5%";
  progressPercent.innerText = "5%";
  progressStatus.innerText = "Initializing DeepSORT & YOLO11 pipeline...";

  // Simulated progress stages for responsive feedback
  let simProg = 5;
  const progressInterval = setInterval(() => {
    if (simProg < 85) {
      simProg += Math.random() * 8 + 3;
      if (simProg > 85) simProg = 85;
      progressBar.style.width = `${Math.floor(simProg)}%`;
      progressPercent.innerText = `${Math.floor(simProg)}%`;

      if (simProg < 25) {
        progressStatus.innerText = "Detecting student keypoints with YOLO11-Pose...";
      } else if (simProg < 45) {
        progressStatus.innerText = "Tracking student trajectories with DeepSORT (73.8 FPS)...";
      } else if (simProg < 65) {
        progressStatus.innerText = "Extracting head pose, eye gaze proxies, and posture...";
      } else {
        progressStatus.innerText = "Computing observable attention scores & drop triggers...";
      }
    }
  }, 400);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    clearInterval(progressInterval);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Pipeline processing failed");
    }

    progressBar.style.width = "100%";
    progressPercent.innerText = "100%";
    progressStatus.innerText = "Rendering diagnostic video with attention HUD...";

    const data = await response.json();
    currentAnalysisData = data;

    // Render results to UI
    renderAnalysisResults(data);

    setTimeout(() => {
      progressCard.style.display = "none";
      btn.disabled = false;
    }, 1200);

  } catch (err) {
    clearInterval(progressInterval);
    progressCard.style.display = "none";
    btn.disabled = false;
    alert(`Error: ${err.message}`);
    console.error(err);
  }
}

// Render Results Dashboard
function renderAnalysisResults(data) {
  // 1. Setup Video Player
  const emptyState = document.getElementById("video-empty-state");
  const player = document.getElementById("main-video-player");

  emptyState.style.display = "none";
  player.style.display = "block";
  player.src = data.output_video_url;
  player.load();
  player.play().catch(() => {});

  // 2. Setup Downloads
  const btnDownloadVideo = document.getElementById("btn-download-video");
  btnDownloadVideo.style.display = "inline-flex";
  btnDownloadVideo.href = data.output_video_url;
  btnDownloadVideo.download = data.output_video || "classroom_attention_annotated.mp4";

  const btnDownloadJson = document.getElementById("btn-download-json");
  btnDownloadJson.style.display = "inline-flex";
  btnDownloadJson.href = `/api/analysis/${data.job_id}`;
  btnDownloadJson.download = `attention_analysis_${data.job_id}.json`;

  // 3. Populate Executive Telemetry
  document.getElementById("metrics-grid").style.display = "grid";
  document.getElementById("metric-class-avg").innerText = `${data.class_average_attention}%`;
  document.getElementById("metric-census-count").innerText = data.census_student_count || data.total_students_tracked;
  document.getElementById("metric-total-students").innerText = data.total_students_tracked;
  document.getElementById("metric-exited-count").innerText = data.exited_camera_count || 0;
  document.getElementById("metric-box-dynamics").innerText = data.last_box_event || "STABLE";
  document.getElementById("metric-attentive-pct").innerText = `${data.attentive_percentage}%`;
  document.getElementById("metric-inattentive-pct").innerText = `${data.inattentive_percentage}%`;
  document.getElementById("metric-drops-count").innerText = data.drop_events_count;

  // 4. Populate Student Trajectory Cards
  const studentsSection = document.getElementById("students-section");
  const studentsGrid = document.getElementById("students-grid");
  studentsGrid.innerHTML = "";

  if (data.student_summaries && data.student_summaries.length > 0) {
    studentsSection.style.display = "block";
    data.student_summaries.forEach(s => {
      const card = document.createElement("div");
      card.className = "student-card";

      let badgeClass = "badge-attentive";
      if (s.predominant_category === "Inattentive") badgeClass = "badge-inattentive";
      else if (s.predominant_category === "Uncertain") badgeClass = "badge-uncertain";

      const barColor = s.mean_score >= 60 ? "var(--attentive)" : (s.mean_score >= 40 ? "var(--uncertain)" : "var(--inattentive)");

      card.innerHTML = `
        <div class="student-card-head">
          <span class="student-id">Student #${s.track_id}</span>
          <span class="${badgeClass}">${s.predominant_category}</span>
        </div>
        <div>
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
            <span style="color: var(--text-muted);">Attention Index:</span>
            <strong style="color: ${barColor}; font-size: 13px;">${s.mean_score}%</strong>
          </div>
          <div class="student-score-bar-bg">
            <div class="student-score-bar-fill" style="width: ${Math.min(100, Math.max(5, s.mean_score))}%; background: ${barColor};"></div>
          </div>
        </div>
        <div class="student-details">
          <span>Behavior: <strong style="color: #FFFFFF;">${s.predominant_behavior}</strong></span>
          <span>Observations: ${s.observations} frames</span>
        </div>
      `;
      studentsGrid.appendChild(card);
    });
  } else {
    studentsSection.style.display = "none";
  }

  // 5. Populate Attention Drops & Recommendations
  const dropsSection = document.getElementById("drops-section");
  const dropFeed = document.getElementById("drop-feed");
  dropFeed.innerHTML = "";

  if (data.drop_events && data.drop_events.length > 0) {
    dropsSection.style.display = "block";
    data.drop_events.forEach(drop => {
      const item = document.createElement("div");
      item.className = "drop-item";
      item.innerHTML = `
        <div class="drop-item-head">
          <span class="drop-item-title">⚠️ Student #${drop.track_id} Attention Drop (-${drop.drop_magnitude}%)</span>
          <span class="drop-item-time">Timestamp: ${drop.timestamp_sec}s (Frame ${drop.frame_idx})</span>
        </div>
        <p class="drop-item-desc"><strong>Driver:</strong> ${drop.recommendation.explanation}</p>
        <div class="drop-item-rec">
          <strong>Pedagogical Action:</strong> ${drop.recommendation.teaching_recommendation}
          <div style="margin-top: 4px; font-size: 11px; color: var(--accent-cyan);">
            <em>Intervention Suggestion:</em> ${drop.recommendation.intervention_suggestion}
          </div>
        </div>
      `;
      dropFeed.appendChild(item);
    });
  } else {
    dropsSection.style.display = "block";
    dropFeed.innerHTML = `
      <div style="padding: 16px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: var(--radius-sm); color: var(--attentive); font-size: 13px;">
        ✅ <strong>Optimal Classroom Attention Sustained:</strong> No severe attention drops (&gt;20% over 2.0s) were detected across visible students in this sequence.
      </div>
    `;
  }

  // Scroll smoothly to results
  player.scrollIntoView({ behavior: "smooth", block: "center" });
}
