const API_BASE = "http://localhost:8000";

const form = document.getElementById("riskForm");
const result = document.getElementById("result");
const submitBtn = document.getElementById("submitBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  result.classList.add("hidden");
  submitBtn.disabled = true;
  submitBtn.textContent = "Calculating...";

  const payload = {
    age_years: Number(document.getElementById("age").value),
    gender: Number(document.getElementById("gender").value),
    height: Number(document.getElementById("height").value),
    weight: Number(document.getElementById("weight").value),
    smoke: document.getElementById("smoke").checked ? 1 : 0,
    alco: document.getElementById("alco").checked ? 1 : 0,
    active: document.getElementById("active").checked ? 1 : 0,
    ap_hi: Number(document.getElementById("ap_hi").value) || null,
    ap_lo: Number(document.getElementById("ap_lo").value) || null,
    cholesterol: Number(document.getElementById("cholesterol").value) || null,
    gluc: Number(document.getElementById("gluc").value) || null,
  };

  try {
    const [predictRes, recsRes, explRes] = await Promise.all([
      fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch(`${API_BASE}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetch(`${API_BASE}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ]);

    if (!predictRes.ok) throw new Error("API error");

    const data = predictRes.json();
    const recs = recsRes.ok ? recsRes.json() : { recommendations: [] };
    const expls = explRes.ok ? explRes.json() : { explanations: {} };

    const [predictData, recsData, explsData] = await Promise.all([data, recs, expls]);
    renderResult(predictData, recsData, explsData);
  } catch (err) {
    if (err.message === "API error") {
      showError("API not reachable. Make sure the server is running on port 8000.");
    } else {
      showError("An error occurred. Please try again.");
    }
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Assess Risk";
  }
});

function renderResult(predictData, recsData, explsData) {
  const pct = predictData.risk_percent;
  const label = predictData.risk_label;
  const icon = { Low: "🟢", Moderate: "🟡", High: "🔴" }[label];

  document.getElementById("riskIcon").textContent = icon;
  document.getElementById("riskLabel").textContent = `${label} Risk`;
  document.getElementById("riskPercent").textContent = `${pct.toFixed(1)}%`;

  const progressFill = document.getElementById("progressFill");
  progressFill.style.width = `${pct}%`;
  progressFill.style.background = pct < 30 ? "#4caf50" : pct < 60 ? "#ff9800" : "#f44336";

  // estimated values
  const estimatedSection = document.getElementById("estimatedSection");
  const estimatedList = document.getElementById("estimatedList");
  estimatedList.innerHTML = "";
  if (predictData.estimated && Object.keys(predictData.estimated).length > 0) {
    estimatedSection.classList.remove("hidden");
    for (const [key, value] of Object.entries(predictData.estimated)) {
      estimatedList.innerHTML += `<div class="explanation-item"><span class="explanation-model">${key}</span><span class="explanation-impact">${value}</span></div>`;
    }
  } else {
    estimatedSection.classList.add("hidden");
  }

  // model breakdown
  const modelBreakdown = document.getElementById("modelBreakdown");
  modelBreakdown.innerHTML = "";
  for (const [name, prob] of Object.entries(predictData.models)) {
    modelBreakdown.innerHTML += `<div class="model-item"><span class="model-name">${name}</span><span class="model-prob">${(prob * 100).toFixed(1)}%</span></div>`;
  }

  // recommendations
  const recsList = document.getElementById("recommendationsList");
  recsList.innerHTML = "";
  if (recsData.recommendations && recsData.recommendations.length > 0) {
    recsData.recommendations.forEach((rec) => {
      recsList.innerHTML += `
        <div class="recommendation-item">
          <div>
            <span class="recommendation-action">${rec.action}</span>
            <div class="recommendation-detail">Risk drops by ${(rec.reduction * 100).toFixed(1)}% → new risk: ${(rec.new_prob * 100).toFixed(1)}%</div>
          </div>
        </div>`;
    });
  } else {
    recsList.innerHTML = "<p style='font-size: 0.875rem; color: #888;'>No actionable recommendations based on current inputs.</p>";
  }

  // explanations
  const explList = document.getElementById("explanationsList");
  explList.innerHTML = "";
  if (explsData.explanations && Object.keys(explsData.explanations).length > 0) {
    for (const [modelName, pairs] of Object.entries(explsData.explanations)) {
      explList.innerHTML += `<div class="explanation-item"><span class="explanation-model">${modelName}</span></div>`;
      pairs.forEach(([feat, impact]) => {
        const sign = impact >= 0 ? "+" : "";
        const effect = impact >= 0 ? "↑ increased risk" : "↓ decreased risk";
        explList.innerHTML += `<div class="explanation-item" style="padding-left: 1rem;"><span class="explanation-detail">${feat}: ${sign}${(impact * 100).toFixed(1)}%</span><span class="explanation-detail">${effect}</span></div>`;
      });
    }
  } else {
    explList.innerHTML = "<p style='font-size: 0.875rem; color: #888;'>No explanations available.</p>";
  }

  result.classList.remove("hidden");
}

function showError(message) {
  result.innerHTML = `<div class="error">${message}</div>`;
  result.classList.remove("hidden");
}
