const ESTIMATED_LABELS = {
    ap_hi: "Systolic BP",
    ap_lo: "Diastolic BP",
    cholesterol: "Cholesterol",
    gluc: "Glucose",
};

const RISK_COLORS = {
    Low: "#111",
    Moderate: "#c8870a",
    High: "#c00",
};

function renderResult(predict, recs, expls) {
    const { risk_percent: pct, risk_label: label } = predict;

    document.querySelector(".risk-display").className =
        `risk-display risk-${label.toLowerCase()}`;
    document.getElementById("riskPercent").textContent = `${pct.toFixed(1)}%`;
    document.getElementById("riskLabel").textContent = `${label} risk`;

    const fill = document.getElementById("barFill");
    fill.style.width = `${pct}%`;
    fill.style.background = RISK_COLORS[label] ?? "#111";

    renderEstimated(predict.estimated ?? {});
    renderRecs(recs.recommendations ?? []);
    renderModels(predict.models ?? {});
    renderExpls(expls.explanations ?? {});
}

function renderEstimated(estimated) {
    const section = document.getElementById("estimated");
    const list = document.getElementById("estimatedList");

    const entries = Object.entries(estimated);
    if (entries.length === 0) {
        section.hidden = true;
        return;
    }

    list.innerHTML = entries
        .map(([k, v]) => `<dt>${ESTIMATED_LABELS[k] ?? k}</dt><dd>${v}</dd>`)
        .join("");
    section.hidden = false;
}

function renderRecs(recs) {
    const el = document.getElementById("recsList");
    if (recs.length === 0) {
        el.innerHTML = `<p class="empty-note">No actionable recommendations.</p>`;
        return;
    }
    el.innerHTML =
        `<div class="detail-inner">` +
        recs
            .map(
                (r) => `
        <div class="rec-item">
            <span class="rec-action">${r.action}</span>
            <span class="rec-detail">Reduces risk by ${(r.reduction * 100).toFixed(1)}% → new risk: ${(r.new_prob * 100).toFixed(1)}%</span>
        </div>`,
            )
            .join("") +
        `</div>`;
}

function renderModels(models) {
    const el = document.getElementById("modelsList");
    el.innerHTML =
        `<div class="detail-inner">` +
        Object.entries(models)
            .map(
                ([name, prob]) => `
        <div class="row-item">
            <span class="row-item-label">${name}</span>
            <span class="row-item-value">${(prob * 100).toFixed(1)}%</span>
        </div>`,
            )
            .join("") +
        `</div>`;
}

function renderExpls(explanations) {
    const el = document.getElementById("explList");
    const entries = Object.entries(explanations);
    if (entries.length === 0) {
        el.innerHTML = `<p class="empty-note">No explanations available.</p>`;
        return;
    }
    el.innerHTML =
        `<div class="detail-inner">` +
        entries
            .map(([model, pairs]) => {
                const feats = pairs
                    .map(([feat, impact]) => {
                        const sign = impact >= 0 ? "+" : "";
                        const effect = impact >= 0 ? "↑ risk" : "↓ risk";
                        return `<div class="expl-feat">
                    <span>${feat}</span>
                    <span>${sign}${(impact * 100).toFixed(1)}% ${effect}</span>
                </div>`;
                    })
                    .join("");
                return `<div class="expl-model">${model}</div>${feats}`;
            })
            .join("") +
        `</div>`;
}

const ALL_PLOT_MODELS = [
    "Decision Tree",
    "KNN",
    "LightGBM",
    "Linear SVM",
    "Logistic Regression",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
];

function toSnakeCase(name) {
    return name.toLowerCase().replace(/\s+/g, "_");
}

function loadAndRenderModels() {
    const tabsContainer = document.querySelector(".model-tabs");
    const viewerContainer = document.querySelector(".model-viewer");

    if (!tabsContainer || !viewerContainer) return;

    tabsContainer.innerHTML = ALL_PLOT_MODELS.map((name, i) =>
        `<button type="button" class="model-tab${i === 0 ? " active" : ""}" data-model="${name}">${name}</button>`
    ).join("");

    viewerContainer.innerHTML = ALL_PLOT_MODELS.map((name, i) => {
        const key = toSnakeCase(name);
        return `<div class="model-images" data-panel="${name}" style="${i !== 0 ? "display:none" : ""}">
            <img src="output/plots/learning_curves/${key}_learning_curve.png" class="plot-img" onclick="openModal(this.src)" onerror="this.style.display='none'">
            <img src="output/plots/confusion_matrices/${key}_confusion_matrix.png" class="plot-img" onclick="openModal(this.src)" onerror="this.style.display='none'">
            <img src="output/plots/roc/${key}_roc_curve.png" class="plot-img" onclick="openModal(this.src)" onerror="this.style.display='none'">
        </div>`;
    }).join("");

    tabsContainer.addEventListener("click", (e) => {
        const tab = e.target.closest(".model-tab");
        if (!tab) return;

        const selected = tab.getAttribute("data-model");

        tabsContainer.querySelectorAll(".model-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        viewerContainer.querySelectorAll(".model-images").forEach(panel => {
            panel.style.display = panel.getAttribute("data-panel") === selected ? "" : "none";
        });
    });
}

document.addEventListener("DOMContentLoaded", loadAndRenderModels);

