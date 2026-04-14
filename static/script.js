document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("csvForm");
    console.log("Form:", form);
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            console.log("Submit clicked");

            const formData = new FormData(form);
            try {
                const res = await fetch("/upload", { method: "POST", body: formData });
                const data = await res.json();

                if (!res.ok || data.error) {
                    setStatus(uploadStatus, data.error || "Upload failed.", "error");
                    resetBtn(submitBtn);
                    return;
                }
            } catch (err) {
                setStatus(uploadStatus, `Network error: ${err.message}`, "error");
                resetBtn(submitBtn);
            }
        });

        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", async () => {
                const target = targetSelect.value;
                if (!target) { setStatus(analyzeStatus, "Please select a target variable.", "error"); return; }

                showSpinner(analyzeBtn, " Analyzing…");
                setStatus(analyzeStatus, "Training models — this may take a moment…", "info");

                try {
                    const res = await fetch("/analyze", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ target, session_id: sessionId })
                    });
                    const data = await res.json();

                    if (!res.ok || data.error) {
                        setStatus(analyzeStatus, data.error || "Analysis failed.", "error");
                        resetBtn(analyzeBtn);
                        return;
                    }
                    sessionStorage.setItem("mlResults", JSON.stringify(data));
                    window.location.href = "/results";
                } catch (err) {
                    setStatus(analyzeStatus, `Network error: ${err.message}`, "error");
                    resetBtn(analyzeBtn);
                }
            });
        }
    }
    const resultsRoot = document.getElementById("resultsRoot");
    if (resultsRoot) {
        const raw = sessionStorage.getItem("mlResults");
        if (!raw) {
            resultsRoot.innerHTML = `<p class="status status--error">No results found. <a href="/">Run an analysis first →</a></p>`;
            return;
        }
        renderResults(JSON.parse(raw));
    }
});
function renderResults(data) {
    const { results, roc_curves, best_models } = data;
    const models = Object.keys(results);
    const colors = {
        "Logistic Regression": { border: "#2563eb", bg: "rgba(37,99,235,0.15)" },
        "Decision Tree": { border: "#16a34a", bg: "rgba(22,163,74,0.15)" },
        "Random Forest": { border: "#dc2626", bg: "rgba(220,38,38,0.15)" }
    };

    // ── Best Models Banner ──
    const banner = document.getElementById("bestModelsBanner");
    if (banner && best_models) {
        const cards = [
            { label: "Best Accuracy", key: "best_accuracy_model", icon: "◈" },
            { label: "Best Precision", key: "best_precision_model", icon: "◉" },
            { label: "Best AUC", key: "best_auc_model", icon: "◍" },
        ];
        banner.innerHTML = cards.map(c => {
            const val = best_models[c.key];
            if (!val) return "";
            return `<div class="best-card">
                <span class="best-icon">${c.icon}</span>
                <span class="best-label">${c.label}</span>
                <span class="best-model">${val[0]}</span>
                <span class="best-score">${(val[1] * 100).toFixed(2)}%</span>
            </div>`;
        }).join("");
    }

    // ── Metrics Table ──
    const tableBody = document.getElementById("metricsTableBody");
    if (tableBody) {
        tableBody.innerHTML = models.map(m => {
            const r = results[m];
            const auc = roc_curves[m] ? roc_curves[m].auc : "—";
            return `<tr>
                <td><span class="model-dot" style="background:${colors[m]?.border}"></span>${m}</td>
                <td>${(r.accuracy * 100).toFixed(2)}%</td>
                <td>${(r.precision * 100).toFixed(2)}%</td>
                <td>${typeof auc === "number" ? auc.toFixed(4) : auc}</td>
                <td>${r.training_time}s</td>
            </tr>`;
        }).join("");
    }

    // ── Bar Chart: Accuracy & Precision ──
    const barCtx = document.getElementById("barChart")?.getContext("2d");
    if (barCtx) {
        new Chart(barCtx, {
            type: "bar",
            data: {
                labels: models,
                datasets: [
                    {
                        label: "Accuracy",
                        data: models.map(m => results[m].accuracy),
                        backgroundColor: models.map(m => colors[m]?.bg),
                        borderColor: models.map(m => colors[m]?.border),
                        borderWidth: 2,
                        borderRadius: 4
                    },
                    {
                        label: "Precision",
                        data: models.map(m => results[m].precision),
                        backgroundColor: models.map(m => colors[m]?.bg.replace("0.15", "0.35")),
                        borderColor: models.map(m => colors[m]?.border),
                        borderWidth: 2,
                        borderRadius: 4,
                        borderDash: [4, 4]
                    }
                ]
            },
            options: chartOptions("Accuracy & Precision", "Score", [0, 1])
        });
    }

    // ── ROC Chart ──
    const rocCtx = document.getElementById("rocChart")?.getContext("2d");
    if (rocCtx) {
        const rocSection = document.getElementById("rocSection");
        if (Object.keys(roc_curves).length === 0) {
            if (rocSection) rocSection.classList.add("hidden");
        } else {
            const datasets = Object.entries(roc_curves).map(([name, d]) => ({
                label: `${name} (AUC ${d.auc})`,
                data: d.fpr.map((x, i) => ({ x, y: d.tpr[i] })),
                borderColor: colors[name]?.border,
                backgroundColor: "transparent",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }));
            // Diagonal reference
            datasets.push({
                label: "Random Classifier",
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: "#94a3b8",
                borderDash: [6, 4],
                borderWidth: 1,
                pointRadius: 0,
                backgroundColor: "transparent"
            });
            new Chart(rocCtx, {
                type: "line",
                data: { datasets },
                options: {
                    ...chartOptions("ROC Curves", "True Positive Rate", [0, 1]),
                    parsing: false,
                    scales: {
                        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate", color: "#64748b" }, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } },
                        y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate", color: "#64748b" }, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } }
                    }
                }
            });
        }
    }

    // ── Confusion Matrices ──
    const cmRoot = document.getElementById("confusionMatrices");
    if (cmRoot) {
        cmRoot.innerHTML = models.map(m => {
            const cm = results[m].confusion_matrix;
            const rows = cm.map(row =>
                `<tr>${row.map(v => `<td>${v}</td>`).join("")}</tr>`
            ).join("");
            return `<div class="cm-block">
                <h4><span class="model-dot" style="background:${colors[m]?.border}"></span>${m}</h4>
                <table class="cm-table">${rows}</table>
            </div>`;
        }).join("");
    }
}

function chartOptions(title, yLabel, yRange) {
    return {
        responsive: true,
        plugins: {
            legend: { labels: { color: "#94a3b8", font: { family: "'DM Mono', monospace" } } },
            title: { display: false }
        },
        scales: {
            x: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { family: "'DM Mono', monospace" } } },
            y: {
                min: yRange?.[0], max: yRange?.[1],
                title: { display: true, text: yLabel, color: "#64748b" },
                grid: { color: "#1e293b" },
                ticks: { color: "#94a3b8", font: { family: "'DM Mono', monospace" } }
            }
        }
    };
}

