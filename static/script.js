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

