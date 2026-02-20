console.log("script loaded");

document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("csvForm");
    console.log("Form:", form);

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

        const data = await res.json();
        console.log("Upload response:", data);
    });

