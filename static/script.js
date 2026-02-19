console.log("script loaded");

document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("csvForm");
    console.log("Form:", form);

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        console.log("Submit clicked");

        const formData = new FormData(form);

        const res = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        console.log("Upload response:", data);
    });

});
