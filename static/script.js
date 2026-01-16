const form = document.querySelector("#csvFile")
form.addEventListener("submit", async (event) => {
  const formData = new FormData(form);

  const response = await fetch("http://localhost:5000/", {
    method: "POST",
    body: formData,
  });
  event.preventDefault();
});