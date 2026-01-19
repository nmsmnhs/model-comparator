const form = document.querySelector("#csvFile")
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);

  const response = await fetch("http://127.0.0.1:5000/load_csv", {
    method: "POST",
    body: formData,
  });
  
});