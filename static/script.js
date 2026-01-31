const form = document.querySelector("#csvFile")
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);

  const response = await fetch("http://127.0.0.1:5000/load_csv", {
    method: "POST",
    body: formData,
  });
  
});
fetch("/upload", {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(data => {
  const select = document.getElementById("targetSelect");

  data.columns.forEach(col => {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    select.appendChild(option);
  });
});

const target = document.getElementById("targetSelect").value;

fetch("/analyze", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ target })
})
.then(res => res.json())
.then(data => {
  console.log(data);
});
