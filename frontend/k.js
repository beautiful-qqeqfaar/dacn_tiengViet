const backendUrl = "http://127.0.0.1:8000"; // đổi nếu deploy khác host/port

const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const output = document.getElementById("output");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});

async function sendPredict(endpoint) {
  const file = fileInput.files[0];
  if (!file) {
    alert("Vui lòng chọn ảnh trước");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  output.textContent = "Đang xử lý";

  try {
    const res = await fetch(`${backendUrl}${endpoint}`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(`HTTP error! ${res.status}`);
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    console.error(err);
    output.textContent = "Lỗi khi gọi API.";
  }
}

document.getElementById("predictChar").addEventListener("click", () => {
  sendPredict("/predict/char");
});

document.getElementById("predictLine").addEventListener("click", () => {
  sendPredict("/predict/line");
});
