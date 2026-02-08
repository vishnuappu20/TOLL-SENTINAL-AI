function login() {
  const role = document.getElementById("role").value;

  if (role === "admin") {
    window.location.href = "admin_dashboard.html";
  } else {
    window.location.href = "dashboard.html";
  }
}

function logout() {
  window.location.href = "index.html";
}
