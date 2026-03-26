

// Active region switching
(() => {
  const frame = document.getElementById("plotFrame");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");

  if (!frame || !prevBtn || !nextBtn) return;

  const total = 5;
  let idx = 1;

  function setFrame(n) {
    idx = n;
    frame.src = `graphs/activePart${idx}.html`;
  }

  prevBtn.addEventListener("click", () => {
    const next = idx - 1 < 1 ? total : idx - 1;
    setFrame(next);
  });

  nextBtn.addEventListener("click", () => {
    const next = idx + 1 > total ? 1 : idx + 1;
    setFrame(next);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") prevBtn.click();
    if (e.key === "ArrowRight") nextBtn.click();
  });
})();


// Collapsible panels
document.querySelectorAll(".collapse-header").forEach(header => {
  header.addEventListener("click", () => {
    const collapse = header.parentElement;
    collapse.classList.toggle("open");
  });
});