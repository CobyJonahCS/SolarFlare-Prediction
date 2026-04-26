

// Active region switching
// Active region switching (specific graph list)
(() => {
  const frame = document.getElementById("plotFrame");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");

  if (!frame || !prevBtn || !nextBtn) return;

  const graphs = [
    "ABSNJZH_residual_distribution.html",
    "R_VALUE_residual_distribution.html",
    "TOTBSQ_residual_distribution.html",
    "TOTPOT_residual_distribution.html",
    "TOTUSJH_residual_distribution.html",
    "TOTUSJZ_residual_distribution.html"
  ];

  let idx = 0;

  function setFrame(n) {
    idx = n;
    frame.src = `../graphs/forecasts/${graphs[idx]}`;
  }

  prevBtn.addEventListener("click", () => {
    setFrame((idx - 1 + graphs.length) % graphs.length);
  });

  nextBtn.addEventListener("click", () => {
    setFrame((idx + 1) % graphs.length);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") prevBtn.click();
    if (e.key === "ArrowRight") nextBtn.click();
  });

  setFrame(0);
})();

// Feature importance switching
(() => {
  const frame = document.getElementById("featureFrame");
  const prevBtn = document.getElementById("featurePrevBtn");
  const nextBtn = document.getElementById("featureNextBtn");

  if (!frame || !prevBtn || !nextBtn) return;

  const graphs = [
    "ABSNJZH.html",
    "R_VALUE.html",
    "TOTBSQ.html",
    "TOTPOT.html",
    "TOTUSJH.html",
    "TOTUSJZ.html"
  ];

  let idx = 0;

  function setFrame(n) {
    idx = n;
    frame.src = `../graphs/forecasts/${graphs[idx]}`;
  }

  prevBtn.addEventListener("click", () => {
    setFrame((idx - 1 + graphs.length) % graphs.length);
  });

  nextBtn.addEventListener("click", () => {
    setFrame((idx + 1) % graphs.length);
  });

  setFrame(0);
})();

// Collapsible panels
document.querySelectorAll(".collapse-header").forEach(header => {
  header.addEventListener("click", () => {
    const collapse = header.parentElement;
    collapse.classList.toggle("open");
  });
});


