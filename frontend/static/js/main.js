/**
 * MAYA foundation scripts.
 * Chart.js and AOS are loaded globally; page-specific charts come later.
 */
document.addEventListener("DOMContentLoaded", () => {
  if (window.AOS) {
    AOS.init({
      duration: 600,
      once: true,
      easing: "ease-out-cubic",
    });
  }
});
