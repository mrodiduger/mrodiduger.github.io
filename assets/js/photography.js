const entries = document.querySelectorAll(".photo-entry");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (entries.length > 0 && "IntersectionObserver" in window && !prefersReducedMotion) {
  document.body.classList.add("motion-ready");

  const observer = new IntersectionObserver(
    (observations) => {
      observations.forEach((observation) => {
        if (!observation.isIntersecting) return;

        observation.target.classList.add("is-visible");
        observer.unobserve(observation.target);
      });
    },
    { rootMargin: "0px 0px -10% 0px", threshold: 0.08 },
  );

  entries.forEach((entry) => observer.observe(entry));
}
