(() => {
  const variableButtons = [...document.querySelectorAll("[data-variable-button]")];
  const dayButtons = [...document.querySelectorAll("[data-day-button]")];
  const views = [...document.querySelectorAll(".forecast-view")];
  const params = new URLSearchParams(window.location.search);
  const allowedVariables = new Set(["temperature", "precipitation"]);
  const allowedDays = new Set(["1", "2", "3"]);
  let variable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let day = allowedDays.has(params.get("day")) ? params.get("day") : "1";

  function render(updateUrl = true) {
    variableButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.variableButton === variable));
    });
    dayButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.dayButton === day));
    });
    views.forEach((view) => {
      view.hidden = !(view.dataset.variable === variable && view.dataset.day === day);
    });
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("variable", variable);
      next.searchParams.set("day", day);
      history.replaceState(null, "", next);
    }
  }

  variableButtons.forEach((button) => {
    button.addEventListener("click", () => {
      variable = button.dataset.variableButton;
      render();
    });
  });
  dayButtons.forEach((button) => {
    button.addEventListener("click", () => {
      day = button.dataset.dayButton;
      render();
    });
  });
  render(false);
})();
