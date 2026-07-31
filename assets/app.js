(() => {
  const variableButtons = [...document.querySelectorAll("[data-variable-button]")];
  const dayButtons = [...document.querySelectorAll("[data-day-button]")];
  const validationCityButtons = [...document.querySelectorAll("[data-validation-city]")];
  const validationVariableButtons = [...document.querySelectorAll("[data-validation-variable]")];
  const validationImage = document.querySelector("#validation-image");
  const validationSummary = document.querySelector("#validation-summary");
  const matchInitSelect = document.querySelector("#match-init-select");
  const matchImage = document.querySelector("#match-image");
  const runSelect = document.querySelector("#run-select");
  const runSummary = document.querySelector("#run-summary");
  const views = [...document.querySelectorAll(".forecast-view")];
  const siteData = JSON.parse(document.querySelector("#archive-data").textContent);
  const runs = siteData.runs;
  const validation = siteData.validation;
  const params = new URLSearchParams(window.location.search);
  const allowedVariables = new Set(["temperature", "precipitation"]);
  const allowedDays = new Set(["1", "2", "3"]);
  const allowedInits = new Set(runs.map((run) => run.id));
  let variable = allowedVariables.has(params.get("variable")) ? params.get("variable") : "temperature";
  let day = allowedDays.has(params.get("day")) ? params.get("day") : "1";
  let init = allowedInits.has(params.get("init")) ? params.get("init") : runs[0].id;
  let validationCity = Object.keys(validation.cities).includes(params.get("city")) ? params.get("city") : Object.keys(validation.cities)[0];
  let validationVariable = allowedVariables.has(params.get("validation")) ? params.get("validation") : "temperature";
  let matchInit = allowedInits.has(params.get("match_init")) ? params.get("match_init") : runs[0].id;

  function render(updateUrl = true) {
    variableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.variableButton === variable)));
    dayButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.dayButton === day)));
    views.forEach((view) => { view.hidden = !(view.dataset.init === init && view.dataset.variable === variable && view.dataset.day === day); });
    const active = runs.find((run) => run.id === init);
    runSelect.value = init;
    runSummary.textContent = `Initialized ${new Date(active.initialization_utc).toLocaleString("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false })} UTC · 6 models · 3-day forecast`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("init", init);
      next.searchParams.set("variable", variable);
      next.searchParams.set("day", day);
      history.replaceState(null, "", next);
    }
  }

  function renderValidation(updateUrl = true) {
    validationCityButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationCity === validationCity)));
    validationVariableButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.validationVariable === validationVariable)));
    const active = validation.cities[validationCity];
    const image = active.images[validationVariable];
    const points = active.summary[validationVariable].matched_points;
    validationImage.src = image.path;
    validationImage.alt = image.alt;
    validationSummary.textContent = `${validationCity} · ${points} matched forecast–observation pairs per model · Open-Meteo ground truth`;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("city", validationCity);
      next.searchParams.set("validation", validationVariable);
      history.replaceState(null, "", next);
    }
    renderMatchedTimeseries(updateUrl);
  }

  function renderMatchedTimeseries(updateUrl = true) {
    const image = validation.cities[validationCity].timeseries[matchInit][validationVariable];
    matchInitSelect.value = matchInit;
    matchImage.src = image.path;
    matchImage.alt = image.alt;
    if (updateUrl) {
      const next = new URL(window.location.href);
      next.searchParams.set("match_init", matchInit);
      history.replaceState(null, "", next);
    }
  }

  variableButtons.forEach((button) => button.addEventListener("click", () => { variable = button.dataset.variableButton; render(); }));
  dayButtons.forEach((button) => button.addEventListener("click", () => { day = button.dataset.dayButton; render(); }));
  runSelect.addEventListener("change", () => { init = runSelect.value; render(); });
  validationCityButtons.forEach((button) => button.addEventListener("click", () => { validationCity = button.dataset.validationCity; renderValidation(); }));
  validationVariableButtons.forEach((button) => button.addEventListener("click", () => { validationVariable = button.dataset.validationVariable; renderValidation(); }));
  matchInitSelect.addEventListener("change", () => { matchInit = matchInitSelect.value; renderMatchedTimeseries(); });
  render(false);
  renderValidation(false);
})();
