const state = {
  league: "NBA",
  bootstrap: null,
  features: {},
  selectedPreset: null,
};

const FEATURE_META = [
  ["elo_diff", "Elo Differential", -2, 2, 0.05],
  ["offensive_diff", "Offensive Edge", -2, 2, 0.05],
  ["defensive_diff", "Defensive Edge", -2, 2, 0.05],
  ["injury_diff", "Health Edge", -2, 2, 0.05],
  ["rest_diff", "Rest Edge", -2, 2, 0.05],
  ["form_diff", "Recent Form", -2, 2, 0.05],
  ["qb_status_diff", "QB / Primary Creator", -2, 2, 0.05],
  ["travel_diff", "Travel Edge", -2, 2, 0.05],
  ["market_spread", "Market Spread", -10, 10, 0.1],
];

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetric(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }
  if (key.includes("loss") || key === "brier") {
    return value.toFixed(3);
  }
  return `${(value * 100).toFixed(1)}%`;
}

function metricLabel(key) {
  const labels = {
    accuracy: "Overall Accuracy",
    baseline_accuracy: "Baseline Accuracy",
    log_loss: "Model Log Loss",
    nba_accuracy: "NBA Accuracy",
    nfl_accuracy: "NFL Accuracy",
    brier: "Brier Score",
  };
  return labels[key] || key;
}

async function fetchBootstrap() {
  const response = await fetch("/api/bootstrap");
  state.bootstrap = await response.json();
  state.features = { ...state.bootstrap.sample_input[state.league] };
}

function renderMetrics() {
  const container = document.getElementById("metricCards");
  document.getElementById("dataSourceLabel").textContent = state.bootstrap.data_source;
  document.getElementById("tapeLabel").textContent =
    state.bootstrap.metadata?.data_source === "real_nba_history" ? "Real NBA history" : "Recent slate";
  const keys = ["accuracy", "baseline_accuracy", "nba_accuracy", "nfl_accuracy", "log_loss", "brier"];
  container.innerHTML = keys
    .map((key) => {
      const value = state.bootstrap.metrics[key];
      const formatted = formatMetric(key, value);
      if (formatted === null) {
        return "";
      }
      return `
        <article class="metric-card">
          <span>${metricLabel(key)}</span>
          <strong>${formatted}</strong>
        </article>
      `;
    })
    .join("");
}

function renderRibbon() {
  const ribbon = document.getElementById("summaryRibbon");
  const meta = state.bootstrap.metadata || {};
  const chips = [
    `Data source: ${state.bootstrap.data_source}`,
    meta.seasons ? `Seasons: ${meta.seasons.join(", ")}` : null,
    meta.row_count ? `Rows modeled: ${meta.row_count}` : null,
    state.bootstrap.metrics.accuracy ? `Model edge: ${(state.bootstrap.metrics.accuracy * 100 - state.bootstrap.metrics.baseline_accuracy * 100).toFixed(1)} pts over baseline` : null,
  ].filter(Boolean);
  ribbon.innerHTML = chips.map((chip) => `<div class="summary-chip">${chip}</div>`).join("");
}

function teamProfile(team) {
  return state.bootstrap.team_profiles?.[team] || null;
}

function teamSnapshot(team) {
  return state.bootstrap.team_snapshots?.[team] || null;
}

function teamLabel(team) {
  return state.bootstrap.team_labels?.[team] || team;
}

function setBuilderStatus(message, isError = false) {
  const node = document.getElementById("builderStatus");
  if (!node) {
    return;
  }
  node.textContent = message;
  node.style.color = isError ? "var(--red)" : "var(--muted)";
}

function normalizeTeamInput(value) {
  return (value || "").trim().toUpperCase();
}

function leagueTeams() {
  return state.bootstrap.teams?.[state.league] || [];
}

function applyTeamSelection(side, team) {
  const normalized = normalizeTeamInput(team);
  document.getElementById(`${side}Search`).value = normalized;
  renderTeamOptions();
}

function buildFeaturesFromTeams(homeTeam, awayTeam, league = "NBA") {
  const home = teamSnapshot(homeTeam);
  const away = teamSnapshot(awayTeam);
  if (!home || !away) {
    return null;
  }

  const leagueIsNba = league === "NBA" ? 1.0 : 0.0;
  const eloDiff = (home.elo - away.elo) / 100.0;
  const offensiveDiff = (home.points_for - away.points_for) / 10.0;
  const defensiveDiff = (away.points_against - home.points_against) / 10.0;
  const formDiff = home.form - away.form;
  const restDiff = 0.0;
  const marketSpread = -(0.9 * eloDiff + 0.7 * (home.avg_margin - away.avg_margin) + 0.45 * restDiff);

  return {
    league_is_nba: leagueIsNba,
    elo_diff: Number(eloDiff.toFixed(4)),
    offensive_diff: Number(offensiveDiff.toFixed(4)),
    defensive_diff: Number(defensiveDiff.toFixed(4)),
    injury_diff: 0.0,
    rest_diff: Number(restDiff.toFixed(4)),
    form_diff: Number(formDiff.toFixed(4)),
    qb_status_diff: 0.0,
    travel_diff: 0.15,
    market_spread: Number(marketSpread.toFixed(4)),
  };
}

function renderTeamOptions() {
  const awayInput = document.getElementById("awaySearch");
  const homeInput = document.getElementById("homeSearch");
  if (!awayInput || !homeInput) {
    return;
  }
  if (!awayInput.value) {
    awayInput.value = "";
  }
  if (!homeInput.value) {
    homeInput.value = "";
  }
  renderSuggestionList("away");
  renderSuggestionList("home");
  renderTeamBoard("away");
  renderTeamBoard("home");
}

function filteredTeams(query) {
  const teams = leagueTeams();
  const normalized = normalizeTeamInput(query);
  if (!normalized) {
    return teams.slice(0, 8);
  }
  return teams
    .filter((team) => {
      const label = teamLabel(team).toUpperCase();
      return team.includes(normalized) || label.includes(normalized);
    })
    .slice(0, 10);
}

function renderTeamBoard(side) {
  const container = document.getElementById(`${side}TeamBoard`);
  const input = document.getElementById(`${side}Search`);
  if (!container || !input) {
    return;
  }

  const selected = normalizeTeamInput(input.value);
  container.innerHTML = leagueTeams()
    .map((team) => {
      const active = selected === team ? "active" : "";
      return `
        <button type="button" class="team-chip ${active}" data-side="${side}" data-team="${team}">
          <span class="team-chip-code">${team}</span>
          <span class="team-chip-name">${teamLabel(team)}</span>
        </button>
      `;
    })
    .join("");

  container.querySelectorAll(".team-chip").forEach((button) => {
    button.addEventListener("click", () => {
      applyTeamSelection(side, button.dataset.team);
      setBuilderStatus(`${teamLabel(button.dataset.team)} set as the ${side} team.`);
    });
  });
}

function inferLeagueForTeams(homeTeam, awayTeam) {
  const nbaTeams = new Set(state.bootstrap.teams?.NBA || []);
  const nflTeams = new Set(state.bootstrap.teams?.NFL || []);
  if (nbaTeams.has(homeTeam) && nbaTeams.has(awayTeam)) {
    return "NBA";
  }
  if (nflTeams.has(homeTeam) && nflTeams.has(awayTeam)) {
    return "NFL";
  }
  return state.league;
}

function renderSuggestionList(side) {
  const input = document.getElementById(`${side}Search`);
  const container = document.getElementById(`${side}Suggestions`);
  if (!input || !container) {
    return;
  }

  const selected = normalizeTeamInput(input.value);
  const matches = filteredTeams(input.value);
  container.innerHTML = matches
    .map((team) => {
      const isActive = selected === team;
      return `<button type="button" class="team-option ${isActive ? "active" : ""}" data-side="${side}" data-team="${team}">${team} - ${teamLabel(team)}</button>`;
    })
    .join("");

  container.querySelectorAll(".team-option").forEach((button) => {
    button.addEventListener("click", () => {
      applyTeamSelection(side, button.dataset.team);
    });
  });
}

function renderQuickPicks() {
  const container = document.getElementById("quickPicks");
  if (!container) {
    return;
  }
  const picks = state.bootstrap.presets.slice(0, 4);
  container.innerHTML = picks
    .map(
      (preset, index) =>
        `<button class="quick-pick" data-index="${index}">${preset.away_team} at ${preset.home_team}</button>`
    )
    .join("");

  container.querySelectorAll(".quick-pick").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = picks[Number(button.dataset.index)];
      document.getElementById("awaySearch").value = preset.away_team;
      document.getElementById("homeSearch").value = preset.home_team;
      state.selectedPreset = preset;
      state.league = preset.league;
      state.features = { ...preset.features };
      syncLeagueButtons();
      renderControls();
      runPrediction(preset.home_team, preset.away_team);
    });
  });
}

function renderTeamCompare(homeTeam, awayTeam) {
  const container = document.getElementById("teamCompare");
  const home = teamProfile(homeTeam);
  const away = teamProfile(awayTeam);

  function renderCard(team, profile, side) {
    if (!profile) {
      return `
        <article class="team-profile-card">
          <h3>${team}</h3>
          <p>No historical profile loaded.</p>
        </article>
      `;
    }

    const form = profile.last_five
      .map((result) => `<span class="${result ? "win" : "loss"}">${result ? "W" : "L"}</span>`)
      .join("");

    return `
      <article class="team-profile-card">
        <h3>${team} ${side}</h3>
        <p>Record: ${profile.wins}-${profile.losses}</p>
        <p>Win rate: ${(profile.win_pct * 100).toFixed(1)}%</p>
        <p>Scoring: ${profile.points_for} for / ${profile.points_against} allowed</p>
        <div class="mini-form">${form}</div>
      </article>
    `;
  }

  container.innerHTML = `${renderCard(awayTeam, away, "Away")} ${renderCard(homeTeam, home, "Home")}`;
}

function renderComparisonChart(homeTeam, awayTeam) {
  const container = document.getElementById("compareChart");
  const home = teamProfile(homeTeam);
  const away = teamProfile(awayTeam);
  if (!container) {
    return;
  }
  if (!home || !away) {
    container.innerHTML = "";
    return;
  }

  const categories = [
    {
      label: "Win Rate",
      away: away.win_pct,
      home: home.win_pct,
      format: (value) => formatPercent(value),
    },
    {
      label: "Points For",
      away: away.points_for,
      home: home.points_for,
      format: (value) => value.toFixed(1),
    },
    {
      label: "Points Allowed",
      away: away.points_against,
      home: home.points_against,
      format: (value) => value.toFixed(1),
    },
  ];

  container.innerHTML = categories
    .map((category) => {
      const total = Math.max(category.away + category.home, 0.0001);
      const awayWidth = `${(category.away / total) * 100}%`;
      const homeWidth = `${(category.home / total) * 100}%`;
      return `
        <article class="compare-row">
          <div class="compare-row-header">
            <strong>${category.label}</strong>
            <span>${awayTeam} vs ${homeTeam}</span>
          </div>
          <div class="compare-track">
            <div class="compare-away" style="width:${awayWidth}"></div>
            <div class="compare-home" style="width:${homeWidth}"></div>
          </div>
          <div class="compare-values">
            <span>${awayTeam}: ${category.format(category.away)}</span>
            <span>${homeTeam}: ${category.format(category.home)}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function buildNarrative(result, homeTeam, awayTeam) {
  const top = result.top_factors?.[0];
  if (!top) {
    return "The model is waiting for enough input to explain the edge.";
  }

  const edgeTeam = result.home_win_probability >= 0.5 ? homeTeam : awayTeam;
  const feature = top.feature.replaceAll("_", " ");
  const homeProfile = teamProfile(homeTeam);
  const awayProfile = teamProfile(awayTeam);
  const recordContext =
    homeProfile && awayProfile
      ? `${awayTeam} enters ${awayProfile.wins}-${awayProfile.losses} while ${homeTeam} sits at ${homeProfile.wins}-${homeProfile.losses}. `
      : "";
  return `${recordContext}${edgeTeam} carries the current edge at ${formatPercent(Math.max(result.home_win_probability, result.away_win_probability))}. The strongest signal right now is ${feature}, and the model grades this matchup as ${result.confidence.toLowerCase()} confidence.`;
}

function renderNarrative(result, homeTeam, awayTeam) {
  const card = document.getElementById("narrativeCard");
  card.innerHTML = `
    <h3>Desk Read</h3>
    <p>${buildNarrative(result, homeTeam, awayTeam)}</p>
  `;
}

function renderPresets() {
  const container = document.getElementById("presetGrid");
  document.getElementById("presetLabel").textContent =
    state.bootstrap.metadata?.data_source === "real_nba_history" ? "Real game presets" : "Quick presets";
  container.innerHTML = state.bootstrap.presets
    .map(
      (preset, index) => `
        <article class="preset-card" data-index="${index}">
          <div class="preset-top">
            <strong>${preset.label}</strong>
            <span class="league-badge">${preset.league}</span>
          </div>
          <div class="factor-meta">${preset.away_team} at ${preset.home_team}</div>
          <div class="preset-bottom">
            <span>Elo ${Number(preset.features.elo_diff).toFixed(2)}</span>
            <span>Spread ${Number(preset.features.market_spread).toFixed(1)}</span>
          </div>
        </article>
      `
    )
    .join("");

  container.querySelectorAll(".preset-card").forEach((card) => {
    card.addEventListener("click", () => {
      container.querySelectorAll(".preset-card").forEach((node) => node.classList.remove("active"));
      card.classList.add("active");
      const preset = state.bootstrap.presets[Number(card.dataset.index)];
      state.selectedPreset = preset;
      state.league = preset.league;
      state.features = { ...preset.features };
      syncLeagueButtons();
      renderControls();
      runPrediction(preset.home_team, preset.away_team);
    });
  });
}

function renderTape() {
  const container = document.getElementById("matchupTape");
  container.innerHTML = state.bootstrap.latest_matchups
    .map(
      (item) => `
        <article class="tape-card">
          <div class="tape-top">
            <strong class="tape-title">${item.away_team} at ${item.home_team}</strong>
            <span class="league-badge">${item.league}</span>
          </div>
          <div class="tape-meta">${item.game_date ?? ""}</div>
          <div class="tape-meta">${item.market_spread === null || item.market_spread === undefined ? "Historical result" : `Market spread: ${item.market_spread}`}</div>
          <div class="tape-result">${item.home_win ? item.home_team : item.away_team} closed with the edge</div>
        </article>
      `
    )
    .join("");
}

function syncLeagueButtons() {
  document.querySelectorAll(".toggle").forEach((button) => {
    button.classList.toggle("active", button.dataset.league === state.league);
  });
  const teams = new Set(leagueTeams());
  ["away", "home"].forEach((side) => {
    const input = document.getElementById(`${side}Search`);
    if (input && input.value && !teams.has(normalizeTeamInput(input.value))) {
      input.value = "";
    }
  });
  renderTeamOptions();
}

function renderControls() {
  const form = document.getElementById("controlForm");
  form.innerHTML = FEATURE_META.map(([key, label, min, max, step]) => {
    const disabled = state.league === "NBA" && key === "qb_status_diff";
    const value = state.features[key];
    return `
      <div class="control-group">
        <label for="${key}">
          <span>${label}</span>
          <span id="${key}Value">${Number(value).toFixed(2)}</span>
        </label>
        <input
          id="${key}"
          type="range"
          min="${min}"
          max="${max}"
          step="${step}"
          value="${value}"
          ${disabled ? "disabled" : ""}
        />
      </div>
    `;
  }).join("");

  FEATURE_META.forEach(([key]) => {
    const input = document.getElementById(key);
    input.addEventListener("input", () => {
      state.features[key] = Number(input.value);
      document.getElementById(`${key}Value`).textContent = Number(input.value).toFixed(2);
    });
  });

  state.features.league_is_nba = state.league === "NBA" ? 1.0 : 0.0;
  if (state.league === "NBA") {
    state.features.qb_status_diff = 0.0;
  }
}

function renderFactors(topFactors) {
  const container = document.getElementById("factorList");
  container.innerHTML = topFactors
    .map((factor) => {
      const width = `${Math.max(12, Math.min(100, Math.abs(factor.impact) * 100))}%`;
      const klass = factor.direction === "home_edge" ? "home-edge" : "away-edge";
      return `
        <article class="factor-card">
          <div class="factor-row">
            <strong class="factor-title">${factor.feature.replaceAll("_", " ")}</strong>
            <span class="factor-meta">${factor.direction === "home_edge" ? "Home Edge" : "Away Edge"}</span>
          </div>
          <div class="factor-meter"><div class="${klass}" style="width:${width}"></div></div>
          <div class="factor-meta">Impact score: ${factor.impact}</div>
        </article>
      `;
    })
    .join("");
}

function updateHero(result, home = "Home Team", away = "Road Team") {
  document.getElementById("heroConfidence").textContent = result.confidence;
  document.getElementById("heroHome").textContent = home;
  document.getElementById("heroAway").textContent = away;
  document.getElementById("homeToken").textContent = home.slice(0, 3);
  document.getElementById("awayToken").textContent = away.slice(0, 3);
  document.getElementById("homeProbability").textContent = formatPercent(result.home_win_probability);
  document.getElementById("awayProbability").textContent = formatPercent(result.away_win_probability);
  document.getElementById("probabilityFill").style.width = `${result.home_win_probability * 100}%`;
  document.getElementById("heroMeta").textContent = `${away} at ${home}`;
  document.getElementById("heroSubstats").innerHTML = `
    <div class="summary-chip">${result.confidence} confidence</div>
    <div class="summary-chip">Home edge: ${formatPercent(result.home_win_probability)}</div>
    <div class="summary-chip">Away edge: ${formatPercent(result.away_win_probability)}</div>
  `;
}

async function runPrediction(homeName = "HOME", awayName = "AWAY") {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features: state.features }),
  });
  const result = await response.json();
  updateHero(result, homeName, awayName);
  renderFactors(result.top_factors);
  renderTeamCompare(homeName, awayName);
  renderComparisonChart(homeName, awayName);
  renderNarrative(result, homeName, awayName);
}

function bindControls() {
  document.querySelectorAll(".toggle").forEach((button) => {
    button.addEventListener("click", () => {
      state.league = button.dataset.league;
      state.features = { ...state.bootstrap.sample_input[state.league] };
      state.selectedPreset = null;
      syncLeagueButtons();
      renderControls();
      setBuilderStatus(`Switched to ${state.league}. Choose teams below or use a quick matchup.`);
      runPrediction();
    });
  });

  document.getElementById("predictButton").addEventListener("click", () => {
    runPrediction();
  });

  document.getElementById("buildMatchupButton").addEventListener("click", () => {
    const awayTeam = normalizeTeamInput(document.getElementById("awaySearch").value);
    const homeTeam = normalizeTeamInput(document.getElementById("homeSearch").value);
    if (!awayTeam || !homeTeam || awayTeam === homeTeam) {
      setBuilderStatus("Pick two different teams to build the matchup.", true);
      return;
    }

    const inferredLeague = inferLeagueForTeams(homeTeam, awayTeam);
    const features = buildFeaturesFromTeams(homeTeam, awayTeam, inferredLeague);
    if (!features) {
      setBuilderStatus("I could not build that matchup from the loaded team history. Try NBA teams like DAL, LAL, MIA, or BOS.", true);
      return;
    }

    state.selectedPreset = {
      home_team: homeTeam,
      away_team: awayTeam,
      league: inferredLeague,
      features,
    };
    state.league = inferredLeague;
    state.features = { ...features };
    syncLeagueButtons();
    renderControls();
    setBuilderStatus(`Built ${awayTeam} at ${homeTeam} using ${inferredLeague} team history.`);
    runPrediction(homeTeam, awayTeam);
  });

  document.getElementById("awaySearch").addEventListener("input", () => {
    renderSuggestionList("away");
    renderTeamBoard("away");
  });
  document.getElementById("homeSearch").addEventListener("input", () => {
    renderSuggestionList("home");
    renderTeamBoard("home");
  });
  document.getElementById("awaySearch").addEventListener("focus", () => {
    renderSuggestionList("away");
  });
  document.getElementById("homeSearch").addEventListener("focus", () => {
    renderSuggestionList("home");
  });
  document.getElementById("awaySearch").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      document.getElementById("buildMatchupButton").click();
    }
  });
  document.getElementById("homeSearch").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      document.getElementById("buildMatchupButton").click();
    }
  });
}

async function init() {
  await fetchBootstrap();
  state.selectedPreset = state.bootstrap.presets[0] || null;
  renderMetrics();
  renderRibbon();
  renderPresets();
  renderQuickPicks();
  renderTape();
  syncLeagueButtons();
  renderTeamOptions();
  renderControls();
  bindControls();
  setBuilderStatus("Choose a real matchup from the team boards or quick picks, then build the case.");
  if (state.selectedPreset) {
    state.league = state.selectedPreset.league;
    state.features = { ...state.selectedPreset.features };
    syncLeagueButtons();
    renderControls();
    runPrediction(state.selectedPreset.home_team, state.selectedPreset.away_team);
  } else {
    runPrediction();
  }
}

init();
