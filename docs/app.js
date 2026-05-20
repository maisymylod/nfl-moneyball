(() => {
  "use strict";

  const STATE = {
    data: null,
    filters: {
      positions: new Set(["QB", "RB", "WR", "TE"]),
      minSnaps: 200,
      minConfidence: "Medium",
      matchedOnly: true,
    },
    sort: { bargains: { col: "value_score", dir: "desc" }, overpaid: { col: "value_score", dir: "asc" } },
    charts: {},
  };

  const CONF_ORDER = { Low: 0, Medium: 1, High: 2 };

  // ---------- formatting ----------
  function fmtNum(v, digits = 2) {
    if (v == null || Number.isNaN(v)) return "-";
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
  }
  function fmtMoney(v) {
    if (v == null || Number.isNaN(v)) return "-";
    return `$${Number(v).toFixed(1)}M`;
  }
  function fmtInt(v) {
    if (v == null || Number.isNaN(v)) return "-";
    return Math.round(Number(v)).toLocaleString();
  }
  function fmtDate(v) {
    if (!v) return "-";
    return String(v).slice(0, 10);
  }
  function badge(label) {
    if (!label) return "-";
    return `<span class="badge ${label}">${label}</span>`;
  }

  // ---------- data loading ----------
  async function loadData() {
    const resp = await fetch(`data.json?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`Failed to load data.json (${resp.status})`);
    return await resp.json();
  }

  // ---------- header ----------
  function renderHeader() {
    const d = STATE.data;
    document.getElementById("m-season").textContent = d.current_season ?? "-";
    document.getElementById("m-total").textContent = fmtInt(d.row_counts?.total_current);
    document.getElementById("m-eligible").textContent = fmtInt(d.row_counts?.eligible_for_ranking);
    document.getElementById("m-trained").textContent = fmtDate(d.model_trained_at);
    document.getElementById("m-generated").textContent = fmtDate(d.generated_at);
  }

  // ---------- filtering ----------
  function filteredPlayers() {
    const f = STATE.filters;
    const threshold = CONF_ORDER[f.minConfidence] ?? 0;
    return (STATE.data.players ?? []).filter((p) => {
      if (!f.positions.has(p.position)) return false;
      if ((p.offense_snaps ?? 0) < f.minSnaps) return false;
      if ((CONF_ORDER[p.confidence] ?? -1) < threshold) return false;
      if (f.matchedOnly && !["exact", "fuzzy_high"].includes(p.contract_match_quality)) return false;
      return true;
    });
  }

  // ---------- generic sortable table ----------
  function sortRows(rows, col, dir) {
    const sorted = [...rows];
    sorted.sort((a, b) => {
      const av = a[col];
      const bv = b[col];
      const aNull = av == null || Number.isNaN(av);
      const bNull = bv == null || Number.isNaN(bv);
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return dir === "asc" ? av - bv : bv - av;
      }
      return dir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return sorted;
  }

  function buildTable(container, rows, columns, sortState, onSortChange) {
    if (!rows || rows.length === 0) {
      container.innerHTML = '<div class="empty-state">No rows match the current filters.</div>';
      return;
    }
    const sorted = sortState ? sortRows(rows, sortState.col, sortState.dir) : rows;
    const head = `<thead><tr>${columns
      .map((c) => {
        const sortClass = sortState && sortState.col === c.key ? (sortState.dir === "asc" ? "sort-asc" : "sort-desc") : "";
        return `<th data-col="${c.key}" class="${sortClass}">${c.label}</th>`;
      })
      .join("")}</tr></thead>`;
    const body = `<tbody>${sorted
      .map(
        (r) =>
          `<tr>${columns
            .map((c) => {
              const v = c.render ? c.render(r[c.key], r) : r[c.key] ?? "-";
              return `<td class="${c.numeric ? "num" : ""}">${v}</td>`;
            })
            .join("")}</tr>`
      )
      .join("")}</tbody>`;
    container.innerHTML = `<table class="data">${head}${body}</table>`;

    if (sortState && onSortChange) {
      container.querySelectorAll("th").forEach((th) => {
        th.addEventListener("click", () => {
          const col = th.dataset.col;
          let dir = "desc";
          if (sortState.col === col && sortState.dir === "desc") dir = "asc";
          else if (sortState.col === col && sortState.dir === "asc") dir = "desc";
          onSortChange({ col, dir });
        });
      });
    }
  }

  // ---------- bargains / overpaid ----------
  const PLAYER_COLS = [
    { key: "player_name", label: "Player" },
    { key: "position", label: "Pos" },
    { key: "team", label: "Team" },
    { key: "age", label: "Age", numeric: true, render: (v) => fmtInt(v) },
    { key: "offense_snaps", label: "Snaps", numeric: true, render: (v) => fmtInt(v) },
    { key: "production", label: "Production", numeric: true, render: (v) => fmtNum(v, 1) },
    { key: "expected_production", label: "Expected", numeric: true, render: (v) => fmtNum(v, 1) },
    { key: "production_residual", label: "Residual", numeric: true, render: (v) => fmtNum(v, 1) },
    { key: "apy_millions", label: "APY", numeric: true, render: (v) => fmtMoney(v) },
    { key: "value_score", label: "Value", numeric: true, render: (v) => fmtNum(v, 2) },
    { key: "confidence", label: "Conf", render: (v) => badge(v) },
  ];

  function renderBargains() {
    const rows = filteredPlayers().filter((p) => p.value_score != null);
    const s = STATE.sort.bargains;
    buildTable(document.getElementById("bargains-table"), rows.slice(0, 200), PLAYER_COLS, s, (next) => {
      STATE.sort.bargains = next;
      renderBargains();
    });
  }

  function renderOverpaid() {
    const rows = filteredPlayers().filter((p) => p.value_score != null);
    const s = STATE.sort.overpaid;
    buildTable(document.getElementById("overpaid-table"), rows.slice(0, 200), PLAYER_COLS, s, (next) => {
      STATE.sort.overpaid = next;
      renderOverpaid();
    });
  }

  // ---------- lookup ----------
  let lookupActiveIdx = -1;
  function renderLookupSuggestions(query) {
    const box = document.getElementById("lookup-suggestions");
    if (!query) { box.classList.remove("open"); box.innerHTML = ""; lookupActiveIdx = -1; return; }
    const q = query.toLowerCase();
    const matches = (STATE.data.players ?? [])
      .filter((p) => p.player_name && p.player_name.toLowerCase().includes(q))
      .slice(0, 8);
    if (matches.length === 0) { box.classList.remove("open"); box.innerHTML = ""; return; }
    box.innerHTML = matches
      .map((p, i) => `<div class="suggestion${i === lookupActiveIdx ? " active" : ""}" data-id="${p.player_id}">${p.player_name} <span style="color:var(--muted)">- ${p.position} ${p.team ?? ""}</span></div>`)
      .join("");
    box.classList.add("open");
    box.querySelectorAll(".suggestion").forEach((el) => {
      el.addEventListener("click", () => {
        selectPlayer(el.dataset.id);
      });
    });
  }

  function selectPlayer(id) {
    const p = (STATE.data.players ?? []).find((x) => String(x.player_id) === String(id));
    const card = document.getElementById("lookup-card");
    const box = document.getElementById("lookup-suggestions");
    box.classList.remove("open");
    document.getElementById("lookup-input").value = p ? p.player_name : "";
    if (!p) {
      card.className = "player-card empty";
      card.innerHTML = "<p>No player selected.</p>";
      return;
    }
    card.className = "player-card";
    card.innerHTML = `
      <h3>${p.player_name} ${badge(p.confidence)}</h3>
      <div class="player-meta">${p.position} - ${p.team ?? "-"} - Age ${fmtInt(p.age)}</div>
      <div class="player-card-grid">
        <div class="player-card-stat"><div class="player-card-stat-label">Snaps</div><div class="player-card-stat-value">${fmtInt(p.offense_snaps)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Production</div><div class="player-card-stat-value">${fmtNum(p.production, 1)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Expected</div><div class="player-card-stat-value">${fmtNum(p.expected_production, 1)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Residual</div><div class="player-card-stat-value">${fmtNum(p.production_residual, 1)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">APY</div><div class="player-card-stat-value">${fmtMoney(p.apy_millions)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Value score</div><div class="player-card-stat-value">${fmtNum(p.value_score, 2)}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Match</div><div class="player-card-stat-value">${p.contract_match_quality ?? "-"}</div></div>
        <div class="player-card-stat"><div class="player-card-stat-label">Source</div><div class="player-card-stat-value">${p.contract_source ?? "-"}</div></div>
      </div>
    `;
  }

  // ---------- analytics ----------
  function renderAnalytics() {
    const h = STATE.data.run_history ?? [];
    const metricsBox = document.getElementById("analytics-metrics");
    if (h.length === 0) {
      metricsBox.innerHTML = '<div class="empty-state">No run history yet. The first daily-train job will seed it.</div>';
      ["chart-counts", "chart-r2", "chart-train"].forEach((id) => {
        const c = document.getElementById(id);
        if (c) c.parentElement.style.display = "none";
      });
      document.getElementById("history-table").innerHTML = "";
      document.getElementById("movers-up").innerHTML = "";
      document.getElementById("movers-down").innerHTML = "";
      return;
    }
    const last = h[h.length - 1];
    metricsBox.innerHTML = [
      ["Runs recorded", fmtInt(h.length)],
      ["Most recent run", fmtDate(last.run_date)],
      ["Eligible players", fmtInt(last.n_eligible)],
      ["Matched contracts", fmtInt(last.n_with_contract)],
    ]
      .map(([k, v]) => `<div class="metric"><span class="metric-label">${k}</span><span class="metric-value">${v}</span></div>`)
      .join("");

    drawCountsChart(h);
    drawR2Chart(h);
    drawTrainChart(h);

    // Top movers
    const movers = STATE.data.top_movers ?? [];
    const moverCols = [
      { key: "player_name", label: "Player" },
      { key: "position", label: "Pos" },
      { key: "team", label: "Team" },
      { key: "value_score_prev", label: "Prev", numeric: true, render: (v) => fmtNum(v, 2) },
      { key: "value_score", label: "Now", numeric: true, render: (v) => fmtNum(v, 2) },
      { key: "value_score_delta", label: "Δ", numeric: true, render: (v) => fmtNum(v, 2) },
      { key: "confidence", label: "Conf", render: (v) => badge(v) },
    ];
    if (movers.length === 0) {
      document.getElementById("movers-up").innerHTML = '<div class="empty-state">Need ≥2 snapshots.</div>';
      document.getElementById("movers-down").innerHTML = '<div class="empty-state">Need ≥2 snapshots.</div>';
    } else {
      const sortedDesc = sortRows(movers, "value_score_delta", "desc").slice(0, 10);
      const sortedAsc = sortRows(movers, "value_score_delta", "asc").slice(0, 10);
      buildTable(document.getElementById("movers-up"), sortedDesc, moverCols);
      buildTable(document.getElementById("movers-down"), sortedAsc, moverCols);
    }

    // Full run history
    const historyCols = [
      { key: "run_date", label: "Date", render: (v) => fmtDate(v) },
      { key: "current_season", label: "Season", numeric: true },
      { key: "n_current", label: "Players", numeric: true, render: (v) => fmtInt(v) },
      { key: "n_eligible", label: "Eligible", numeric: true, render: (v) => fmtInt(v) },
      { key: "n_low_sample", label: "Low sample", numeric: true, render: (v) => fmtInt(v) },
      { key: "n_no_match", label: "No match", numeric: true, render: (v) => fmtInt(v) },
      { key: "qb_r2", label: "QB R²", numeric: true, render: (v) => fmtNum(v, 3) },
      { key: "rb_r2", label: "RB R²", numeric: true, render: (v) => fmtNum(v, 3) },
      { key: "wr_r2", label: "WR R²", numeric: true, render: (v) => fmtNum(v, 3) },
      { key: "te_r2", label: "TE R²", numeric: true, render: (v) => fmtNum(v, 3) },
    ];
    buildTable(document.getElementById("history-table"), [...h].reverse(), historyCols);
  }

  function destroyChart(id) {
    if (STATE.charts[id]) {
      STATE.charts[id].destroy();
      delete STATE.charts[id];
    }
  }

  function drawCountsChart(history) {
    const canvas = document.getElementById("chart-counts");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("counts");
    const labels = history.map((r) => fmtDate(r.run_date));
    STATE.charts.counts = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Eligible", data: history.map((r) => r.n_eligible), borderColor: "#0a7f3f", tension: 0.2 },
          { label: "Low sample", data: history.map((r) => r.n_low_sample), borderColor: "#b8860b", tension: 0.2 },
          { label: "No contract match", data: history.map((r) => r.n_no_match), borderColor: "#b03030", tension: 0.2 },
        ],
      },
      options: lineOptions(),
    });
  }

  function drawR2Chart(history) {
    const canvas = document.getElementById("chart-r2");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("r2");
    const labels = history.map((r) => fmtDate(r.run_date));
    const palette = { qb: "#2563eb", rb: "#dc2626", wr: "#0a7f3f", te: "#9333ea" };
    STATE.charts.r2 = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: ["qb", "rb", "wr", "te"].map((p) => ({
          label: p.toUpperCase(),
          data: history.map((r) => r[`${p}_r2`]),
          borderColor: palette[p],
          tension: 0.2,
        })),
      },
      options: lineOptions({ yMin: 0, yMax: 1 }),
    });
  }

  function drawTrainChart(history) {
    const canvas = document.getElementById("chart-train");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("train");
    const labels = history.map((r) => fmtDate(r.run_date));
    const palette = { qb: "#2563eb", rb: "#dc2626", wr: "#0a7f3f", te: "#9333ea" };
    STATE.charts.train = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: ["qb", "rb", "wr", "te"].map((p) => ({
          label: p.toUpperCase(),
          data: history.map((r) => r[`${p}_n_train`]),
          borderColor: palette[p],
          tension: 0.2,
        })),
      },
      options: lineOptions(),
    });
  }

  function lineOptions(extra = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { beginAtZero: false, min: extra.yMin, max: extra.yMax, ticks: { font: { size: 10 } } },
      },
    };
  }

  // ---------- fantasy draft ----------
  const FANTASY_FILTERS = { onlyPlayed: false, season: "all", position: "all" };
  let fantasySeasonsPopulated = false;

  function fantasyRows() {
    const all = STATE.data.fantasy?.players ?? [];
    return all.filter((p) => {
      if (FANTASY_FILTERS.onlyPlayed && !p.has_stats) return false;
      if (FANTASY_FILTERS.season !== "all" && String(p.season) !== FANTASY_FILTERS.season) return false;
      if (FANTASY_FILTERS.position !== "all" && p.position !== FANTASY_FILTERS.position) return false;
      return true;
    });
  }

  function populateFantasySeasons() {
    if (fantasySeasonsPopulated) return;
    const seasons = [...new Set((STATE.data.fantasy?.players ?? []).map((p) => p.season))]
      .filter((s) => s != null)
      .sort((a, b) => b - a);
    const sel = document.getElementById("fantasy-season");
    for (const s of seasons) {
      const opt = document.createElement("option");
      opt.value = String(s);
      opt.textContent = String(Math.round(s));
      sel.appendChild(opt);
    }
    fantasySeasonsPopulated = true;
  }

  function renderFantasy() {
    populateFantasySeasons();
    drawFantasyRoundChart();
    drawFantasyPositionChart();

    const rows = fantasyRows();
    const scored = rows.filter((p) => p.surplus != null);
    const steals = sortRows(scored, "surplus", "desc").slice(0, 15);
    const busts = sortRows(scored, "surplus", "asc").slice(0, 15);

    const moverCols = [
      { key: "season", label: "Yr", numeric: true, render: (v) => (v == null ? "-" : String(Math.round(v))) },
      { key: "adp", label: "ADP", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "player_name", label: "Player" },
      { key: "position", label: "Pos" },
      { key: "team", label: "Team" },
      { key: "realized_points", label: "Realized", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "expected_points", label: "Expected", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "surplus", label: "Surplus", numeric: true, render: (v) => fmtNum(v, 1) },
    ];
    buildTable(document.getElementById("fantasy-steals"), steals, moverCols);
    buildTable(document.getElementById("fantasy-busts"), busts, moverCols);

    const fullCols = [
      { key: "season", label: "Yr", numeric: true, render: (v) => (v == null ? "-" : String(Math.round(v))) },
      { key: "adp", label: "ADP", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "adp_round", label: "Rd", numeric: true, render: (v) => fmtInt(v) },
      { key: "player_name", label: "Player" },
      { key: "position", label: "Pos" },
      { key: "team", label: "Team" },
      { key: "times_drafted", label: "Drafts", numeric: true, render: (v) => fmtInt(v) },
      { key: "realized_points", label: "Realized", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "expected_points", label: "Expected", numeric: true, render: (v) => fmtNum(v, 1) },
      { key: "surplus", label: "Surplus", numeric: true, render: (v) => fmtNum(v, 1) },
    ];
    const sorted = sortRows(rows, "surplus", "desc");
    buildTable(document.getElementById("fantasy-table"), sorted, fullCols);
  }

  function drawFantasyRoundChart() {
    const canvas = document.getElementById("chart-fantasy-round");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("fantasy-round");

    const rows = fantasyRows();
    const byRound = {};
    for (const p of rows) {
      if (p.surplus == null) continue;
      (byRound[p.adp_round] ??= []).push(p.surplus);
    }
    const rounds = Object.keys(byRound).map(Number).sort((a, b) => a - b);
    const medians = rounds.map((r) => {
      const arr = byRound[r].slice().sort((a, b) => a - b);
      const mid = Math.floor(arr.length / 2);
      return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    });

    STATE.charts["fantasy-round"] = new Chart(canvas, {
      type: "bar",
      data: {
        labels: rounds.map((r) => `R${r}`),
        datasets: [{
          label: "Median surplus",
          data: medians,
          backgroundColor: medians.map((v) => (v >= 0 ? "#0a7f3f" : "#b03030")),
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  function drawFantasyPositionChart() {
    const canvas = document.getElementById("chart-fantasy-position");
    if (!canvas || typeof Chart === "undefined") return;
    destroyChart("fantasy-position");

    const rows = fantasyRows();
    const byPos = {};
    for (const p of rows) {
      if (p.surplus == null) continue;
      (byPos[p.position] ??= []).push(p.surplus);
    }
    const positions = ["QB", "RB", "WR", "TE"].filter((p) => byPos[p]?.length);
    const medians = positions.map((pos) => {
      const arr = byPos[pos].slice().sort((a, b) => a - b);
      const mid = Math.floor(arr.length / 2);
      return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    });

    STATE.charts["fantasy-position"] = new Chart(canvas, {
      type: "bar",
      data: {
        labels: positions,
        datasets: [{
          label: "Median surplus",
          data: medians,
          backgroundColor: medians.map((v) => (v >= 0 ? "#0a7f3f" : "#b03030")),
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  function setupFantasyFilters() {
    document.getElementById("fantasy-only-played").addEventListener("change", (e) => {
      FANTASY_FILTERS.onlyPlayed = e.target.checked;
      renderFantasy();
    });
    document.getElementById("fantasy-season").addEventListener("change", (e) => {
      FANTASY_FILTERS.season = e.target.value;
      renderFantasy();
    });
    document.getElementById("fantasy-position").addEventListener("change", (e) => {
      FANTASY_FILTERS.position = e.target.value;
      renderFantasy();
    });
  }

  // ---------- diagnostic ----------
  function renderDiagnostic() {
    const breakdown = STATE.data.match_quality_breakdown ?? [];
    buildTable(document.getElementById("breakdown-table"), breakdown, [
      { key: "position", label: "Position" },
      { key: "contract_match_quality", label: "Match quality" },
      { key: "count", label: "Count", numeric: true, render: (v) => fmtInt(v) },
    ]);

    const excluded = STATE.data.excluded_players ?? [];
    document.getElementById("excluded-heading").textContent = `Excluded players (${excluded.length})`;
    buildTable(document.getElementById("excluded-table"), excluded, [
      { key: "season", label: "Season", numeric: true },
      { key: "player_name", label: "Player" },
      { key: "position", label: "Pos" },
      { key: "team", label: "Team" },
      { key: "offense_snaps", label: "Snaps", numeric: true, render: (v) => fmtInt(v) },
      { key: "contract_match_quality", label: "Match" },
      { key: "contract_match_score", label: "Score", numeric: true, render: (v) => fmtInt(v) },
      { key: "exclusion_reason", label: "Reason" },
    ]);
  }

  // ---------- wiring ----------
  function rerenderAll() {
    renderBargains();
    renderOverpaid();
  }

  function setupFilters() {
    document.querySelectorAll('#filter-position input[type="checkbox"]').forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const v = e.target.value;
        if (e.target.checked) STATE.filters.positions.add(v);
        else STATE.filters.positions.delete(v);
        rerenderAll();
      });
    });
    const snaps = document.getElementById("filter-snaps");
    snaps.addEventListener("input", (e) => {
      STATE.filters.minSnaps = Number(e.target.value);
      document.getElementById("filter-snaps-value").textContent = e.target.value;
      rerenderAll();
    });
    document.getElementById("filter-confidence").addEventListener("change", (e) => {
      STATE.filters.minConfidence = e.target.value;
      rerenderAll();
    });
    document.getElementById("filter-matched").addEventListener("change", (e) => {
      STATE.filters.matchedOnly = e.target.checked;
      rerenderAll();
    });
  }

  function setupTabs() {
    document.querySelectorAll(".tab").forEach((t) => {
      t.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
        t.classList.add("active");
        document.getElementById(`panel-${t.dataset.tab}`).classList.add("active");
        if (t.dataset.tab === "analytics") renderAnalytics();
        if (t.dataset.tab === "diagnostic") renderDiagnostic();
        if (t.dataset.tab === "fantasy") renderFantasy();
      });
    });
  }

  function setupLookup() {
    const input = document.getElementById("lookup-input");
    input.addEventListener("input", (e) => renderLookupSuggestions(e.target.value));
    input.addEventListener("blur", () => setTimeout(() => document.getElementById("lookup-suggestions").classList.remove("open"), 150));
    input.addEventListener("focus", (e) => renderLookupSuggestions(e.target.value));
  }

  async function init() {
    try {
      STATE.data = await loadData();
    } catch (e) {
      document.querySelector("main").innerHTML = `<div class="empty-state">Could not load data.json: ${e.message}</div>`;
      return;
    }
    renderHeader();
    setupFilters();
    setupTabs();
    setupLookup();
    setupFantasyFilters();
    rerenderAll();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
