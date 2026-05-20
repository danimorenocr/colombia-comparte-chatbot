import json

from fastapi.responses import HTMLResponse


def _first_present(item: dict, keys: list[str], default=None):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_number(value, default=0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value):
    number = _as_number(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _table_rows(items: list[dict], kind: str) -> str:
    if kind == "leads":
        return "".join(
            f"<tr><td>{_first_present(item, ['country', 'pais'], 'N/D')}</td><td>{_format_number(_first_present(item, ['total_sesiones', 'total_leads', 'sesiones', 'leads'], 0))}</td><td>{_format_number(_first_present(item, ['sesiones_con_lead', 'leads_conversacion', 'leads'], 0))}</td><td>{_format_number(_first_present(item, ['pct_leads', 'porcentaje_leads'], 0))}%</td></tr>"
            for item in items
        ) or "<tr><td colspan='4'>Sin datos</td></tr>"

    if kind == "daily":
        return "".join(
        f"<tr><td>{_first_present(item, ['dia', 'fecha', 'day', 'date'], 'N/D')}</td><td>{_format_number(_first_present(item, ['total_mensajes', 'total_sesiones', 'mensajes', 'cantidad', 'total'], 0))}</td><td>{_format_number(_first_present(item, ['sesiones_unicas', 'sesiones_con_lead', 'leads', 'leads_conversacion'], 0))}</td></tr>"
            for item in items
        ) or "<tr><td colspan='4'>Sin datos</td></tr>"

    return "".join(
        f"<tr><td>{_first_present(item, ['pregunta', 'faq', 'query'], 'N/D')}</td><td>{_format_number(_first_present(item, ['veces', 'cantidad', 'total'], 0))}</td><td>{_first_present(item, ['ultima_vez', 'last_seen', 'updated_at'], 'N/D')}</td></tr>"
        for item in items
    ) or "<tr><td colspan='3'>Sin datos</td></tr>"


def render_analytics_dashboard(leads: list[dict], diaria: list[dict], faqs: list[dict]) -> HTMLResponse:
    total_leads = sum(
        _as_number(_first_present(item, ["sesiones_con_lead", "leads_conversacion", "leads"], 0))
        for item in leads
    )
    total_mensajes = sum(_as_number(_first_present(item, ["total_mensajes", "mensajes", "cantidad", "total"], 0)) for item in diaria)
    total_sesiones_unicas = sum(_as_number(_first_present(item, ["sesiones_unicas", "sesiones", "total_sesiones"], 0)) for item in diaria)
    total_preguntas = sum(_as_number(_first_present(item, ["veces", "cantidad", "total"], 0)) for item in faqs)

    leads_labels = [_first_present(item, ["country", "pais"], "N/D") for item in leads]
    leads_values = [_as_number(_first_present(item, ["sesiones_con_lead", "leads_conversacion", "leads"], 0)) for item in leads]
    daily_labels = [_first_present(item, ["dia", "fecha", "day", "date"], "N/D") for item in diaria]
    daily_messages = [_as_number(_first_present(item, ["total_mensajes", "mensajes", "cantidad", "total"], 0)) for item in diaria]
    daily_sessions = [_as_number(_first_present(item, ["sesiones_unicas", "total_sesiones", "sesiones", "leads_conversacion"], 0)) for item in diaria]

    leads_rows = _table_rows(leads, "leads")
    diaria_rows = _table_rows(diaria, "daily")
    faqs_rows = _table_rows(faqs, "faqs")

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analytics - Colombia Comparte</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --accent-2: #38bdf8;
      --border: rgba(148, 163, 184, 0.2);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top, #1e293b, var(--bg) 55%); color: var(--text); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 24px; }}
    .title h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 42px); }}
    .title p {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: 0 16px 50px rgba(0,0,0,.24); }}
    .metric {{ font-size: 34px; font-weight: 800; margin: 6px 0 0; }}
    .label {{ color: var(--muted); font-size: 14px; text-transform: uppercase; letter-spacing: .08em; }}
    .charts {{ display: grid; grid-template-columns: 1.3fr .9fr; gap: 16px; margin-bottom: 16px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: #bfdbfe; font-weight: 600; }}
    .footer {{ margin-top: 18px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{ .grid, .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="title">
        <h1>Analytics de Colombia Comparte</h1>
        <p>Dashboard rápido consumiendo Supabase desde el backend.</p>
      </div>
      <div class="label">/analytics/dashboard</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Leads por país</div><div class="metric">{total_leads}</div></div>
      <div class="card"><div class="label">Mensajes totales</div><div class="metric">{_format_number(total_mensajes)}</div></div>
      <div class="card"><div class="label">Sesiones únicas</div><div class="metric">{_format_number(total_sesiones_unicas)}</div></div>
      <div class="card"><div class="label">FAQ destacadas</div><div class="metric">{total_preguntas}</div></div>
    </div>

    <div class="charts">
      <div class="card panel">
        <h2>Leads por país</h2>
        <p style="margin-top:-4px;color:var(--muted);font-size:13px;">Sesiones totales, sesiones con lead y porcentaje por país.</p>
        <canvas id="leadsChart" height="140"></canvas>
      </div>
      <div class="card panel">
        <h2>Actividad diaria</h2>
        <p style="margin-top:-4px;color:var(--muted);font-size:13px;">Mensajes por día y sesiones únicas por día.</p>
        <canvas id="dailyChart" height="140"></canvas>
      </div>
    </div>

    <div class="charts">
      <div class="card panel">
        <h2>Leads por país</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>País</th><th>Sesiones</th><th>Leads</th><th>% Leads</th></tr></thead>
            <tbody>{leads_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="card panel">
        <h2>Actividad diaria</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Día</th><th>Total mensajes</th><th>Sesiones únicas</th></tr></thead>
            <tbody>{diaria_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card panel">
      <h2>Preguntas frecuentes</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Pregunta</th><th>Veces</th><th>Última vez</th></tr></thead>
          <tbody>{faqs_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="footer">Si quieres, luego lo conectamos a filtros por país, rango de fechas o exportación CSV.</div>
  </div>

  <script>
    const leadsData = {json.dumps([{"label": leads_labels[i], "value": leads_values[i]} for i in range(len(leads_labels))], ensure_ascii=False)};
    const dailyData = {json.dumps([{ "label": daily_labels[i], "messages": daily_messages[i], "sessions": daily_sessions[i]} for i in range(len(daily_labels))], ensure_ascii=False)};

    new Chart(document.getElementById('leadsChart'), {{
      type: 'bar',
      data: {{
        labels: leadsData.map(item => item.label || 'N/D'),
        datasets: [{{
          label: 'Leads',
          data: leadsData.map(item => item.value ?? 0),
          backgroundColor: '#22c55e',
          borderRadius: 10,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#cbd5e1' }} }}, x: {{ ticks: {{ color: '#cbd5e1' }} }} }}
      }}
    }});

    new Chart(document.getElementById('dailyChart'), {{
      type: 'bar',
      data: {{
        labels: dailyData.map(item => item.label || 'N/D'),
        datasets: [
          {{
            label: 'Mensajes',
            data: dailyData.map(item => item.messages ?? 0),
            backgroundColor: '#38bdf8',
            borderRadius: 10,
          }},
          {{
            label: 'Sesiones únicas',
            data: dailyData.map(item => item.sessions ?? 0),
            backgroundColor: '#22c55e',
            borderRadius: 10,
          }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#e5e7eb' }} }} }},
        scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#cbd5e1' }} }}, x: {{ ticks: {{ color: '#cbd5e1' }} }} }}
      }}
    }});
  </script>
</body>
</html>"""

    return HTMLResponse(content=html)
