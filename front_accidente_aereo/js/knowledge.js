document.addEventListener('DOMContentLoaded', () => {
  renderSymptoms();
  renderDiagnosesRules();
});

// Renderizar el catálogo de síntomas
function renderSymptoms() {
  const grid = document.getElementById('symptomsGrid');
  if (!grid) return;

  grid.innerHTML = '';
  Object.values(SYMPTOMS).forEach(sym => {
    const card = document.createElement('div');
    card.className = 'symptom-card';
    card.innerHTML = `
      <span class="sym-code">${sym.id}</span>
      <h3>${sym.label}</h3>
      <p>${sym.desc}</p>
    `;
    grid.appendChild(card);
  });
}

// Renderizar las reglas asociadas a cada diagnóstico
function renderDiagnosesRules() {
  const container = document.getElementById('diagnosesList');
  if (!container) return;

  container.innerHTML = '';
  Object.values(DIAGNOSES).forEach(diag => {
    const section = document.createElement('div');
    section.style.borderLeft = `5px solid ${diag.color}`;
    section.style.paddingLeft = '20px';
    section.style.marginBottom = '28px';

    // Severidad Badge
    let severityClass = 'badge-primary';
    if (diag.severity === 'moderada') severityClass = 'badge-warning';
    else if (diag.severity === 'alta') severityClass = 'badge-danger';
    else if (diag.severity === 'critica') severityClass = 'badge-danger';

    // Crear lista de reglas
    let rulesHTML = '';
    if (diag.rules && diag.rules.length > 0) {
      diag.rules.forEach((rule, index) => {
        const conditionTags = rule.map(cond => getConditionHTML(cond));
        rulesHTML += `
          <div class="rule-row">
            <span class="badge badge-primary">Regla ${index + 1}</span>
            ${conditionTags.join(' <span class="rule-operator">+</span> ')}
            <span class="rule-arrow">➔</span>
            <span class="badge" style="background-color: ${diag.color}15; color: ${diag.color}; font-weight: 700;">${diag.name}</span>
          </div>
        `;
      });
    } else {
      rulesHTML = '<p style="font-size: 13px; color: var(--color-text-muted);">Sin reglas lógicas explícitas definidas.</p>';
    }

    section.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
        <h3 style="font-size: 20px; font-family: var(--font-body); color: var(--color-primary); font-weight: 700;">${diag.name}</h3>
        <div>
          <span class="badge badge-primary" style="margin-right: 6px;">${diag.type}</span>
          <span class="badge ${severityClass}">${diag.severity}</span>
        </div>
      </div>
      <p style="font-size: 14px; color: var(--color-text); margin-bottom: 12px;">${diag.description}</p>
      
      <div style="margin-top: 10px;">
        <h4 style="font-size: 12px; text-transform: uppercase; color: var(--color-text-muted); margin-bottom: 8px; font-weight: 700;">Reglas de disparo lógicas:</h4>
        ${rulesHTML}
      </div>
    `;

    container.appendChild(section);
  });
}

// Helper para convertir condiciones de regla en etiquetas HTML
function getConditionHTML(cond) {
  if (cond.includes('/')) {
    const parts = cond.split('/');
    const tags = parts.map(p => {
      const sym = SYMPTOMS[p];
      return `<span class="symptom-tag" title="${sym ? sym.desc : ''}">${sym ? sym.label : p}</span>`;
    });
    return `(${tags.join(' <span class="rule-operator">o</span> ')})`;
  } else if (cond.startsWith('no ')) {
    const p = cond.replace('no ', '');
    const sym = SYMPTOMS[p];
    return `<span class="rule-operator">NO</span> <span class="symptom-tag" style="background-color: #FEE2E2; color: #991B1B;" title="${sym ? sym.desc : ''}">${sym ? sym.label : p}</span>`;
  } else {
    const sym = SYMPTOMS[cond];
    return `<span class="symptom-tag" title="${sym ? sym.desc : ''}">${sym ? sym.label : cond}</span>`;
  }
}
