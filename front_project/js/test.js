const API_URL = 'http://127.0.0.1:8000/diagnosticar';

// Mapeo de síntomas a grupos visuales
const GROUPS = {
  1: ['s1', 's2', 's3', 's4', 's5', 's6'],
  2: ['s7', 's8', 's9', 's10'],
  3: ['s11', 's12', 's13', 's14', 's15'],
  4: ['s16', 's17', 's18', 's19', 's20', 's21', 's22'],
  5: ['s23', 's24', 's25']
};

document.addEventListener('DOMContentLoaded', () => {
  renderCuestionario();
  setupCheckboxListeners();
});

// Renderizar dinámicamente los checkboxes agrupados
function renderCuestionario() {
  Object.keys(GROUPS).forEach(groupId => {
    const container = document.getElementById(`group-${groupId}`);
    if (!container) return;

    container.innerHTML = '';
    const symIds = GROUPS[groupId];
    
    symIds.forEach(id => {
      const sym = SYMPTOMS[id];
      if (!sym) return;

      const card = document.createElement('div');
      card.className = 'checkbox-card';
      card.setAttribute('data-id', id);
      card.innerHTML = `
        <input type="checkbox" id="chk-${id}" value="${id}">
        <div class="checkbox-label">
          <span class="sym-name">${sym.label} <small style="color: var(--color-primary-light); font-weight:700;">(${id})</small></span>
          <span class="sym-desc">${sym.desc}</span>
        </div>
      `;

      // Clic en la tarjeta activa el checkbox
      card.addEventListener('click', (e) => {
        if (e.target.tagName !== 'INPUT') {
          const chk = card.querySelector('input[type="checkbox"]');
          chk.checked = !chk.checked;
          chk.dispatchEvent(new Event('change'));
        }
      });

      container.appendChild(card);
    });
  });
}

// Configurar los listeners de eventos para contar seleccionados y habilitar botón
function setupCheckboxListeners() {
  const checkboxes = document.querySelectorAll('.checkbox-card input[type="checkbox"]');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const counterLabel = document.getElementById('counterLabel');

  checkboxes.forEach(chk => {
    chk.addEventListener('change', () => {
      const card = chk.closest('.checkbox-card');
      if (chk.checked) {
        card.classList.add('selected');
      } else {
        card.classList.remove('selected');
      }

      // Contar seleccionados
      const selected = getSelectedSymptomIds();
      const count = selected.length;
      counterLabel.textContent = `${count} síntoma${count !== 1 ? 's' : ''} seleccionado${count !== 1 ? 's' : ''}`;
      
      // Habilitar o deshabilitar botón
      btnAnalyze.disabled = count === 0;
    });
  });
}

// Obtener los ids seleccionados
function getSelectedSymptomIds() {
  const checked = document.querySelectorAll('.checkbox-card input[type="checkbox"]:checked');
  return Array.from(checked).map(chk => chk.value);
}

// Limpiar todas las selecciones
window.clearSelection = function() {
  const checkboxes = document.querySelectorAll('.checkbox-card input[type="checkbox"]');
  checkboxes.forEach(chk => {
    chk.checked = false;
    chk.dispatchEvent(new Event('change'));
  });
  resetTest();
};

// Reiniciar test para volver a jugar
window.resetTest = function() {
  document.getElementById('resultsSection').classList.remove('visible');
  document.getElementById('emergencyBanner').style.display = 'none';
  document.getElementById('noDiagnosisMsg').style.display = 'none';
  document.getElementById('diagnosesResultsList').innerHTML = '';
  
  // Hacer scroll arriba suavemente
  window.scrollTo({ top: 0, behavior: 'smooth' });
};

// Enviar síntomas a FastAPI
async function analyzeSymptoms() {
  const selectedSymptoms = getSelectedSymptomIds();
  if (selectedSymptoms.length === 0) return;

  const spinner = document.getElementById('loadingSpinner');
  const resultsSection = document.getElementById('resultsSection');
  const btnAnalyze = document.getElementById('btnAnalyze');

  // Limpiar resultados anteriores
  resultsSection.classList.remove('visible');
  document.getElementById('emergencyBanner').style.display = 'none';
  document.getElementById('noDiagnosisMsg').style.display = 'none';
  document.getElementById('diagnosesResultsList').innerHTML = '';

  // Mostrar cargando
  spinner.classList.add('visible');
  btnAnalyze.disabled = true;

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sintomas: selectedSymptoms })
    });

    if (!response.ok) {
      throw new Error('Error al conectar con la API de diagnóstico.');
    }

    const data = await response.json();
    displayResults(data.diagnosticos, selectedSymptoms);
  } catch (error) {
    alert('Error: No se pudo conectar con el servidor de diagnóstico. Asegúrate de tener FastAPI activo en localhost:8000.\n' + error.message);
  } finally {
    spinner.classList.remove('visible');
    btnAnalyze.disabled = false;
  }
}

// Mostrar los resultados en la interfaz
function displayResults(diagnosticos, sintomasSeleccionados) {
  const resultsSection = document.getElementById('resultsSection');
  const container = document.getElementById('diagnosesResultsList');
  const noDiagMsg = document.getElementById('noDiagnosisMsg');
  const emergencyBanner = document.getElementById('emergencyBanner');
  
  container.innerHTML = '';
  resultsSection.classList.add('visible');

  // 1. Mostrar lista de síntomas analizados
  const names = sintomasSeleccionados.map(id => SYMPTOMS[id] ? SYMPTOMS[id].label : id);
  document.getElementById('symptomsAnalyzedList').textContent = names.join(', ');

  if (!diagnosticos || diagnosticos.length === 0) {
    noDiagMsg.style.display = 'block';
    // Hacer scroll al mensaje
    setTimeout(() => {
      noDiagMsg.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 150);
    return;
  }

  // 2. Verificar si hay diagnósticos de emergencia crítica (cad, ehh)
  const isEmergency = diagnosticos.some(d => d === 'cad' || d === 'ehh');
  if (isEmergency) {
    emergencyBanner.style.display = 'flex';
  }

  // Ordenar para mostrar emergencias críticas primero en la lista
  const sortedDiagnoses = [...diagnosticos].sort((a, b) => {
    const severityA = DIAGNOSES[a] ? DIAGNOSES[a].severity : '';
    const severityB = DIAGNOSES[b] ? DIAGNOSES[b].severity : '';
    if (severityA === 'critica' && severityB !== 'critica') return -1;
    if (severityA !== 'critica' && severityB === 'critica') return 1;
    return 0;
  });

  // 3. Crear tarjetas de diagnóstico
  sortedDiagnoses.forEach(diagId => {
    const diag = DIAGNOSES[diagId];
    if (!diag) return;

    const card = document.createElement('div');
    card.className = 'result-diagnosis-card';
    card.style.borderLeftColor = diag.color;

    let severityClass = 'badge-primary';
    if (diag.severity === 'moderada') severityClass = 'badge-warning';
    else if (diag.severity === 'alta') severityClass = 'badge-danger';
    else if (diag.severity === 'critica') severityClass = 'badge-danger';

    // Formatear fuentes bibliográficas
    const sourcesHTML = diag.sources.map(src => 
      `<a href="${src.url}" target="_blank" style="color: var(--color-primary-light); text-decoration: none; font-size: 11.5px; font-weight: 500;">${src.label} ↗</a>`
    ).join(' | ');

    card.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
        <h3 style="font-size: 20px; font-family: var(--font-body); color: var(--color-primary); font-weight: 700; margin: 0;">${diag.name}</h3>
        <div>
          <span class="badge badge-primary" style="margin-right: 6px;">${diag.type}</span>
          <span class="badge ${severityClass}">${diag.severity}</span>
        </div>
      </div>
      <p style="font-size: 13.5px; margin-bottom: 14px; color: var(--color-text);">${diag.description}</p>
      
      <div style="background-color: #F8FAFC; border: 1px solid var(--color-border); border-radius: var(--radius-sm); padding: 16px; margin-bottom: 14px;">
        <h4 style="font-size: 11px; text-transform: uppercase; color: var(--color-text-muted); margin-bottom: 4px; font-weight: 700;">📋 Recomendación Médica:</h4>
        <p style="font-size: 13px; font-weight: 500; color: #1E3A5F; line-height: 1.5; margin: 0;">${diag.recommendation}</p>
      </div>

      <div style="border-top: 1px solid var(--color-border); padding-top: 10px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span style="font-size: 11px; text-transform: uppercase; color: var(--color-text-muted); font-weight: 700;">Fuentes:</span>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">${sourcesHTML}</div>
      </div>
    `;
    container.appendChild(card);
  });

  // Hacer scroll a los resultados
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 150);
}
