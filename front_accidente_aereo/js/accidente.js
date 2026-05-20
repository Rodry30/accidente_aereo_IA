const API_PREDICT = 'http://127.0.0.1:8000/predecir';

function getVal(id) {
  const el = document.getElementById(id);
  return el ? el.value : '';
}

// Enviar datos del pasajero a la API del modelo de Machine Learning
async function predictSurvival() {
  ocultarResultado();
  ocultarError();

  const campos = ['sex', 'age', 'clase', 'fare', 'deck', 'embark_town', 'n_siblings_spouses', 'parch', 'alone'];
  
  // Validar que todos los campos estén llenos
  for (const c of campos) {
    const val = getVal(c);
    if (!val && val !== '0') {
      mostrarError('Por favor, completa todos los campos del formulario antes de realizar la predicción.');
      return;
    }
  }

  // Armar cuerpo de la petición
  const body = {
    sex: getVal('sex'),
    age: parseFloat(getVal('age')),
    n_siblings_spouses: parseInt(getVal('n_siblings_spouses')),
    parch: parseInt(getVal('parch')),
    fare: parseFloat(getVal('fare')),
    clase: getVal('clase'),
    deck: getVal('deck'),
    embark_town: getVal('embark_town'),
    alone: getVal('alone')
  };

  const spinner = document.getElementById('loadingSpinner');
  const btnPredict = document.getElementById('btnPredict');

  // Mostrar spinner
  spinner.classList.add('visible');
  btnPredict.disabled = true;

  try {
    const res = await fetch(API_PREDICT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      throw new Error('Error al conectar con la API de predicción (Código de respuesta no OK).');
    }

    const data = await res.json();
    mostrarResultado(data);
  } catch (error) {
    mostrarError('No se pudo establecer conexión con el servidor. Verifica que FastAPI esté corriendo en localhost:8000.\nDetalle: ' + error.message);
  } finally {
    spinner.classList.remove('visible');
    btnPredict.disabled = false;
  }
}

// Renderizar el resultado de supervivencia
function mostrarResultado(data) {
  const alertEl = document.getElementById('resultAlert');
  const porcentaje = (data.probabilidad * 100).toFixed(1);

  alertEl.className = `result-alert visible ${data.sobrevive ? 'survive' : 'die'}`;
  document.getElementById('resIcon').textContent = data.sobrevive ? '✈️' : '💀';
  document.getElementById('resTitle').textContent = data.sobrevive ? 'SOBREVIVE ✅' : 'NO SOBREVIVE ❌';
  document.getElementById('resProb').textContent = `${porcentaje}%`;

  // Animar la barra de carga de probabilidad
  setTimeout(() => {
    document.getElementById('probBar').style.width = `${porcentaje}%`;
  }, 100);

  // Scroll suave al resultado
  setTimeout(() => {
    alertEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 150);
}

function ocultarResultado() {
  const alertEl = document.getElementById('resultAlert');
  alertEl.className = 'result-alert';
  document.getElementById('probBar').style.width = '0%';
}

function mostrarError(msg) {
  const errorEl = document.getElementById('errorAlert');
  errorEl.textContent = msg;
  errorEl.classList.add('visible');
}

function ocultarError() {
  document.getElementById('errorAlert').classList.remove('visible');
}

// Limpiar formulario completo
window.clearForm = function() {
  const campos = ['sex', 'age', 'clase', 'fare', 'deck', 'embark_town', 'n_siblings_spouses', 'parch', 'alone'];
  campos.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  ocultarResultado();
  ocultarError();
  
  // Volver arriba suave
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
