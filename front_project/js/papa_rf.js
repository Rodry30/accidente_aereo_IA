const API_URL = 'http://127.0.0.1:8000/predecir_papa?modelo_tipo=rf';

// Elementos de la interfaz
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imgPreview = document.getElementById('imgPreview');
const btnClassify = document.getElementById('btnClassify');
const btnClear = document.getElementById('btnClear');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorAlert = document.getElementById('errorAlert');
const resultCard = document.getElementById('resultCard');

// Elementos de resultados
const resVariety = document.getElementById('resVariety');
const resConfianza = document.getElementById('resConfianza');
const confidenceBar = document.getElementById('confidenceBar');
const imgExample = document.getElementById('imgExample');
const top5Container = document.getElementById('top5Container');

let selectedFile = null;

// Eventos de Drag & Drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

dropZone.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

// Procesar el archivo seleccionado
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Por favor, selecciona una imagen válida (.jpg, .jpeg o .png).');
    return;
  }
  
  selectedFile = file;
  ocultarError();
  ocultarResultado();

  // Mostrar previsualización
  const reader = new FileReader();
  reader.onload = (e) => {
    imgPreview.src = e.target.result;
    dropZone.style.display = 'none';
    previewContainer.style.display = 'block';
    btnClassify.disabled = false;
  };
  reader.readAsDataURL(file);
}

// Clasificar imagen de papa
async function clasificarPapa() {
  if (!selectedFile) return;

  ocultarError();
  ocultarResultado();

  // Mostrar cargando
  loadingSpinner.classList.add('visible');
  btnClassify.disabled = true;
  btnClear.disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Error al procesar la imagen en el servidor.');
    }

    const data = await res.json();
    mostrarResultado(data);
  } catch (err) {
    showError('Error en la clasificación: ' + err.message + '\nVerifica que el backend esté ejecutándose en localhost:8000.');
  } finally {
    loadingSpinner.classList.remove('visible');
    btnClassify.disabled = false;
    btnClear.disabled = false;
  }
}

// Renderizar resultados en el DOM
function mostrarResultado(data) {
  resultCard.classList.add('visible');
  
  // Variedad y Confianza Principal
  resVariety.textContent = data.variedad_predicha;
  const confPorcentaje = (data.confianza * 100).toFixed(2);
  resConfianza.textContent = `${confPorcentaje}%`;
  
  // Barra de confianza principal
  setTimeout(() => {
    confidenceBar.style.width = `${confPorcentaje}%`;
  }, 100);

  // Imagen de Ejemplo Real del Dataset
  if (data.imagen_ejemplo_url) {
    imgExample.src = data.imagen_ejemplo_url;
    imgExample.style.display = 'block';
    document.getElementById('noExampleText').style.display = 'none';
  } else {
    imgExample.style.display = 'none';
    document.getElementById('noExampleText').style.display = 'block';
  }

  // Top 5 Probabilidades
  top5Container.innerHTML = '';
  data.top_5.forEach((item, index) => {
    const probPorc = (item.probabilidad * 100).toFixed(2);
    
    const row = document.createElement('div');
    row.className = 'top5-row';
    row.innerHTML = `
      <div class="top5-info">
        <span class="top5-name">${index + 1}. ${item.variedad}</span>
        <span class="top5-val">${probPorc}%</span>
      </div>
      <div class="top5-bar-wrap">
        <div class="top5-bar" style="width: 0%; ${index === 0 ? 'background-color: var(--color-accent);' : ''}"></div>
      </div>
    `;
    top5Container.appendChild(row);

    // Animar la barra después de inyectarla
    setTimeout(() => {
      row.querySelector('.top5-bar').style.width = `${probPorc}%`;
    }, 150 + index * 50);
  });

  // Scroll suave al resultado
  setTimeout(() => {
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 200);
}

function ocultarResultado() {
  resultCard.classList.remove('visible');
  confidenceBar.style.width = '0%';
}

function showError(msg) {
  errorAlert.textContent = msg;
  errorAlert.classList.add('visible');
}

function ocultarError() {
  errorAlert.classList.remove('visible');
}

// Limpiar formulario completo
window.clearForm = function() {
  selectedFile = null;
  fileInput.value = '';
  imgPreview.src = '';
  previewContainer.style.display = 'none';
  dropZone.style.display = 'flex';
  btnClassify.disabled = true;
  ocultarError();
  ocultarResultado();
  
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
