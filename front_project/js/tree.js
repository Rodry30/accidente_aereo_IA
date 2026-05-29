// Árbol de decisión estructurado
const treeData = {
  id: 'root',
  type: 'condition',
  label: '¿Presenta síntomas?',
  children: [
    {
      id: 'has_s1_yes',
      type: 'symptom',
      label: 'SÍ: Poliuria (s1)',
      children: [
        {
          id: 'has_s2_yes',
          type: 'symptom',
          label: 'SÍ: Polidipsia (s2)',
          children: [
            {
              id: 'has_s3_yes',
              type: 'symptom',
              label: 'SÍ: Polifagia (s3)',
              children: [
                {
                  id: 'diag_dm2_1',
                  type: 'diagnosis',
                  diagId: 'dm2',
                  label: 'Diabetes Tipo 2 (Tríada 3P)'
                }
              ]
            },
            {
              id: 'has_s3_no',
              type: 'symptom',
              label: 'NO: Polifagia (s3)',
              children: [
                {
                  id: 'has_s23_s24',
                  type: 'condition',
                  label: '¿Inicio súbito (s23) o Edad <30 (s24)?',
                  children: [
                    {
                      id: 'diag_dm1_1',
                      type: 'diagnosis',
                      diagId: 'dm1',
                      label: 'Diabetes Tipo 1'
                    },
                    {
                      id: 'diag_dm2_2',
                      type: 'diagnosis',
                      diagId: 'dm2',
                      label: 'Diabetes Tipo 2'
                    }
                  ]
                }
              ]
            },
            {
              id: 'has_cad_rules',
              type: 'condition',
              label: '¿Náuseas/Aliento Cetónico (s11+s12)?',
              children: [
                {
                  id: 'diag_cad_1',
                  type: 'diagnosis',
                  diagId: 'cad',
                  label: 'Cetoacidosis Diabética (CAD)'
                }
              ]
            },
            {
              id: 'has_ehh_rules',
              type: 'condition',
              label: '¿Deshidratación/Confusión (s15+s14)?',
              children: [
                {
                  id: 'diag_ehh_1',
                  type: 'diagnosis',
                  diagId: 'ehh',
                  label: 'Estado Hiperosmolar (EHH)'
                }
              ]
            }
          ]
        },
        {
          id: 'has_s2_no',
          type: 'symptom',
          label: 'NO: Polidipsia (s2)',
          children: [
            {
              id: 'has_s4_yes',
              type: 'symptom',
              label: '¿Pérdida de peso (s4)?',
              children: [
                {
                  id: 'diag_dm2_3',
                  type: 'diagnosis',
                  diagId: 'dm2',
                  label: 'Diabetes Tipo 2'
                }
              ]
            }
          ]
        },
        {
          id: 'has_s1_complications',
          type: 'condition',
          label: '¿Complicaciones Crónicas?',
          children: [
            {
              id: 'comp_neuro',
              type: 'diagnosis',
              diagId: 'neuropatia',
              label: 'Neuropatía (s9/s16/s17)'
            },
            {
              id: 'comp_retino',
              type: 'diagnosis',
              diagId: 'retinopatia',
              label: 'Retinopatía (s18/s19)'
            },
            {
              id: 'comp_nefro',
              type: 'diagnosis',
              diagId: 'nefropatia',
              label: 'Nefropatía (s20/s21)'
            }
          ]
        }
      ]
    },
    {
      id: 'has_s1_no',
      type: 'symptom',
      label: 'NO: Poliuria (s1)',
      children: [
        {
          id: 'no_s1_cond1',
          type: 'condition',
          label: '¿Fatiga (s5) + Acantosis (s10)?',
          children: [
            {
              id: 'diag_prediabetes_1',
              type: 'diagnosis',
              diagId: 'prediabetes',
              label: 'Prediabetes'
            }
          ]
        },
        {
          id: 'no_s1_cond2',
          type: 'condition',
          label: '¿Hormigueo en pies (s9+s16)?',
          children: [
            {
              id: 'comp_neuro_2',
              type: 'diagnosis',
              diagId: 'neuropatia',
              label: 'Neuropatía Diabética'
            }
          ]
        },
        {
          id: 'no_s1_cond3',
          type: 'condition',
          label: '¿Manchas visuales (s18+s6)?',
          children: [
            {
              id: 'comp_retino_2',
              type: 'diagnosis',
              diagId: 'retinopatia',
              label: 'Retinopatía Diabética'
            }
          ]
        }
      ]
    }
  ]
};

// Variables de Zoom y Pan
let scale = 0.75;
let translateX = 300;
let translateY = 40;
let isDragging = false;
let startX, startY;

// Diccionario plano de todos los nodos para búsqueda rápida y enlaces a padres
const nodesMap = {};

document.addEventListener('DOMContentLoaded', () => {
  initTree();
  setupZoomAndDrag();
});

function initTree() {
  const svg = document.getElementById('treeSvg');
  const group = document.getElementById('treeGroup');
  if (!svg || !group) return;

  // 1. Mapear nodos y asignar referencias de padres
  buildParentReferences(treeData, null);

  // 2. Calcular posiciones de nodos recursivamente
  computeLayout(treeData, 0, 0);

  // 3. Dibujar enlaces y nodos
  renderTreeElements(treeData);

  // 4. Aplicar transformación inicial
  updateTransform();
}

// Configurar referencias del padre para trazar caminos de vuelta a la raíz
function buildParentReferences(node, parent) {
  node.parent = parent;
  nodesMap[node.id] = node;
  if (node.children) {
    node.children.forEach(child => buildParentReferences(child, node));
  }
}

// Algoritmo de distribución del árbol
function computeLayout(node, depth = 0, leftBoundary = 0) {
  node.y = depth * 130 + 40; // Distancia vertical entre niveles

  if (!node.children || node.children.length === 0) {
    node.x = leftBoundary * 200 + 100; // Separación horizontal
    node.width = 1;
    return leftBoundary + 1; // Siguiente columna libre
  }

  let currentLeft = leftBoundary;
  let childXCoordinates = [];
  
  node.children.forEach(child => {
    currentLeft = computeLayout(child, depth + 1, currentLeft);
    childXCoordinates.push(child.x);
  });

  // Centrar el padre sobre sus hijos
  node.x = childXCoordinates.reduce((a, b) => a + b, 0) / childXCoordinates.length;
  return currentLeft;
}

// Renderizar visualmente el árbol
function renderTreeElements(node) {
  const group = document.getElementById('treeGroup');
  if (!group) return;

  // Dibujar enlaces a los hijos primero (para que queden por debajo de las cajas de nodos)
  if (node.children) {
    node.children.forEach(child => {
      // Crear camino Bezier curvado
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const startX = node.x;
      const startY = node.y;
      const endX = child.x;
      const endY = child.y;
      
      // Coordenadas de control Bezier
      const controlY1 = startY + 60;
      const controlY2 = endY - 60;
      
      const d = `M ${startX} ${startY} C ${startX} ${controlY1}, ${endX} ${controlY2}, ${endX} ${endY}`;
      
      path.setAttribute('d', d);
      path.setAttribute('class', 'link');
      path.setAttribute('id', `link-${node.id}-${child.id}`);
      group.appendChild(path);
      
      // Llamada recursiva para los hijos
      renderTreeElements(child);
    });
  }

  // Dibujar el nodo
  const nodeG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  nodeG.setAttribute('class', 'node');
  nodeG.setAttribute('id', `node-${node.id}`);
  nodeG.addEventListener('click', (e) => {
    e.stopPropagation();
    handleNodeClick(node);
  });

  if (node.type === 'condition') {
    // Rombo para preguntas
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    const halfW = 45;
    const halfH = 30;
    const points = `${node.x},${node.y - halfH} ${node.x + halfW},${node.y} ${node.x},${node.y + halfH} ${node.x - halfW},${node.y}`;
    
    polygon.setAttribute('points', points);
    polygon.setAttribute('fill', '#1E3A5F');
    polygon.setAttribute('stroke', '#0F172A');
    polygon.setAttribute('stroke-width', '1.5');
    nodeG.appendChild(polygon);

    // Texto de la condición
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', node.x);
    text.setAttribute('y', node.y + 4);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#FFFFFF');
    text.style.fontSize = '9px';
    text.style.fontWeight = 'bold';
    text.textContent = node.label.length > 15 ? node.label.slice(0, 13) + '...' : node.label;
    nodeG.appendChild(text);

  } else if (node.type === 'symptom') {
    // Rectángulo redondeado para síntomas
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const width = 160;
    const height = 36;
    
    rect.setAttribute('x', node.x - width / 2);
    rect.setAttribute('y', node.y - height / 2);
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('rx', '6');
    rect.setAttribute('fill', '#FFFFFF');
    rect.setAttribute('stroke', '#2563EB');
    rect.setAttribute('stroke-width', '1.5');
    nodeG.appendChild(rect);

    // Texto de síntoma
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', node.x);
    text.setAttribute('y', node.y + 4);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#0F172A');
    text.style.fontSize = '10px';
    text.style.fontWeight = '500';
    text.textContent = node.label;
    nodeG.appendChild(text);

  } else if (node.type === 'diagnosis') {
    // Pill para diagnóstico final
    const diag = DIAGNOSES[node.diagId];
    const color = diag ? diag.color : '#10B981';
    
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const width = 180;
    const height = 40;
    
    rect.setAttribute('x', node.x - width / 2);
    rect.setAttribute('y', node.y - height / 2);
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('rx', '20');
    rect.setAttribute('fill', color);
    rect.setAttribute('stroke', '#FFFFFF');
    rect.setAttribute('stroke-width', '2');
    nodeG.appendChild(rect);

    // Texto de diagnóstico
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', node.x);
    text.setAttribute('y', node.y + 4);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#FFFFFF');
    text.style.fontSize = '10px';
    text.style.fontWeight = 'bold';
    text.textContent = node.label;
    nodeG.appendChild(text);
  }

  group.appendChild(nodeG);
}

// Manejar clic en nodos
function handleNodeClick(node) {
  // 1. Resaltar el camino a la raíz
  highlightPath(node);

  // 2. Si es un diagnóstico, abrir el sidebar con información
  if (node.type === 'diagnosis') {
    openSidebarForDiagnosis(node.diagId);
  } else {
    closeSidebar();
  }
}

// Resaltar todos los nodos y enlaces en el camino de la raíz al nodo seleccionado
function highlightPath(node) {
  // Limpiar resaltado previo
  document.querySelectorAll('.link').forEach(link => link.classList.remove('highlighted'));
  document.querySelectorAll('.node').forEach(n => n.classList.remove('highlighted'));

  let current = node;
  while (current) {
    const nodeEl = document.getElementById(`node-${current.id}`);
    if (nodeEl) nodeEl.classList.add('highlighted');

    if (current.parent) {
      const linkEl = document.getElementById(`link-${current.parent.id}-${current.id}`);
      if (linkEl) linkEl.classList.add('highlighted');
    }
    current = current.parent;
  }
}

// Abrir el sidebar e inyectar detalles clínicos del diagnóstico
function openSidebarForDiagnosis(diagId) {
  const diag = DIAGNOSES[diagId];
  if (!diag) return;

  const sidebar = document.getElementById('detailsSidebar');
  const layoutWrapper = document.getElementById('layoutWrapper');
  
  document.getElementById('sideTitle').textContent = diag.name;
  document.getElementById('sideTitle').style.color = diag.color;
  
  // Configurar badges
  const badgeType = document.getElementById('sideBadgeType');
  badgeType.textContent = diag.type;
  badgeType.className = 'badge badge-primary';
  
  const badgeSev = document.getElementById('sideBadgeSeverity');
  badgeSev.textContent = `Severidad: ${diag.severity}`;
  
  let severityClass = 'badge-primary';
  if (diag.severity === 'moderada') severityClass = 'badge-warning';
  else if (diag.severity === 'alta' || diag.severity === 'critica') severityClass = 'badge-danger';
  badgeSev.className = `badge ${severityClass}`;

  document.getElementById('sideDesc').textContent = diag.description;
  document.getElementById('sideRec').textContent = diag.recommendation;

  // Ajustar el link de la base de conocimientos para resaltar el diagnóstico
  document.getElementById('sideKnowledgeLink').href = `index.html#diag-${diagId}`;

  // Mostrar sidebar y acomodar grilla
  layoutWrapper.classList.add('with-sidebar');
  sidebar.classList.add('visible');
}

function closeSidebar() {
  const sidebar = document.getElementById('detailsSidebar');
  const layoutWrapper = document.getElementById('layoutWrapper');
  if (sidebar && layoutWrapper) {
    sidebar.classList.remove('visible');
    layoutWrapper.classList.remove('with-sidebar');
  }
}

// Zoom & Drag setup
function setupZoomAndDrag() {
  const svg = document.getElementById('treeSvg');
  if (!svg) return;

  // 1. Mouse drag (Pan)
  svg.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // Solo clic izquierdo
    isDragging = true;
    startX = e.clientX - translateX;
    startY = e.clientY - translateY;
  });

  svg.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;
    updateTransform();
  });

  svg.addEventListener('mouseup', () => isDragging = false);
  svg.addEventListener('mouseleave', () => isDragging = false);

  // 2. Mouse Wheel Zoom
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = 1.1;
    if (e.deltaY < 0) {
      scale = Math.min(scale * zoomFactor, 3.0);
    } else {
      scale = Math.max(scale / zoomFactor, 0.25);
    }
    updateTransform();
  });
}

function updateTransform() {
  const group = document.getElementById('treeGroup');
  if (group) {
    group.setAttribute('transform', `translate(${translateX}, ${translateY}) scale(${scale})`);
  }
}

// Funciones globales para los botones de control
window.zoomIn = function() {
  scale = Math.min(scale * 1.2, 3.0);
  updateTransform();
};

window.zoomOut = function() {
  scale = Math.max(scale / 1.2, 0.25);
  updateTransform();
};

window.resetZoom = function() {
  scale = 0.75;
  translateX = 300;
  translateY = 40;
  updateTransform();
  
  // Limpiar resaltados
  document.querySelectorAll('.link').forEach(link => link.classList.remove('highlighted'));
  document.querySelectorAll('.node').forEach(n => n.classList.remove('highlighted'));
  closeSidebar();
};
