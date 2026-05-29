// Catálogo de Síntomas codificados del s1 al s25
const SYMPTOMS = {
  s1:  { id: 's1',  label: 'Poliuria',              desc: 'Orinar con mucha frecuencia y en gran cantidad' },
  s2:  { id: 's2',  label: 'Polidipsia',            desc: 'Sed extrema que no se quita fácilmente' },
  s3:  { id: 's3',  label: 'Polifagia',             desc: 'Hambre excesiva incluso después de comer' },
  s4:  { id: 's4',  label: 'Pérdida de peso',       desc: 'Adelgazamiento involuntario sin dieta' },
  s5:  { id: 's5',  label: 'Fatiga extrema',        desc: 'Cansancio intenso sin causa aparente' },
  s6:  { id: 's6',  label: 'Visión borrosa',        desc: 'Dificultad para enfocar o ver con claridad' },
  s7:  { id: 's7',  label: 'Cicatrización lenta',   desc: 'Heridas o cortaduras que tardan en sanar' },
  s8:  { id: 's8',  label: 'Infecciones recurrentes', desc: 'Infecciones frecuentes de hongos, piel u orina' },
  s9:  { id: 's9',  label: 'Hormigueo en extremidades', desc: 'Adormecimiento u hormigueo en manos y/o pies' },
  s10: { id: 's10', label: 'Acantosis nigricans',   desc: 'Manchas oscuras y aterciopeladas en cuello o axilas' },
  s11: { id: 's11', label: 'Náuseas / vómitos',     desc: 'Náuseas o vómitos frecuentes' },
  s12: { id: 's12', label: 'Aliento cetónico',      desc: 'Olor dulce o frutal en el aliento' },
  s13: { id: 's13', label: 'Dolor abdominal',       desc: 'Dolor abdominal frecuente o intenso' },
  s14: { id: 's14', label: 'Confusión mental',      desc: 'Desorientación, dificultad para pensar' },
  s15: { id: 's15', label: 'Deshidratación severa', desc: 'Boca muy seca, ojos hundidos, piel sin turgencia' },
  s16: { id: 's16', label: 'Dolor en pies/piernas', desc: 'Dolor, ardor o puntadas en pies o piernas' },
  s17: { id: 's17', label: 'Pérdida de sensibilidad', desc: 'Sin tacto ni sensación en los pies' },
  s18: { id: 's18', label: 'Manchas visuales',      desc: 'Manchas, líneas flotantes o destellos en la visión' },
  s19: { id: 's19', label: 'Pérdida de visión',     desc: 'Pérdida parcial o total de visión' },
  s20: { id: 's20', label: 'Orina espumosa/oscura', desc: 'Orina con espuma o de color oscuro' },
  s21: { id: 's21', label: 'Edema (hinchazón)',     desc: 'Hinchazón en tobillos, pies o piernas' },
  s22: { id: 's22', label: 'Hipertensión',          desc: 'Presión arterial alta diagnosticada' },
  s23: { id: 's23', label: 'Inicio súbito',         desc: 'Síntomas que aparecieron en días o semanas' },
  s24: { id: 's24', label: 'Edad < 30 años',        desc: 'El paciente tiene menos de 30 años' },
  s25: { id: 's25', label: 'Antecedentes familiares / obesidad', desc: 'Familiar con DM2 o paciente con sobrepeso/obesidad' }
};

// Mapeo detallado de Diagnósticos
const DIAGNOSES = {
  prediabetes: {
    id: 'prediabetes',
    name: 'Prediabetes',
    type: 'Condición Previa',
    severity: 'moderada',
    color: '#F59E0B',
    description: 'Estado intermedio entre glucosa normal y diabetes. Glucosa en ayunas 100–125 mg/dL o HbA1c 5.7–6.4%. Reversible con cambios de estilo de vida.',
    recommendation: 'Consulte con su médico para realizar una prueba de glucosa en ayunas o HbA1c. Adopte hábitos saludables: dieta balanceada, ejercicio regular (150 min/semana) y control de peso. La prediabetes es reversible con intervención temprana.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 2', url: 'https://doi.org/10.2337/dc23-S002' },
      { label: 'PubMed — ADA Classification and Diagnosis 2023', url: 'https://pubmed.ncbi.nlm.nih.gov/36507649/' }
    ],
    rules: [
      ['s5', 's10'],
      ['s5', 's25', 's2'],
      ['s10', 's25']
    ]
  },
  dm2: {
    id: 'dm2',
    name: 'Diabetes Mellitus Tipo 2',
    type: 'Enfermedad Principal',
    severity: 'alta',
    color: '#EF4444',
    description: 'Trastorno metabólico por resistencia a la insulina con déficit progresivo de secreción. Representa el 90–95% de todos los casos de diabetes. Inicio gradual (meses a años).',
    recommendation: 'Busque atención médica para confirmación diagnóstica con glucosa en ayunas (≥126 mg/dL × 2 mediciones) o HbA1c ≥6.5%. El tratamiento incluye cambios de estilo de vida, metformina y otros fármacos según indicación médica. No se automedique.',
    sources: [
      { label: 'WHO HEARTS-D — Diagnosis and Management of Type 2 Diabetes (2020)', url: 'https://www.who.int/publications/i/item/who-ucn-ncd-20.1' },
      { label: 'PDF oficial OMS HEARTS-D', url: 'https://apps.who.int/iris/bitstream/handle/10665/331710/WHO-UCN-NCD-20.1-eng.pdf' },
      { label: 'ADA Standards of Care 2023', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s1', 's2', 's3'],
      ['s1', 's4'],
      ['s1', 's2', 's25'],
      ['s5', 's2', 's25'],
      ['s5', 's6', 's1'],
      ['s1/s2/s5/s6/s7', 's8'],
      ['s1/s2/s5/s8', 's7'],
      ['s1/s2/s3/s5', 's10'],
      ['s6', 's2', 's5']
    ]
  },
  dm1: {
    id: 'dm1',
    name: 'Diabetes Mellitus Tipo 1',
    type: 'Enfermedad Principal',
    severity: 'alta',
    color: '#DC2626',
    description: 'Destrucción autoinmune de células beta pancreáticas con déficit absoluto de insulina. Inicio súbito, frecuente en menores de 30 años. Requiere insulina de por vida.',
    recommendation: 'Acuda a urgencias o a su médico de inmediato. La DM Tipo 1 requiere insulina para sobrevivir. Si presenta náuseas, vómitos y aliento frutal, puede estar en cetoacidosis — EMERGENCIA MÉDICA. No demore la atención.',
    sources: [
      { label: 'WHO Classification of Diabetes', url: 'https://www.who.int/publications/i/item/who-ucn-ncd-20.1' },
      { label: 'ADA Standards of Care 2023', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s1', 's2', 's4', 's23'],
      ['s1', 's2', 's3', 's24'],
      ['s1', 's4', 's11', 's12'],
      ['s1', 's2', 's11', 's23'],
      ['s4', 's5', 's23', 's24']
    ]
  },
  cad: {
    id: 'cad',
    name: 'Cetoacidosis Diabética (CAD)',
    type: 'Complicación Aguda',
    severity: 'critica',
    color: '#991B1B',
    description: '⚠️ EMERGENCIA MÉDICA. Complicación aguda grave de DM1. Triada diagnóstica: hiperglucemia + cetonemia + acidosis metabólica. Mortalidad alta sin tratamiento inmediato.',
    recommendation: '🚨 LLAME AL SERVICIO DE EMERGENCIAS DE INMEDIATO. La cetoacidosis diabética es una emergencia que pone en riesgo la vida. Síntomas: náuseas, vómitos, aliento a frutas, confusión, dolor abdominal. No espere — acuda a urgencias ahora.',
    sources: [
      { label: 'Manual Merck — Cetoacidosis Diabética', url: 'https://www.merckmanuals.com/es-us/professional/trastornos-endocrinol%C3%B3gicos-y-metab%C3%B3licos/diabetes-mellitus-y-trastornos-del-metabolismo-de-los-hidratos-de-carbono/cetoacidosis-diab%C3%A9tica-cad' },
      { label: 'ADA Standards of Care 2023', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s11', 's12', 's13', 's1'],
      ['s11', 's12', 's14'],
      ['s12', 's11', 's15'],
      ['s13', 's14', 's1', 's4']
    ]
  },
  ehh: {
    id: 'ehh',
    name: 'Estado Hiperosmolar Hiperglucémico (EHH)',
    type: 'Complicación Aguda',
    severity: 'critica',
    color: '#B91C1C',
    description: '⚠️ EMERGENCIA MÉDICA. Complicación aguda grave de DM2. Glucosa muy elevada (>600 mg/dL), deshidratación extrema y confusión, SIN cetosis significativa. Diferente a la CAD.',
    recommendation: '🚨 LLAME AL SERVICIO DE EMERGENCIAS DE INMEDIATO. El EHH es una emergencia que pone en riesgo la vida. Caracterizado por confusión extrema, deshidratación severa y glucosa altísima. Requiere hospitalización urgente con hidratación intravenosa.',
    sources: [
      { label: 'Manual Merck — Estado Hiperosmolar', url: 'https://www.merckmanuals.com/es-us/professional/trastornos-endocrinol%C3%B3gicos-y-metab%C3%B3licos/diabetes-mellitus-y-trastornos-del-metabolismo-de-los-hidratos-de-carbono/estado-hipergluc%C3%A9mico-hiperosmolar' },
      { label: 'ADA Standards of Care 2023', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s1', 's15', 's14'],
      ['s15', 's2', 's14', 'no s12'],
      ['s14', 's11', 's1', 'no s12']
    ]
  },
  neuropatia: {
    id: 'neuropatia',
    name: 'Neuropatía Diabética',
    type: 'Complicación Crónica',
    severity: 'moderada',
    color: '#7C3AED',
    description: 'Daño nervioso periférico causado por hiperglucemia crónica. Afecta principalmente pies y piernas. Síntoma cardinal: hormigueo, adormecimiento o dolor ardoroso en extremidades.',
    recommendation: 'Consulte con su médico para evaluación neurológica. Controle estrictamente su glucosa. Realice revisión diaria de sus pies. Evite caminar descalzo. Use calzado adecuado. La neuropatía puede progresar a pie diabético sin tratamiento.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 12 (Neuropatía)', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s9', 's16'],
      ['s9', 's17'],
      ['s16', 's1'],
      ['s17', 's5', 's1']
    ]
  },
  pie_diabetico: {
    id: 'pie_diabetico',
    name: 'Pie Diabético',
    type: 'Complicación Crónica',
    severity: 'alta',
    color: '#6D28D9',
    description: 'Forma avanzada de neuropatía con pérdida de sensibilidad en pies combinada con cicatrización lenta e infecciones. Alto riesgo de úlceras y amputación.',
    recommendation: 'Acuda al médico urgentemente. El pie diabético requiere evaluación especializada. Inspeccione sus pies diariamente. Cualquier herida que no sane en 2–3 días debe ser evaluada por un médico. Requiere atención podológica especializada.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 12', url: 'https://doi.org/10.2337/dc23-S002' },
      { label: 'WHO HEARTS-D 2020', url: 'https://www.who.int/publications/i/item/who-ucn-ncd-20.1' }
    ],
    rules: [
      ['s17', 's7'],
      ['s17', 's7', 's8']
    ]
  },
  retinopatia: {
    id: 'retinopatia',
    name: 'Retinopatía Diabética',
    type: 'Complicación Crónica',
    severity: 'moderada',
    color: '#2563EB',
    description: 'Daño vascular en la retina por hiperglucemia prolongada. Principal causa de ceguera en adultos en edad laboral. Etapa inicial: manchas o destellos visuales.',
    recommendation: 'Consulte con oftalmólogo para fondo de ojo. El control estricto de glucosa puede retrasar la progresión. Realice exámenes oftalmológicos anuales si tiene diabetes. La detección temprana permite tratamiento con láser que preserva la visión.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 12 (Retinopatía)', url: 'https://doi.org/10.2337/dc23-S002' },
      { label: 'WHO HEARTS-D 2020', url: 'https://www.who.int/publications/i/item/who-ucn-ncd-20.1' }
    ],
    rules: [
      ['s18', 's6'],
      ['s18', 's1']
    ]
  },
  retinopatia_avanzada: {
    id: 'retinopatia_avanzada',
    name: 'Retinopatía Diabética Avanzada',
    type: 'Complicación Crónica',
    severity: 'alta',
    color: '#1D4ED8',
    description: 'Etapa proliferativa de la retinopatía. Formación de nuevos vasos frágiles que pueden sangrar. Riesgo alto de pérdida permanente de visión o desprendimiento de retina.',
    recommendation: 'Consulte con oftalmólogo de URGENCIA. La pérdida de visión puede ser permanente si no se trata a tiempo. El tratamiento con fotocoagulación láser o inyecciones intravítreas puede preservar la visión restante.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 12', url: 'https://doi.org/10.2337/dc23-S002' }
    ],
    rules: [
      ['s19', 's18'],
      ['s19', 's6', 's1']
    ]
  },
  nefropatia: {
    id: 'nefropatia',
    name: 'Nefropatía Diabética',
    type: 'Complicación Crónica',
    severity: 'alta',
    color: '#0891B2',
    description: 'Daño glomerular renal progresivo por hiperglucemia crónica. Principal causa de enfermedad renal crónica terminal en el mundo. Indicadores: proteinuria, edema e hipertensión.',
    recommendation: 'Consulte con su médico para análisis de orina (microalbuminuria) y función renal (creatinina, filtrado glomerular). Controle glucosa Y presión arterial estrictamente. Evite AINES y nefrotóxicos. Un nefrólogo puede ser necesario en etapas avanzadas.',
    sources: [
      { label: 'ADA Standards of Care 2023 — Sección 11 (Nefropatía)', url: 'https://doi.org/10.2337/dc23-S002' },
      { label: 'WHO HEARTS-D 2020', url: 'https://www.who.int/publications/i/item/who-ucn-ncd-20.1' }
    ],
    rules: [
      ['s20', 's21'],
      ['s20', 's22'],
      ['s21', 's22', 's1']
    ]
  }
};
