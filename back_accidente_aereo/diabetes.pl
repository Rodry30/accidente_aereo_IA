:- dynamic tiene/1.

% Limpiar la base de datos de hechos dinámicos
limpiar :- retractall(tiene(_)).

% Insertar la lista de síntomas seleccionados por el paciente
assert_sintomas([]).
assert_sintomas([H|T]) :-
    assertz(tiene(H)),
    assert_sintomas(T).

% --- REGLAS DE DIAGNÓSTICO ---

% 1. Prediabetes
diagnostico(prediabetes) :- tiene(s5), tiene(s10).
diagnostico(prediabetes) :- tiene(s5), tiene(s25), tiene(s2).
diagnostico(prediabetes) :- tiene(s10), tiene(s25).

% 2. Diabetes Mellitus Tipo 2 (DM2)
diagnostico(dm2) :- tiene(s1), tiene(s2), tiene(s3).
diagnostico(dm2) :- tiene(s1), tiene(s4).
diagnostico(dm2) :- tiene(s1), tiene(s2), tiene(s25).
diagnostico(dm2) :- tiene(s5), tiene(s2), tiene(s25).
diagnostico(dm2) :- tiene(s5), tiene(s6), tiene(s1).
diagnostico(dm2) :- (tiene(s1); tiene(s2); tiene(s5); tiene(s6); tiene(s7)), tiene(s8).
diagnostico(dm2) :- (tiene(s1); tiene(s2); tiene(s5); tiene(s8)), tiene(s7).
diagnostico(dm2) :- (tiene(s1); tiene(s2); tiene(s3); tiene(s5)), tiene(s10).
diagnostico(dm2) :- tiene(s6), tiene(s2), tiene(s5).

% 3. Diabetes Mellitus Tipo 1 (DM1)
diagnostico(dm1) :- tiene(s1), tiene(s2), tiene(s4), tiene(s23).
diagnostico(dm1) :- tiene(s1), tiene(s2), tiene(s3), tiene(s24).
diagnostico(dm1) :- tiene(s1), tiene(s4), tiene(s11), tiene(s12).
diagnostico(dm1) :- tiene(s1), tiene(s2), tiene(s11), tiene(s23).
diagnostico(dm1) :- tiene(s4), tiene(s5), tiene(s23), tiene(s24).

% 4. Cetoacidosis Diabética (CAD) - Emergencia
diagnostico(cad) :- tiene(s11), tiene(s12), tiene(s13), tiene(s1).
diagnostico(cad) :- tiene(s11), tiene(s12), tiene(s14).
diagnostico(cad) :- tiene(s12), tiene(s11), tiene(s15).
diagnostico(cad) :- tiene(s13), tiene(s14), tiene(s1), tiene(s4).

% 5. Estado Hiperosmolar Hiperglucémico (EHH) - Emergencia
diagnostico(ehh) :- tiene(s1), tiene(s15), tiene(s14).
diagnostico(ehh) :- tiene(s15), tiene(s2), tiene(s14), \+ tiene(s12).
diagnostico(ehh) :- tiene(s14), tiene(s11), tiene(s1), \+ tiene(s12).

% 6. Neuropatía Diabética
diagnostico(neuropatia) :- tiene(s9), tiene(s16).
diagnostico(neuropatia) :- tiene(s9), tiene(s17).
diagnostico(neuropatia) :- tiene(s16), tiene(s1).
diagnostico(neuropatia) :- tiene(s17), tiene(s5), tiene(s1).

% 7. Pie Diabético
diagnostico(pie_diabetico) :- tiene(s17), tiene(s7).
diagnostico(pie_diabetico) :- tiene(s17), tiene(s7), tiene(s8).

% 8. Retinopatía Diabética
diagnostico(retinopatia) :- tiene(s18), tiene(s6).
diagnostico(retinopatia) :- tiene(s18), tiene(s1).

% 9. Retinopatía Diabética Avanzada
diagnostico(retinopatia_avanzada) :- tiene(s19), tiene(s18).
diagnostico(retinopatia_avanzada) :- tiene(s19), tiene(s6), tiene(s1).

% 10. Nefropatía Diabética
diagnostico(nefropatia) :- tiene(s20), tiene(s21).
diagnostico(nefropatia) :- tiene(s20), tiene(s22).
diagnostico(nefropatia) :- tiene(s21), tiene(s22), tiene(s1).

% --- MÉTODO PRINCIPAL DE EVALUACIÓN ---
% Limpia los hechos anteriores, aserta los síntomas actuales,
% encuentra todos los diagnósticos y los escribe separados por comas.
evaluar(ListaSintomas) :-
    limpiar,
    assert_sintomas(ListaSintomas),
    findall(D, diagnostico(D), DiagnosticosRepetidos),
    list_to_set(DiagnosticosRepetidos, Diagnosticos),
    escribir_resultados(Diagnosticos).

escribir_resultados([]) :- write('').
escribir_resultados([H|T]) :-
    write(H),
    (T = [] -> true ; write(','), escribir_resultados(T)).
