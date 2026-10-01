# Reporte de Reproducibilidad y Smoke Test

**Objetivo:** Validar el determinismo estricto del pipeline de entrenamiento (semillas, aumento de datos y orden de operaciones) y garantizar que los pesos óptimos (`best.pt`) puedan ser restaurados en procesos independientes para la fase de inferencia.

## 1. Prueba de Determinismo (Modo SMOKE)

Se ejecutaron dos ciclos de entrenamiento cortos consecutivos (`max_epochs=15`, truncados por early stopping) utilizando exactamente la misma configuración y semillas de control (`seed_split=42, seed_train=43, seed_aug=44, seed_model=45`).

| Métrica Final | Corrida 1 (Run ID: 205917f8d90f44ef9df35045800bf0c7) | Corrida 2 (Run ID: e69e526510684f869fee2e7750fc0fbc) | Diferencia |
| :--- | :--- | :--- | :--- |
| **best_val_loss** | 0.7195 | 0.7195 | 0.0000 |
| **best_val_accuracy** | 0.5000 | 0.5000 | 0.0000 |
| **best_val_macro_f1** | 0.3333 | 0.3333 | 0.0000 |
| **Tiempo por corrida** | 3.70 s | 1.65 s | - |

**Veredicto:** Las métricas coinciden con una tolerancia de error de `0.0000`. El contrato de semillas inyectado en `train_with_mlflow` previene fluctuaciones estocásticas entre corridas.

## 2. Validación de Checkpoint en Proceso Aislado

Se levantó un subproceso limpio en memoria (`multiprocessing.Process`) desvinculado del ciclo de entrenamiento original para probar el mejor checkpoint de la Corrida 1.

1. Se descargó el artefacto `checkpoint/best.pt` usando la API de `MlflowClient`.
2. Se instanció la arquitectura cruda del modelo usando `build_model(config)`.
3. Se inyectó el diccionario de estado con `load_state_dict`.
4. Se generó un pase hacia adelante (`forward pass`) con un tensor sintético determinista.

**Veredicto:** El modelo restauró sus pesos exitosamente sin errores de mapeo topológico, reproduciendo las predicciones esperadas (`tensor([[0.0028, 0.2946]])`) en el nuevo proceso. La procedencia del artefacto está certificada por su etiqueta `checkpoint_sha256`.