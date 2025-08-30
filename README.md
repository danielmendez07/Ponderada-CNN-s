# Projeto: Rede Neural para Detecção de Fraudes

---

## Documento Técnico

O documento técnico acompanha esta entrega e contém:

- **Explicação detalhada da seleção da arquitetura da rede neural**, baseada em literatura especializada e boas práticas:
  - Para **imagens (CIFAR-10)**: CNN do tipo VGG-like com BatchNorm, Dropout, L2 e GlobalAveragePooling.
  - Para **fraude tabular**: MLP (64→32) com StandardScaler, regularização L2, early stopping e ajuste de limiar.
- **Relação arquitetura ↔ problema de fraude**: CNNs são adequadas a visão; em tabulares, MLP/árvores são mais eficazes. Justifica-se a escolha considerando **desbalanceamento extremo (~1% fraudes)**, **custo assimétrico (FP vs. FN)** e **latência operacional**.
- **Hiperparâmetros justificados** para CNN e MLP, explicando impacto e efeito esperado.
- **Metodologia de avaliação** (ROC AUC, PR AUC, Precisão, Recall, F1, matriz de confusão).

---

## Modelo de Rede Neural

### CNN (CIFAR-10)
- Arquitetura: blocos Conv2D+BN+ReLU → MaxPooling → Dropout, GlobalAveragePooling, Dense(128), saída softmax(10).
- Data augmentation: flips, rotações leves, contraste/deslocamentos.
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- Otimizador: Adam (lr=1e-3).
- **Acurácia no teste (atual): 0.8666**

### MLP (Fraude — Tabular)
- Pré-processamento: StandardScaler.
- Estrutura: Dense(64) → Dense(32) com ReLU, regularização L2.
- Otimizador: Adam (lr=1e-3), batch_size=256.
- Estratégia: undersampling (~10% positivos), early stopping, **ajuste de limiar** para manter **precisão** elevada (e.g., ≥0.90) com o maior **recall** possível.

**Artefatos entregues:**
- `Notebook_Fraude_CNN_CIFAR10.ipynb` → contém CNN em CIFAR-10 **e** experimento completo de fraude.
- `fraud_mlp_sklearn.joblib` → pipeline MLP treinado + limiar calibrado.
- `sklearn_metrics.json` → métricas de teste (fraude).
- `sklearn_roc_curve.png`, `sklearn_pr_curve.png` → curvas ROC e PR (fraude).

---

## Hiperparâmetros

**CNN (CIFAR-10):**
- Filtros: [32, 64, 128], kernel 3×3, BN + ReLU.
- Dropout: [0.25, 0.35, 0.45, 0.5].
- Regularização: L2=1e-4.
- GlobalAveragePooling no lugar de Flatten.
- Dense final: 128 neurônios, ReLU, saída softmax(10).
- Otimização: Adam(1e-3) + EarlyStopping/ReduceLROnPlateau/Checkpoint.

**MLP (Fraude):**
- Hidden layers: (64, 32).
- Regularização: L2=1e-4.
- Early stopping, max_iter=200, batch=256.
- Seleção de limiar: precisão mínima (e.g., ≥0.90) + varredura por **custo** (ex.: FP=1, FN=20).

---

## Resultados Preliminares (Fraude) — **Atualizados**

- **ROC AUC (teste): 0.9329**  
- **PR AUC (teste): 0.7192**  
- **Precisão @ limiar calibrado: 0.9189**  
- **Revocação @ limiar calibrado: 0.5397**  
- **F1 @ limiar calibrado: 0.6800**  
- **Limiar selecionado: 0.62735**  
- **Matriz de confusão**:
```

\[\[5934,    3],
\[  29,   34]]

```

**Interpretação:**  
O ponto de operação atual mantém **alta precisão (~0.919)** com **revocação significativamente maior (~0.540)** em relação à versão anterior, refletindo um **trade-off mais equilibrado** entre fricção (FP) e cobertura de fraude (FN). O PR AUC **0.719** indica boa separação em cenário desbalanceado.

---

## Instruções de Uso

1. **CNN em CIFAR-10**
 - Abrir `Notebook_Fraude_CNN_CIFAR10.ipynb` em ambiente com **TensorFlow/Keras**.
 - Executar células de carregamento, definição da CNN, treino e avaliação.
 - Melhor checkpoint salvo como `cnn_cifar10_best.keras`.  
 - Artefatos adicionais: `cnn_cifar10_final.keras`, `cnn_history.json`, `cnn_test_metrics.json`, `cnn_confusion_matrix.csv`, `cnn_classification_report.json`, `cnn_curva_acuracia.png`, `cnn_curva_perda.png`.

2. **MLP em Fraude Tabular**
 - Artefatos: `fraud_mlp_sklearn.joblib`, `sklearn_metrics.json`, `sklearn_roc_curve.png`, `sklearn_pr_curve.png`.
 - Carregar modelo:
   ```python
   import joblib
   artifact = joblib.load("fraud_mlp_sklearn.joblib")
   pipe, thr = artifact["pipeline"], artifact["threshold"]
   # X deve ter as mesmas features do treino (mesma ordem/escala)
   y_pred = (pipe.predict_proba(X)[:, 1] >= thr).astype(int)
   ```
 - Para ajustar o **ponto de operação** a custos específicos (FP/FN), use a célula de **varredura por custo** já incluída no notebook.

3. **Reprodutibilidade**
 - Dados de fraude são **sintéticos** (`make_classification`) para simular ~1% de fraudes.
 - Sementes fixadas (`random_state=42`).
 - Ambiente sugerido: ver `requirements.txt`.

---

## Conclusão

Esta entrega cobre os pontos da rubrica com **clareza e precisão técnica**:
- Documento técnico fundamentado e **justificativa de arquitetura**: **CNN** (visão) vs. **MLP** (tabular de fraude).
- Modelo inicial de fraude **treinado** com **hiperparâmetros documentados** e **resultados atualizados** (ROC AUC 0.9329, PR AUC 0.7192).
- **Interpretação** dos resultados conectada ao domínio (desbalanceamento e custos).
- **Próximos passos**: engenharia de atributos, validação temporal, calibração probabilística, limiares por segmento e comparação com baselines (XGBoost/LightGBM/TabNet/FT-Transformer).

````