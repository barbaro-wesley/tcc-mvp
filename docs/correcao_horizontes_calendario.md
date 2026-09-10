# Experimento de horizontes por calendario

O script `scripts/33_s10_horizon_experiment.py` agora regulariza os precos com
`weekly_grid` antes do walk-forward. Uma previsao h=4 aponta para a data da
origem mais quatro semanas, mesmo quando faltam observacoes entre essas datas.

## Contrato dos dados

- Semanas ausentes ficam como `NaN` no historico do ARIMA. Nao ha preenchimento
  nem compressao do tempo. Datas duplicadas ou fora da grade sao rejeitadas.
- Cada linha registra `origin_date` e `target_date`. Uma origem sem preco nao
  emite previsao. Um alvo ausente continua ausente; a previsao daquela origem
  observada e emitida sem consultar a disponibilidade futura do alvo.
- `--start-index` conta semanas do calendario desde a primeira data, incluindo
  lacunas. `--min-train` conta precos observados anteriores a origem. Portanto,
  indices antigos em dados comprimidos nao representam necessariamente a mesma
  data inicial.
- MAE e a razao contra persistencia usam os mesmos pares completos. `n_scored`
  e `n_unscored` tornam a exclusao visivel. Sem pares, as metricas sao `null`;
  se o erro da persistencia for zero, a razao tambem fica `null`.

## Replay economico e resultados

A politica de compra em `procurement.py` nao foi alterada. Ela exige uma janela
semanal completa para validar as origens, calcular custos e anualizar economia.
Se uma lacuna estiver na janela de previsoes, o experimento continua calculando
a acuracia dos pares observados, mas registra
`procurement_status=unavailable_missing_observations` e metricas economicas
`null`. Nao e evidencia de economia zero ou de fracasso do modelo. Lacunas apenas
no historico de treinamento nao impedem o replay de uma janela completa.

Horizontes sem replay valido nao participam do ranking economico. Se nenhum for
elegivel, `best_horizon` e `null`. A evidencia continua restrita a desenvolvimento
por padrao; a janela de holdout nao foi movida.

Os resultados novos usam `pipeline_version=horizon-calendar-v2` e o diretorio
padrao `reports/vs_epl_krls/s10_horizon_v2`, preservando os relatorios anteriores.
Os valores antigos de economia e melhor horizonte precisam ser recalculados
antes de serem apresentados como resultados do pipeline corrigido.

```bash
python -m pytest tests/test_horizon_experiment.py tests/test_procurement.py
python scripts/33_s10_horizon_experiment.py
```

O experimento acima requer as fontes locais e o ambiente de producao do projeto.
Esta correcao nao retreina, publica nem promove uma release.
