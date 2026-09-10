# Model card — previsão semanal nacional do Diesel B S10

- Modelo primário aprovado: **ARIMA**
- Challenger monitorado: VS-ePL-KRLS corrigido e normalizado
- Horizonte: 1 semana
- Fonte: ANP, preço médio nacional de revenda
- Treino: 2012-12-30 a 2026-08-30
- Observações: 705
- Fingerprint dos dados: `0c86155ba61673b6edb304b12d48e8af0e1e1a2e9d5506e9b1e25b4f3d95f6c4`
- SHA-256 do artefato: `bb7d0aa61f3df2f633bd00983538600b66b9f32edac019659264698ebc036550`
- Versão do contrato do artefato: 1.2.0
- Latência end-to-end p95 (100 chamadas): 13.43 ms
- Tamanho do artefato: 1.50 MiB

## Holdout final (104 semanas)

| model | rmse | mae | smape | directional_accuracy | rmse_ratio_vs_naive | dm_pvalue |
| --- | --- | --- | --- | --- | --- | --- |
| ARIMA | 0.08145 | 0.02706 | 0.41207 | 0.59155 | 0.85175 | 0.13380 |
| ensemble | 0.08421 | 0.02852 | 0.43286 | 0.78873 | 0.88060 | 0.12053 |
| VS-ePL-KRLS | 0.09382 | 0.03360 | 0.50865 | 0.63380 | 0.98105 | 0.03953 |
| persistencia | 0.09563 | 0.03260 | 0.48984 | 0.00000 | 1.00000 | 1.00000 |
| Ridge | 0.10694 | 0.03955 | 0.58963 | 0.38028 | 1.11824 | 0.07157 |

## Intervalo calibrado

- Cobertura nominal P10–P90: 80.0%
- Cobertura no holdout: 92.3%
- Largura média: R$ 0.121/L
- Resíduos de calibração: 156
- Janela adaptativa máxima após atualização online: 156 semanas

## Previsão atual

- Última observação: 2026-08-30, R$ 6.880/L
- Próxima semana: 2026-09-06
- Ponto: R$ 6.880/L
- Intervalo P10–P90: R$ 6.841–6.902/L
- Fallback usado: False

## Saúde do challenger

- Estado: warning
- Regras: 13/20
- Maior dicionário: 20/20
- Substituições de elementos KRLS: 144 (20.81% das atualizações)
- Cobertura online observada: aguardando 20 semanas
- MAE online recente: aguardando observações
- Avisos: dictionary_capacity_pressure, dictionary_replacement_churn, beta_floor_pressure

## Política operacional

- O VS-ePL-KRLS permanece em shadow mode porque não passou os gates de promoção.
- Previsões não finitas ou mudanças semanais fora do limite robusto acionam fallback.
- Nova semana deve ser incorporada por `update_one`; datas repetidas ou preços inválidos são rejeitados.
- Pressão de regras/dicionários, churn, clipping e cobertura online aparecem em `health()`.
- Reexecutar seleção antes de trocar o primário; nunca promover usando o holdout para ajustar parâmetros.

## Limitações

- Preço médio nacional, não preço de um posto ou estado.
- O intervalo por quantis de resíduos é empírico; após cada atualização, usa uma janela móvel e só alerta cobertura após 20 realizações online.
- Choques de política de preços e eventos externos podem exceder os padrões históricos.
- O gap de coleta da ANP em 2020 permanece no histórico e deve ser monitorado.