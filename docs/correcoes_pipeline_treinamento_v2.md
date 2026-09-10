# Correções do pipeline de treinamento — primeira etapa

Branch: `codex/corrigir-pipeline-treinamento`.

As alterações corrigem a preparação temporal, preservam diferenças entre entradas fora da faixa histórica e aproximam avaliação e execução. Os artefatos servidos, os resultados históricos e os ledgers não foram atualizados por esta etapa. As alterações locais que já existiam antes da criação da branch foram preservadas.

## Comportamento implementado

- **Calendário:** uma grade explícita insere semanas ausentes com NaN. Alvos são associados à data de origem mais o horizonte, nunca à próxima linha disponível. Exemplos do VS/Ridge precisam de 13 semanas consecutivas de contexto; alvos ausentes não são inventados.
- **Aprendizado online:** a liberação dos alvos usa suas datas reais. Quando há um intervalo entre origens elegíveis, todos os alvos que já ficaram disponíveis são incorporados antes da previsão.
- **Normalização:** `robust_bounded` usa mediana e IQR do treino, seguidos de arctangente para manter as entradas no intervalo das regras fuzzy. A transformação permanece congelada com o modelo. Candidatos novos da grade usam essa opção; candidatos antigos sem o campo continuam com MinMax. A normalização faz parte da configuração serializada e da identidade dos novos candidatos.
- **Paridade:** treinamento e previsão da próxima semana compartilham a função de atributos. A próxima linha tem preço e alvo desconhecidos; cointegração, volatilidade e lags são calculados apenas a partir das observações anteriores. A grade preserva semanas ausentes.
- **ARIMA:** a seleção principal usa a mesma rotina de ajuste do bundle, por padrão a cada semana, e recebe todo o histórico em grade semanal. Não usa mais apenas os preços das linhas que sobreviveram ao aquecimento dos atributos. As atualizações online do bundle rejeitam resultados de outra semana antes de alterar o estado.
- **Consumidores:** o shadow preserva semanas e observações intermediárias no histórico ARIMA; o experimento híbrido também conserva o histórico completo. A comparação estadual direta alinha as previsões por data depois da regularização do painel.
- **Seleção:** o ranking prioriza razão de MAE, pior fold e dispersão entre folds. RMSE continua reportado. A taxa de corte das entradas aparece nas métricas. Os pesos do ensemble são ajustados por MAE, mas seu desempenho nos mesmos folds não constitui avaliação independente.
- **Holdout e cache:** o script de seleção termina no desenvolvimento por padrão e escreve em `s10_selection_v2`. A avaliação do holdout requer `--evaluate-holdout`. O reaproveitamento do ranking confere um fingerprint dos dados de desenvolvimento, configurações, folds e código; caches antigos incompatíveis são recusados.

## Experimento controlado

Foi comparado apenas o scaler, mantendo os parâmetros do candidato histórico e os mesmos três folds de desenvolvimento, de 22/08/2021 a 11/08/2024. Ambos os caminhos usam as correções de calendário. Nenhuma busca de hiperparâmetros ou avaliação nova do holdout foi executada nesse experimento.

| Indicador nas 156 semanas | MinMax com corte | Robust bounded |
|---|---:|---:|
| MAE, R$/L | 0,057177 | 0,055336 |
| RMSE, R$/L | 0,116860 | 0,118802 |
| Maior fração de entradas cortadas em um fold | 99,04% | 0% |
| Pior razão de MAE contra persistência | 1,0833 | 1,0159 |

O MAE melhorou aproximadamente **3,2%**, com redução nos três folds. O RMSE agregado piorou aproximadamente **1,7%**; no terceiro fold, o RMSE aumentou de 0,06379 para 0,07775. A nova representação ainda perde da persistência em MAE no segundo fold. Esses resultados não justificam promover o VS ao lugar do ARIMA.

A redução de MAE mede a comparação entre scalers no pipeline corrigido. Não é uma comparação completa entre a release antiga e uma nova release. Também não é evidência prospectiva independente.

Resultados e código reproduzível:

- [Ablation JSON](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_training_v2/ablation.json)
- [Métricas por fold](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_training_v2/folds.csv)
- [Script do experimento](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/35_s10_training_ablation.py)

No PowerShell, a partir da raiz do projeto:

```powershell
.venv/Scripts/python.exe scripts/35_s10_training_ablation.py
.venv/Scripts/python.exe scripts/05_s10_model_selection.py --n-random 6
```

O primeiro comando repete a comparação sem busca. O segundo inicia uma seleção nova somente em desenvolvimento. Seus manifestos de desenvolvimento não são autorização nem entrada completa para promover uma release. O holdout nacional já utilizado não deve ser usado para escolher novos parâmetros.

## Limites e trabalho seguinte

Esta etapa não criou uma série de anúncios da Petrobras, não reconstruiu timestamps históricos de publicação e não consolidou o painel legado com o causal. Essas são frentes de dados posteriores às correções demonstradas aqui.

A comparação deve ser seguida de avaliação dos erros por regime, comparação do candidato contra o ARIMA pelo mesmo protocolo e evidência prospectiva antes de qualquer troca do modelo servido. Os folds utilizados para ajustar pesos do ensemble precisam de avaliação temporal independente. Normalização suave preserva mais informação, mas não garante que grandes choques passem a ser previstos corretamente.

Atualização online agora exige a semana prevista. Em caso de lacuna real, não preencher artificialmente a observação para satisfazer o contrato; recuperar a fonte oficial ou realizar um refit explícito, preservando a ausência no calendário. O componente fuzzy precisa de contexto recente completo para emitir atributos.

## Verificação

Foram adicionados testes de alvos por calendário, revelação atrasada após lacunas, preservação de entradas fora da faixa de treino, serialização do novo scaler, igualdade de atributos e previsões da paridade, proteção do holdout, invalidação do cache e uso do histórico completo no ARIMA. Lint, compilação e o escopo de tipos usado pela CI passaram.

Execução final: **463 testes passaram, 2 foram ignorados por dependências opcionais ausentes e a cobertura foi de 91,69%**, acima do mínimo de 90%. As dependências opcionais ausentes são GBM e torch. Nenhuma release foi promovida.

Uma checagem adicional de tipos sobre seleção, híbrido, shadow, produção e paridade, mais ampla que o escopo da CI, ainda aponta 22 erros. O mesmo comando sobre os arquivos do HEAD anterior, copiados para um diretório temporário, aponta 23 erros. A dívida de tipagem fora da CI não foi resolvida nesta etapa.
