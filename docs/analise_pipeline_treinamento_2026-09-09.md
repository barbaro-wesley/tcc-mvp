# Análise do pipeline de treinamento — 09/09/2026

A prioridade é corrigir a representação dos dados e a consistência entre avaliação e produção, depois medir o benefício de informação externa mais recente. Há problemas concretos antes de qualquer necessidade de aumentar a complexidade do modelo. Nenhum ganho futuro de precisão está demonstrado por esta auditoria.

O primário atual é ARIMA; VS-ePL-KRLS é challenger. Foram inspecionados preparação, seleção temporal, produção, ingestão causal, calibração e resultados existentes. Os cálculos reproduzíveis estão no [diagnóstico](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/training_audit_2026-09-09/diagnostics.json) e no [script de auditoria](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/training_audit_2026-09-09/audit.py). A auditoria não alterou código de treinamento, artefatos servidos ou ledgers.

## O que significa melhorar o acerto

Recalculando as previsões já publicadas nas mesmas 104 semanas:

| Modelo | MAE, R$/L | RMSE, R$/L | Acerto da direção* | Erro até R$ 0,02/L |
|---|---:|---:|---:|---:|
| ARIMA | 0,02707 | 0,08146 | 59,2% | 73,1% |
| Paridade | 0,02752 | 0,08081 | 71,8% | 68,3% |
| Persistência | 0,03260 | 0,09563 | 0,0%* | 71,2% |

*Direção é calculada somente nas 71 semanas com variação diferente de zero. As outras 33 ficam fora dessa métrica; persistência prevê variação zero, portanto esse indicador a penaliza por construção. Não é uma acurácia de três classes.*

A paridade melhora a direção, mas seu MAE é aproximadamente 1,7% maior que o do ARIMA. Nas semanas com variação de até R$ 0,02/L, seu MAE é 43,95% maior. Para erro de até R$ 0,01/L, a persistência acerta 61,5%, o ARIMA 55,8% e a paridade 41,3%. Logo, a ordem dos modelos depende da tolerância e do objetivo.

Recomendação: declarar MAE como objetivo principal para preço, RMSE como diagnóstico dos grandes erros, e medir separadamente direção alta/estável/queda, acerto dentro de tolerâncias e precisão dos sinais de compra. O limiar de estabilidade deve ser fixado antes do experimento.

Fonte: [previsões históricas de paridade e ARIMA](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_parity/holdout_predictions.csv). O ARIMA da seleção original tem números ligeiramente diferentes; não misture as duas implementações numa comparação sem identificar a origem.

## 1. Alta prioridade: a normalização destrói informação no VS-ePL-KRLS

Em [selection.py](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:429), o MinMax é ajustado no passado e toda entrada posterior é cortada para [0,1].

No conjunto `lags` do candidato selecionado:

| Validação | Valores cortados | Semanas com algum corte | Vetores distintos antes → depois |
|---|---:|---:|---:|
| 22/08/2021–14/08/2022 | 99,04% | 100% | 52 → 2 |
| 21/08/2022–13/08/2023 | 5,77% | 25% | 52 → 52 |
| 20/08/2023–11/08/2024 | 0% | 0% | 52 → 52 |

Isso elimina quase toda a distinção entre os vetores de entrada no primeiro fold. O modelo ainda aprende alvos ao longo do tempo, mas recebe apenas dois padrões de atributos naquele bloco.

**Melhoria proposta:** comparar, isoladamente, uma representação baseada em variações, retornos e desvios relativos; preservar a geometria exigida pelas regras fuzzy com transformação limitada, definida no passado, que evite o corte rígido. Medir saturação e colisões como parte do ranking. Comparar depois janelas de treino e refit conjuntos de scaler e modelo.

Não atualizar apenas o scaler de um KRLS já treinado: centros e dicionários passariam a representar outra escala. O resultado acima comprova perda de informação; o ganho de MAE após a correção ainda precisa ser medido.

## 2. Alta prioridade: “uma semana” às vezes significa 14 ou 63 dias

[build_s10_supervised](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:209) cria o alvo com `shift(-horizon)` depois de remover linhas. A auditoria encontrou:

- 09/08/2015 → 23/08/2015: 14 dias.
- 16/08/2020 → 18/10/2020: 63 dias.

Lags e médias também atravessam essas lacunas como se as observações fossem consecutivas. Isso afeta o histórico usado para treinar, embora essas duas lacunas sejam anteriores aos folds recentes.

**Melhoria proposta:** manter grade semanal explícita; construir alvos por data e exigir diferença de 7 × horizonte dias. Para VS/Ridge, excluir exemplos sem alvo real e definir tratamento explícito das janelas de atributos que atravessam lacunas. Para ARIMA, preservar a passagem do tempo com observações ausentes ou avaliar segmentos regulares. Não inventar preços-alvo por interpolação.

A validação temporal exige preservar a ordem e a distância temporal das observações; a documentação também destaca a necessidade de amostras igualmente espaçadas para comparação entre folds. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## 3. Alta prioridade: treinamento e previsão devem compartilhar a mesma construção de atributos

A rotina de [produção da paridade](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/23_s10_parity_production.py:184) monta a próxima linha manualmente. Ela reaproveita `coint_par` da última linha, apenas corrigindo a escala pelo preço. Porém essa última linha já contém uma defasagem; reconstruir o próximo período pela função canônica incorpora a observação mais recente ao cálculo da cointegração.

A auditoria em desenvolvimento confirma que o atributo e a previsão mudam. Foram usados dois valores fictícios diferentes para o alvo seguinte e os atributos canônicos permaneceram iguais, verificando que a diferença não depende de conhecer esse alvo. A volatilidade também é reaproveitada manualmente e deve ser atualizada pela mesma função, ainda que possa coincidir em algumas semanas.

**Melhoria proposta:** uma única função para produzir os atributos de uma origem, usada tanto no walk-forward quanto no serving. Acrescentar uma verificação de igualdade dos atributos e da previsão em uma origem histórica. Isso corrige uma divergência, sem demonstrar por si só ganho de precisão.

No ARIMA há outra divergência: [seleção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/05_s10_model_selection.py:61) refaz o ajuste a cada 13 semanas e considera sete ordens em [classical.py](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/benchmarks/classical.py:19); [produção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/production.py:201) considera três ordens e reajusta a cada atualização. A seleção também parte da série reduzida pelo aquecimento dos atributos, enquanto produção usa o histórico completo.

No diagnóstico com histórico comum, 26 alvos de desenvolvimento entre 18/02/2024 e 11/08/2024 tiveram diferença máxima de apenas R$ 0,0000146/L e MAE praticamente idêntico. Portanto, **não há evidência de que essa diferença do ARIMA seja o principal gargalo**. Ainda assim, centralizar ordens, frequência de ajuste, histórico, fallback e calibração permite avaliar exatamente o algoritmo servido. A avaliação por origens móveis deve refletir o processo real de previsão. [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html).

## 4. Alta prioridade: alinhar a seleção com o critério de qualidade

O [ranking do VS](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:594) ainda combina média, pior fold e dispersão da razão de RMSE. Já os gates mais novos usam MAE, regimes e bootstrap.

Nas previsões históricas do ARIMA, uma semana concentra 75,09% do erro quadrático e três concentram 91,17%. Isso demonstra concentração do RMSE; não significa formalmente que existam apenas três observações independentes.

**Melhoria proposta:** selecionar por MAE fora da amostra, impor limites de regressão por regime e usar bootstrap em blocos para quantificar incerteza. Manter RMSE como métrica secundária. Incluir concentração dos erros, taxa de corte das entradas e churn do dicionário no relatório.

Os pesos do ensemble são ajustados e pontuados sobre os mesmos resíduos de validação em [05_s10_model_selection.py](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/05_s10_model_selection.py:116). Reservar um bloco posterior ou usar validação temporal aninhada para avaliar esses pesos reduz o otimismo dessa escolha.

As datas do holdout já foram fixadas, o que é correto. Contudo, o script 05 ainda o avalia em toda execução, e `--reuse-validation` reaproveita ranking por arquivo/ID sem conferir integralmente dados, código e configuração. Separar seleção de avaliação final e verificar essas identidades antes de reutilizar resultados. O holdout nacional já observado serve como evidência histórica; a confirmação de novos candidatos deve vir de previsões futuras congeladas.

## 5. Maior hipótese de ganho de sinal: dados externos disponíveis na hora da emissão

Existem dois caminhos de dados. No painel antigo, `ulsd` e `ulsd_l1` estão 100% ausentes; no painel causal novo, ULSD e paridade estão completos nas 705 linhas. A coluna `distribuicao_l1` antiga está ausente em 43,55% das linhas. O `petrobras_reajuste` legado é derivado do preço de revenda; não é um histórico de anúncios da Petrobras.

**Melhoria proposta:** consolidar o painel causal como fonte dos próximos experimentos, renomear o proxy para refletir sua origem e exigir qualidade e idade máxima por fonte. `ffill` sem limite pode manter uma série aparentemente completa depois que a fonte para de atualizar.

O painel causal alinha preços diários ao domingo inicial da semana; os atributos da paridade ainda aplicam defasagem. Isso é conservador, mas pode descartar dados que já existiam no instante real da previsão. Modelar separadamente `reference_date`, `published_at` e `forecast_issued_at`; fazer junção pelo que estava publicado até a emissão. Preservar versões históricas das fontes para não usar revisões retrospectivas como se estivessem disponíveis na época.

A hipótese com maior justificativa econômica é estruturar anúncios de reajuste: data de publicação, início de vigência, magnitude em R$/L e produto. Há comunicado oficial, por exemplo, com anúncio em 31/01/2025, vigência em 01/02 e aumento de R$ 0,22/L no diesel A. Isso comprova que a fonte contém variáveis estruturáveis, não que seu uso assegure ganho. [Petrobras](https://agencia.petrobras.com.br/pt/w/negocio/petrobras-ajusta-precos-de-diesel-para-distribuidoras).

Atrasos reais de publicação do produtor precisam continuar respeitados. Não reduzir defasagens arbitrariamente para melhorar o backtest.

## Sequência de experimentos recomendada

1. Fixar objetivo e tolerâncias, versão dos dados, calendário e protocolo temporal.
2. Corrigir geração por data e unificar atributos de treinamento/serving; reproduzir o baseline usando o caminho de produção.
3. Comparar uma mudança por vez no VS: representação sem saturação, depois memória de treino. Medir MAE, regimes, cortes e estabilidade nos mesmos folds.
4. Introduzir dados causais por disponibilidade real; adicionar anúncios estruturados quando houver histórico verificável. Comparar com ARIMA e persistência.
5. Congelar o candidato e acompanhar prospectivamente. Não trocar o primário apenas porque um novo experimento venceu no desenvolvimento.

A documentação registra tentativas anteriores sem ganho robusto com mais capacidade, forgetting, novos kernels, boosting, misturas de regime e híbridos. Repeti-las com os mesmos dados tem prioridade menor; corrigir a representação pode justificar uma nova comparação controlada.

Os horizontes maiores já foram explorados, mas não significam maior precisão do preço: o experimento existente tem MAE de aproximadamente R$ 0,0503/L em h=1 e R$ 0,4705/L em h=12, em amostras diferentes e com outro ARIMA. Economia de compra e erro de previsão precisam continuar separados.

O conformal adaptativo já existe no bundle 1.2.0. A cobertura de 92,3% exibida na model card é evidência histórica do intervalo anterior; não mede automaticamente o intervalo adaptativo atual. Avaliar calibração com resíduos causais do algoritmo realmente servido. O runner semanal também atualiza paridade/RS, mas não o ARIMA servido; a atualização operacional desse primário deve ter fluxo próprio, separado da escolha de um novo modelo.

## Verificação

Foram executados os testes de seleção S10, janelas fixas, produção S10, ingestão causal, repasse e calibração: **118 passaram**, em 16,89 segundos. Houve apenas aviso de escrita no cache do pytest. A suíte não detectou os problemas de representação e consistência medidos acima.

O diagnóstico novo de ARIMA usa apenas desenvolvimento. As métricas do holdout foram recalculadas a partir de previsões já gravadas, sem usá-las para ajustar candidatos. A auditoria não executou busca de hiperparâmetros nem promoção de modelo.

