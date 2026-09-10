# Roadmap do produto — o que avançar e o que endurecer

Documento prospectivo. O histórico do que já foi feito está em
[s10_improvement_roadmap.md](s10_improvement_roadmap.md); aqui só o que ainda não existe.

Cada item traz **por que** (a evidência que o motiva), **esforço** e **o que destrava**.
Nenhum item foi incluído por ser boa prática genérica: todos saem de algo medido neste
repositório.

---

## Rodada executada — 24/08/2026

Cinco itens saíram do papel. Dois entregaram o que prometiam, dois foram **testados e
fechados com resultado negativo**, e um mudou de forma no caminho.

| item | resultado |
|---|---|
| **B1** servir mais estados | **entregue.** Dez estados ingeridos com um download; a previsão nacional é calculada uma vez e compartilhada. O custo marginal de outro estado é ler uma tabela. |
| **C3** pooling hierárquico | **fechado, negativo.** Melhorou o MAE em 5 de 10 estados, e a correlação entre número de postos e ganho é **+0,009** — nula. A premissa estava errada. |
| **A1** recalibrar o intervalo | **entregue.** O bundle aprende o nível online e nasce calibrado. |
| **A3** alertas do ledger | **entregue.** Quatro sinais, script com código de saída, exposto em `/v1/governance`. |
| **C4** VS-ePL-KRLS no spread | **fechado, negativo** — e o motivo é mais interessante que o resultado. |

### C3 — por que o pooling não era o remédio

O estimador de efeitos aleatórios devolveu pesos próprios entre 0,81 e 0,98 para todos os
estados: `tau²`, a variância *entre* estados, é grande em relação ao erro de cada
estimativa. Em português, **os estados realmente revertem em velocidades diferentes**, então
há pouco a tomar emprestado. É o próprio estimador dizendo que não é aqui.

O erro do diagnóstico foi confundir dois tamanhos de amostra: o número de postos afeta o
ruído de **cada observação semanal**, enquanto `kappa` é estimado sobre **centenas de
semanas** e por isso já chega preciso em todo estado. Pooling resolve poucas observações,
não observações ruidosas.

**Onde o tamanho da amostra de fato importa:** as semanas do quartil inferior de postos têm
**cerca de 1,9× a volatilidade** do spread das demais — de 1,3× em São Paulo a 2,6× em Mato
Grosso. A ANP publica esse número em toda linha e o modelo o ignorava. A correção é mínimos
quadrados ponderados pelo número de postos, disponível em
`SpreadForecaster(weight_by_stations=True)`. Ganho medido: melhora em 6 de 10 estados,
decidível em 3. Modesto e honesto.

### A1 — o intervalo errava dos dois lados, não só para mais

O diagnóstico anterior dizia "conservador demais". A medição sequencial mostrou algo pior:
a banda de quantil fixo cobriu **62,2% no terço volátil** e **97,8% no terço calmo** da
janela de calibração. Não é uma banda larga, é uma banda que não serve nenhum dos dois
regimes — e o erro para menos acontece exatamente nas semanas que decidem a compra.

O bundle passou ao contrato `1.2.0` com nível conformal adaptativo, aceitando artefatos
`1.1.0` sem alteração de comportamento. E `warm_start_interval_alpha()` faz a release
**nascer calibrada**, em vez de levar dezenas de semanas para descobrir sozinha o que a
janela de resíduos já diz.

### C4 — o VS não perde para o linear; os dois perdem para o horizonte

O VS-ePL-KRLS no spread ficou atrás da correção linear em todos os quatro estados testados.
Mas o número que importa está na linha da persistência: **o modelo linear a supera por uma
fração de por cento**, e em Minas Gerais nem isso. Não é que o VS seja ruim no spread — é
que em uma semana não há o que prever, para ninguém.

R² da variação futura do spread explicada pelo desvio corrente da média:

| UF | h=1 | h=4 | h=8 | **h=12** | h=26 |
|---|---:|---:|---:|---:|---:|
| RS | 0,031 | 0,060 | 0,118 | **0,154** | 0,145 |
| SP | 0,026 | 0,062 | 0,110 | **0,134** | 0,076 |
| MG | 0,033 | 0,053 | 0,079 | **0,096** | 0,084 |
| PR | 0,013 | 0,018 | 0,024 | **0,029** | 0,025 |

**Cinco vezes mais sinal em doze semanas do que em uma**, com pico exatamente onde a
meia-vida de reversão de ~20 semanas prediz, e queda em 26 quando o ruído acumulado engole
o sinal.

Isso promove **B2 (horizonte maior)** de "boa ideia comercial" para **a próxima coisa a
fazer**, com evidência quantitativa. E é onde regras evolutivas merecem seu teste justo:
o experimento do C4 fica pronto para ser reexecutado em h=12 trocando um parâmetro.

---

## A. Endurecer o que já temos

Trabalho definido, sem pesquisa. É o que transforma "funciona" em "opera".

### A1. Recalibrar o intervalo do bundle de produção — **entregue**

**Por quê:** o intervalo servido cobre 92,3% para um nominal de 80%. A ferramenta que
corrige isso já existe, está testada e já roda no modelo de paridade, onde reduziu a banda
em 19,4% com a cobertura no alvo. O bundle de produção continua com o intervalo antigo
porque ele vive dentro do artefato congelado.

**Esforço:** baixo — uma release nova com hash, ledger e evidência próprios.

**Destrava:** cenários de custo úteis. Uma banda larga demais empurra o P90 para longe do
plausível e infla o risco aparente de não antecipar — exatamente a decisão que o produto
existe para informar.

### A2. Gates no CI, não sob demanda

**Por quê:** `24_s10_gate_review.py` roda quando alguém lembra. Um veredito que muda sem
ninguém perceber é o mesmo problema da janela que escorregava.

**Esforço:** baixo — um passo no workflow que roda a revisão e falha se um veredito mudar
sem atualização da evidência versionada.

**Destrava:** governança que não depende de disciplina humana semanal.

### A3. Alerta sobre os ledgers prospectivos — **entregue**

**Por quê:** os ledgers acumulam corretamente, mas ninguém é avisado de nada. Os três
sinais que importam: cobertura do intervalo saindo da faixa operacional, erro do challenger
superando o primário por N semanas seguidas, e a contagem chegando a 26.

**Esforço:** baixo — os dados já estão no ledger; falta o leitor.

**Destrava:** a promoção deixa de exigir que alguém olhe. É o que faz a evidência
prospectiva valer alguma coisa na prática.

### A4. Refazer a seleção de produção com os atributos causais

**Por quê:** achado registrado e nunca corrigido — `05_s10_model_selection.py:203` monta
apenas os conjuntos `price`, `lags` e `dynamics`. **O campeão de produção foi selecionado
sem jamais ver Brent ou câmbio.** Depois disso o projeto descobriu que o sinal causal é o
que importa.

**Esforço:** médio. E há uma restrição dura: o holdout nacional está gasto, então a
reseleção só pode acontecer em desenvolvimento, com a especificação pré-registrada e a
confirmação vindo do ledger.

**Destrava:** saber se o primário atual é o melhor primário possível, ou apenas o melhor
entre os que foram olhados.

### A5. Extrair a mecânica repetida de walk-forward

**Por quê:** `national_walk_forward` está duplicado entre os scripts 26 e 27, e os scripts
22, 25 e 26 repetem a mesma montagem de folds. Duas cópias divergem; foi por isso que a
mecânica de ledger foi para `audit.py`.

**Esforço:** baixo.

**Destrava:** menos superfície para bug silencioso, e execução mais rápida se o resultado
for cacheado entre scripts.

### A6. Executar o container e medir SLO de verdade

**Por quê:** o Dockerfile está endurecido e nunca foi executado; o teste de carga é
in-process. 267 req/s nesta máquina não é SLO.

**Esforço:** médio, depende de infraestrutura.

**Destrava:** o bloqueador nº 2 do documento de prontidão.

---

## B. Avanços de produto

Capacidade nova para o cliente, sem depender de pesquisa.

### B1. Servir os 27 estados — **entregue para 10**

**Por quê:** `UF_REGION` já cobre todas as unidades da federação e o script aceita `--uf`.
O trabalho do RS **é** o trabalho de qualquer estado — falta rodar e servir.

**Esforço:** baixo por estado. Priorize por volume de diesel: SP, MG, MT, GO, PR, RS, BA.

**Destrava:** multiplica o mercado endereçável sem uma linha de pesquisa nova. É o item de
melhor razão valor/esforço do roadmap inteiro.

**Cuidado:** estados com poucos postos pesquisados terão séries mais ruidosas que o RS. A
correção certa é o item C3, não vender o número como se fosse igual.

### B2. Horizonte maior que uma semana — **prioridade nº 1 agora**

**Por quê:** o produto prevê h=1. Compra de diesel raramente é decidida com sete dias de
antecedência — contrato, frete e lote mínimo empurram a decisão para semanas. E há um
argumento empírico forte: o spread estadual reverte com meia-vida de ~20 semanas, ou seja,
**o sinal de médio prazo é mais forte que o de curto**, ao contrário do preço em nível.

**Esforço:** médio. A infraestrutura de folds, gates e intervalo já é agnóstica ao
horizonte; o replay de política precisa de uma versão multi-semana.

**Destrava:** muda o uso de "antecipar esta semana" para "planejar o trimestre", que é a
conversa que o comprador realmente tem.

#### Rodada de 09/09/2026 — replay multi-horizonte entregue, hipótese confirmada

O replay de política deixou de ser 1-semana: `simulate_horizon_prebuy` replica a mesma
política para qualquer `h`, e `simulate_one_week_prebuy` virou o caso `h=1` — os números de
h=1 já publicados foram reconferidos e **não mudaram** (economia anualizada R$ 8.563,11,
IC90 [116,50, 23.653,40], 9 disparos).

Duas correções que o horizonte exige e que estão explícitas no código: antecipar `h` semanas
paga **`h` semanas de carrego**, e o bootstrap usa blocos de no mínimo `h` semanas, porque
decisões de horizonte `h` se sobrepõem e tratá-las como independentes estreitaria o IC90
artificialmente.

Medido em `scripts/33_s10_horizon_experiment.py`, walk-forward ARIMA na janela de
desenvolvimento (598 semanas, holdout **não** reaberto), 200 mil L/mês:

| h | decisões | MAE (× persistência) | disparos | precisão | economia/ano | IC90 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 196 | 0,0503 (0,98×) | 42 | 0,71 | R$ 9.285 | [3.289, 17.103] |
| 2 | 194 | 0,0958 (0,99×) | 53 | 0,72 | R$ 16.101 | [6.426, 28.266] |
| 4 | 190 | 0,1806 (1,00×) | 60 | 0,68 | R$ 26.024 | [11.265, 43.617] |
| 8 | 182 | 0,3342 (0,99×) | 55 | 0,78 | R$ 49.830 | [18.892, 77.800] |
| **12** | 174 | 0,4705 (0,97×) | 51 | **0,80** | **R$ 62.566** | [20.352, 107.508] |

**h=12 rende 6,74× o h=1, com IC90 positivo em todos os horizontes.** Com carrego de
R$ 0,002/L/semana o ganho cai para 6,46× e o IC90 continua positivo — o resultado não é
artefato de ignorar custo de estoque.

O detalhe que decide o próximo passo: **o MAE não melhora** (0,97–1,00× a persistência em
todo horizonte), mas a **precisão do gatilho sobe de 0,71 para 0,80**. O ganho não vem de
prever melhor o nível — vem de acertar mais a *direção* quando o movimento é grande o
bastante para superar o limiar. É exatamente o argumento do C4: o sinal aproveitável está na
dinâmica de médio prazo, não no nível de curto prazo.

**Ressalva:** economia anualizada em horizontes longos vem de menos episódios independentes.
Os IC90 são largos por isso, e a leitura honesta é "ordem de grandeza", não ponto.

### B3. Cenário de orçamento anual

**Por quê:** a decisão semanal economiza ~0,1% do gasto. O orçamento anual **é** o gasto.
Dado volume e horizonte, projetar a faixa de custo do ano com o intervalo calibrado é o
único número deste produto que um CFO aprova sem intermediário.

**Esforço:** baixo — o intervalo calibrado e a série estadual já existem.

**Destrava:** o produto passa a falar a língua de quem assina.

### B4. Alertas em vez de consulta

**Por quê:** o cliente não vai chamar a API toda terça. Três eventos merecem notificação: o
gatilho disparou, o spread entrou em extremo histórico, o modelo saiu da faixa de erro.

**Esforço:** baixo a médio.

**Destrava:** o produto passa a ter uso recorrente sem exigir hábito novo do cliente.

### B5. O preço que o cliente paga de verdade

**Por quê:** é o passo seguinte à mesma lógica que levou do nacional ao estadual. A média
estadual da ANP é mais próxima do cliente que a nacional, mas ainda não é a nota fiscal
dele. Ingerir as cotações do próprio cliente e modelar o spread dele contra o estado usa
**exatamente a arquitetura de decomposição que já está construída e validada**, só que um
nível abaixo.

**Esforço:** médio, e depende de o cliente fornecer os dados — o que é uma barreira
comercial e, ao mesmo tempo, um fosso: esse dado ninguém mais tem.

**Destrava:** fecha a tese comercial. O erro de base deixa de ser estimado e passa a ser
medido contra o que ele realmente pagou.

---

## C. Avanços de modelo

Pesquisa com teto conhecido. Cada item tem uma razão específica para ser tentado.

### C1. Anúncios da Petrobras com data e magnitude

**Por quê:** é a fronteira nº 1 confirmada e ainda aberta. A variação do preço de produtor
correlaciona +0,566 com a variação seguinte da revenda; na defasagem publicável restam
+0,097. A pesquisa confirmou que os anúncios são públicos e trazem data e magnitude em
R$/L, mas saem como texto de assessoria.

**Como fazer sem fragilidade:** não montar raspador contínuo. Construir **uma vez** um
conjunto histórico de anúncios a partir dos comunicados públicos, testar nos folds de
desenvolvimento, e só então decidir se vale automatizar a captura.

**Esforço:** alto, majoritariamente de curadoria.

**Destrava:** o ganho medido como potencial é de +1,7% para +13,7% sobre a persistência.

### C2. Modelo de evento em dois estágios

**Por quê:** é a arquitetura que a estrutura dos dados pede e que nunca foi testada. Dois
terços das semanas são paradas, e nelas a persistência bate todos os modelos; todo o valor
está na *magnitude* dos saltos. Um classificador "vai haver evento?" seguido de um regressor
de magnitude condicional separa esses dois problemas, em vez de pedir a um modelo só que
resolva ambos.

E o primeiro estágio já tem um preditor pronto: a pressão de repasse separa taxa de evento
de **55,6% contra 6,1%** entre quintis extremos.

**Esforço:** médio.

**Destrava:** ataca diretamente o `mae_quiet` — o gate que reprovou tanto a paridade quanto
o modelo estadual. Um modelo que fica quieto na semana parada passa onde os atuais falham.

### C3. Pooling hierárquico entre estados — **testado e fechado**

**Por quê:** são 27 séries com o mesmo mecanismo econômico e tamanhos de amostra muito
diferentes — 262 postos no RS, bem menos em estados pequenos. Um modelo hierárquico que
estima o spread de cada UF com encolhimento para a média nacional resolve isso exatamente:
estado grande fica com o próprio sinal, estado pequeno toma emprestado.

**Esforço:** médio.

**Destrava:** o item B1 com qualidade. Sem isso, servir estados pequenos significa servir
ruído.

### C4. O VS-ePL-KRLS no spread, não no preço — **testado e fechado em h=1**

**Por quê:** este é o item mais interessante para o lado acadêmico do trabalho. O
VS-ePL-KRLS foi reprovado como previsor do **nível** do preço — série dominada por saltos
raros, onde regras fuzzy evolutivas não têm o que fazer. Mas ele nunca foi testado no
**spread estadual**, que é uma série completamente diferente: pequena, estacionária, com
reversão à média e mudanças de regime — exatamente o perfil para o qual aprendizado
participativo evolutivo foi projetado.

**Esforço:** baixo. O modelo, os folds, os gates e o painel estadual já existem; é trocar o
alvo.

**Destrava:** dá à biblioteca do TCC um teste justo, num problema que corresponde às suas
hipóteses. Um resultado positivo aqui vale mais academicamente do que qualquer ganho
marginal no previsor de preço.

---

## D. O que não fazer, e por quê

Registrado para não ser tentado de novo.

| não fazer | motivo medido |
|---|---|
| trocar de arquitetura no previsor de nível | gradient boosting ficou pior que regressão linear robusta e perdeu da persistência num fold |
| ampliar a capacidade do KRLS | reduziu churn mas piorou a validação |
| abrir o holdout estadual agora | é a única leitura out-of-sample limpa que resta; sem semanas prospectivas ela vira a terceira leitura de sempre |
| vender o modelo estadual como mais preciso | ele **não** é: MAE 0,0573 contra 0,0505 do nacional, e a causa é amostral, não de modelagem |
| perseguir RMSE | tamanho amostral efetivo de ~3 nesta série; foi o que travou o projeto por meses |

---

## Ordem sugerida — revisada após a rodada

1. **B2, horizonte maior.** Deixou de ser argumento comercial e virou resultado medido:
   cinco vezes mais sinal em doze semanas do que em uma. Muda a conversa de "antecipar esta
   semana" para "planejar o trimestre" **e** dá ao C4 o teste justo que h=1 não permitiu.
2. **B5, o preço que o cliente paga.** Mesma arquitetura de decomposição, um nível abaixo,
   e é o que fecha a tese comercial com dado proprietário.
3. **C2, modelo de evento em dois estágios.** Continua sendo a arquitetura que a estrutura
   dos dados pede, e o primeiro estágio já tem preditor pronto.

Os 17 estados restantes do B1 entram quando houver demanda comercial: o caminho está
aberto e o custo é de execução, não de pesquisa.
