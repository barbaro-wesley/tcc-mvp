# Modelo estadual do Diesel B S10 — RS

**O holdout nao foi lido.** A avaliacao termina em 2024-08-11, pela mesma janela congelada por data que
protege a serie nacional.

## Por que estadual

O produto previa a media nacional de revenda da ANP. Nenhum comprador paga esse
preco: ele agrega 3.173 postos em 27 unidades da federacao, com tributo, frete e
estrutura de distribuicao diferentes em cada uma. A distancia entre a serie
modelada e a serie que o cliente enfrenta era a maior fragilidade comercial do
projeto — e ela nao se resolve com modelo melhor, se resolve com o dado certo.

A ANP publica a mesma pesquisa por estado: 704 semanas para o RS, mediana de 261 postos por semana.

## A decomposicao, e a evidencia que a escolheu

Medido sobre as semanas comuns: a variacao semanal do estado correlaciona **0,939**
com a nacional — **88% da variancia estadual e movimento do pais** — e o desvio da
variacao do *spread* e apenas 0,0264 contra 0,0769 do preco. Modelar o estado
direto joga fora o sinal nacional, que e melhor medido, e paga o ruido estadual
inteiro.

| modelo | mae | mae_quiet | mae_event | directional_accuracy | net_savings_brl | triggered | precision |
|---|---|---|---|---|---|---|---|
| persistencia | 0.059731 | 0.008536 | 0.100333 | 0.000000 | 0.000000 | 0 | None |
| rs_direto | 0.059437 | 0.017566 | 0.092645 | 0.652174 | 38,088 | 56 | 0.517857 |
| nacional+carrego | 0.057541 | 0.019326 | 0.087849 | 0.659420 | 43,015 | 53 | 0.584906 |
| nacional+spread | 0.057294 | 0.019832 | 0.087005 | 0.695652 | 46,292 | 60 | 0.600000 |
| nacional+spread+ancora | 0.057403 | 0.021652 | 0.085757 | 0.688406 | 46,892 | 59 | 0.593220 |

A decomposicao entrega **+21.5% de economia** e **+8.2 pontos de precisao de
gatilho** sobre a especificacao aplicada direto ao estado. O ganho de MAE existe mas
**nao e decidivel**: o bootstrap pareado em blocos coloca zero dentro do IC90. E o
mesmo padrao que este projeto ja encontrou duas vezes — decide melhor do que preve.

Contra a persistencia, que e o que o comprador faz hoje sem nenhum modelo, o ganho
de MAE e de 4.1% — modesto, como sempre foi nesta serie, porque em dois
tercos das semanas o preco simplesmente nao se move.

A ancora de produtor da regiao Sul nao ajudou: peso estimado de -0.0102, praticamente nulo. Coerente com a
medicao previa de que o spread de produtor explica o **nivel** do spread de revenda
(+0,25) e quase nada da variacao semanal dele (+0,06).

## O que este trabalho NAO mostrou

O modelo estadual nao e melhor que o nacional. Na mesma janela ele tem MAE maior
(0.057294 contra 0,050492) e acuracia direcional menor
(69.6% contra 74,3%). A causa e estrutural e nao tem
conserto por modelagem: 262 postos pesquisados contra 3.173.

Focar no estado nao torna a previsao melhor. Torna o numero **verdadeiro** em vez de
aproximado — e e ai que esta o valor.

## Onde o valor realmente esta

Para um comprador de 200,000 L/mes no RS:

| fonte de valor | R$/ano |
|---|---:|
| erro de orcamento por usar a serie nacional | **357,692** |
| economia da politica de antecipacao | 15,631 |

Usar a base errada custa **22.9x** o que o gatilho
semanal economiza. O produto estadual nao se vende pela previsao: vende-se por
entregar a serie que o cliente efetivamente enfrenta. Hoje o preco gaucho esta
4.8% abaixo do nacional; quem
orca pela media do pais erra para cima nessa proporcao.

## A posicao de hoje

O spread esta em R$ -0.3300/L, z = -3.12, percentil 0.4% de 704 semanas.

| faixa | limiar | semanas | episodios independentes | variacao 12s depois | positiva em |
|---|---|---|---|---|---|
| q2% | -0.274700 | 15 | 3 | 0.142000 | 1.000000 |
| q5% | -0.212550 | 36 | 4 | 0.100875 | 0.916667 |
| q10% | -0.151700 | 71 | 6 | 0.054542 | 0.745763 |

**Leia a coluna de episodios antes da de semanas.** Um spread extremo dura meses:
das 5 semanas ja vistas no nivel de hoje, a maioria e o
episodio corrente, e sobram 1 precedentes
de verdade. A direcao da reversao e sustentada nas faixas com mais episodios (78% a
89% de altas em 3 a 6 episodios distintos); a **magnitude** no extremo atual repousa
sobre pouquissimos casos e nao deve ser tratada como previsao.

## Conclusao

1. A serie estadual **e** o produto certo: e a que o cliente paga, e o erro de base
   supera a economia do gatilho por uma ordem de grandeza.
2. A decomposicao nacional-mais-spread **e** a arquitetura certa para o estado, por
   economia e precisao de gatilho, ainda que nao por MAE decidivel.
3. O modelo estadual **nao** e mais preciso que o nacional, e nao deve ser vendido
   como se fosse.
4. **Nada e promovido.** O holdout estadual continua fechado, e a confirmacao vem do
   ledger prospectivo, semana a semana.

