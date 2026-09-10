# Gates bloqueantes na promocao de release

`scripts/32_s10_promote_release.py` avanca a origem dos dados de uma release
existente. Ele nao autoriza a primeira release, uma troca de modelo primario,
um novo treino ou uma alteracao da API em execucao.

## O que bloqueia a publicacao

- Ausencia de uma release anterior `validated_candidate`, modelo primario
  desconhecido/inconsistente ou SHA-256 incorreto do artefato anterior.
- Previsao nao finita, intervalo que nao respeita
  `0 < p10 <= point <= p90`, ou uso de fallback.
- Modelo primario diferente entre o manifesto anterior, os metadados do
  bundle e a previsao atual.
- Datas divergentes entre treino, saude e previsao, alvo diferente da semana
  seguinte ou data de release que nao avanca em semanas completas.
- Arquivo de destino existente, alteracao da release anterior durante a
  validacao ou outra publicacao em andamento.
- Copia com hash diferente do original ou previsao diferente ao recarregar os
  bytes copiados. Os gates tambem sao aplicados ao modelo recarregado.

Avisos de saude do challenger, como pressao no dicionario ou no beta, continuam
registrados. Um aviso nao equivale automaticamente a falha do modelo primario.

O campo `fallback_used` continua no manifesto como diagnostico; o gate positivo
correspondente e `fallback_not_used`. O status `validated_candidate` so e
publicado depois de todos os gates bloqueantes passarem.

## Publicacao e recuperacao

A copia e o JSON sao preparados em diretorios temporarios. O script publica
primeiro o artefato validado e, por ultimo, o manifesto completo, usando hard
links exclusivos no filesystem de cada destino. Um destino existente nunca e
sobrescrito. Se a publicacao do manifesto falhar normalmente, o script remove
apenas o artefato que acabou de publicar.

O lock `.promotion.lock` serializa publicacoes deste comando. Em uma interrupcao
abrupta do processo ou do sistema, podem restar temporarios, um lock ou um
artefato sem manifesto. Isso nao e uma release publicada: o operador deve
confirmar que nao existe processo ativo e inspecionar esses arquivos antes da
recuperacao. Nao ha promessa de transacao atomica entre dois diretorios em caso
de queda do sistema. O filesystem deve suportar hard links; caso contrario,
a operacao falha sem recorrer a sobrescrita.

`--force` foi desativado para preservar a imutabilidade. Uma primeira release ou
troca de modelo precisa do fluxo de aprovacao correspondente, fora deste
comando de atualizacao de dados.

```bash
python scripts/32_s10_promote_release.py --dry-run
python -m pytest tests/test_promote_release.py
```

O dry-run executa os gates e o roundtrip em um temporario do sistema, sem
gravar nos destinos de release. A publicacao efetiva continua sendo uma
operacao separada. Os testes usam modelos simulados, hashes e arquivos reais
temporarios, sem carregar pickles nem promover artefatos de producao.
