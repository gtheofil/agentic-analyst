# Groundedness

You are evaluating whether a report's claims are supported by the evidence it
cites. You will be given the report and the full list of findings available to
its author, each with the verbatim quote and the source URL it came from.

Citations look like `[F3]`, referring to finding 3 in the list.

The question is **not** whether a citation resolves — that is checked by code
before you see this. The question is whether the cited finding *says what the
sentence claims it says*. Three failures matter, in descending order:

1. **Misattribution** — the sentence makes a claim the quoted evidence does not
   support. A quote about the UK market cited for a claim about Ireland; a
   figure for 2019 cited as current; a single company's experience cited as an
   industry pattern.
2. **Uncited load-bearing claims** — a specific number, date or named fact
   carrying part of the argument with no citation at all.
3. **Overstatement** — the evidence supports a weaker version of the claim than
   the one written ("suggests" reported as "shows").

General reasoning and clearly-marked inference do not need citations. Penalising
those produces reports that cite every sentence and argue nothing.

## Anchors

- **10** — Every load-bearing claim is cited and every citation supports what it
  is attached to. Uncertainty in the evidence is carried into the prose.
- **8** — All claims cited and supported; one instance of the prose being a
  little stronger than its evidence.
- **6** — One load-bearing claim uncited, or one citation that only partly
  supports its sentence.
- **4** — Several uncited specifics, or a citation that does not support its
  claim at all.
- **2** — Most specifics are uncited, or the citations are decorative — attached
  to sentences they have little to do with.
- **1** — The report's claims are unrelated to the evidence it cites.

Name the single worst instance you found in your justification, quoting the
sentence and the finding id. If you cannot find one, say so explicitly.
