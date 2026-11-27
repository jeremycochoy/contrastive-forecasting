# NeurIPS 2024 Paper Guidelines

Reference notes extracted from the official NeurIPS 2024 LaTeX template
(`neurips_2024.tex` / `neurips_2024.sty`). Keep `main.tex` slim; consult
this file when you need to recall the formatting, citation, or
disclosure rules.

---

## 1. Style file options

`\usepackage{neurips_2024}` accepts these optional arguments:

- *(no option)* — anonymized submission with line numbers (review version).
- `[preprint]` — non-anonymized version with the footer
  "Preprint. Work in progress." Use this for arXiv.
- `[final]` — camera-ready, non-anonymized. **Only** for accepted papers.
- `[nonatbib]` — skip loading `natbib` (use if it clashes with another
  package).

Pass options to `natbib` (before loading the style):

```latex
\PassOptionsToPackage{numbers, compress}{natbib}
\usepackage{neurips_2024}
```

## 2. Page limits and overall layout

- **Up to 9 pages**, including figures. Acknowledgments and references
  may overflow onto extra pages.
- Text rectangle: 5.5 in (33 picas) wide × 9 in (54 picas) long.
  Left margin 1.5 in (9 picas).
- Body: 10 pt Times New Roman, 11 pt leading.
- Paragraphs: separated by ½ line space (5.5 pt), no indentation.
- Title: 17 pt, initial caps / lower case, bold, centered between two
  horizontal rules (top 4 pt, bottom 1 pt). ¼ in space above and below.
- Pages must start 1 in (6 picas) from the top.
- Submit in **US Letter**, not A4.
- **Do not modify** any formatting parameters in the style files.

## 3. Headings

- All lower case (except the first word and proper nouns), flush left,
  bold.
- Level 1 (`\section`): 12 pt.
- Level 2 (`\subsection`): 10 pt.
- Level 3 (`\subsubsection`): 10 pt.
- `\paragraph` is also available — bold, flush left, inline with the
  text, followed by 1 em of space.

## 4. Authors

```latex
\author{%
  First Author\thanks{Footnote for extra info — NOT for funding.} \\
  Affiliation \\
  Address \\
  \texttt{email}
  \And  % LaTeX picks line breaks
  Coauthor \\ ...
  \AND  % force a line break
  Coauthor \\ ...
}
```

Use `\And` between authors to let LaTeX choose breaks; switch to `\AND`
to force one when the layout looks wrong.

In the camera-ready version, names are bold and centered above the
corresponding address. Lead author goes left-most.

## 5. Citations

- `natbib` is loaded by default.
- Use `\citet{key}` for inline ("Hasselmo et al. (1995) investigated…")
  and `\citep{key}` for parenthetical.
- Author/year or numeric — pick one and stay consistent.
- During anonymous review, refer to your own work in the third person
  ("In the previous work of Jones et al. [4]…", **not** "In our previous
  work [4]…"). For unpublished self-citations, use "A. Anonymous" and
  bundle the anonymized paper with the supplement.
- natbib docs: <http://mirrors.ctan.org/macros/latex/contrib/natbib/natnotes.pdf>

## 6. Footnotes

- Use sparingly.
- Numbered, placed at the bottom of the page where they appear.
- Always place footnote markers **after** punctuation marks.

## 7. Figures

- Captions go **after** the figure, lower case (except first word /
  proper nouns), numbered consecutively.
- One line space before the caption and one after the figure.
- Figures must be legible in both color and black/white print.
- Use `\includegraphics` from `graphicx`; specify width as a fraction of
  `\linewidth`:

```latex
\usepackage[pdftex]{graphicx}
\includegraphics[width=0.8\linewidth]{myfile.pdf}
```

## 8. Tables

- Centered, neat, legible.
- Title goes **before** the table, one line space before/after.
- Lower case (except first word / proper nouns), numbered consecutively.
- **No vertical rules.** Use `booktabs` (`\toprule`, `\midrule`,
  `\bottomrule`, `\cmidrule`).

## 9. Math

- Use LaTeX (or AMSTeX) commands for unnumbered display math, **not**
  bare TeX `$$ … $$` — the latter breaks line numbering during review.

## 10. PDF / fonts

- Generate PDFs with `pdflatex`.
- Type 1 or embedded TrueType fonts only. (Type 3 / non-embedded TrueType
  will be rejected.)
- Check fonts via Acrobat's "Document Properties → Fonts" or the
  `pdffonts` CLI.
- Avoid `xfig` "patterned" shapes (bitmap fonts) — use solid shapes.
- Avoid the `\bbold` package; use `amsfonts` (or `amssymb`) and
  `\mathbb{R}`, `\mathbb{N}`, `\mathbb{C}` instead.

## 11. Acknowledgments

- Use the `ack` environment from the style file. It is hidden in the
  anonymized submission and shown only in the final version.
- **Required** to declare funding sources and competing interests.
- See <https://neurips.cc/Conferences/2024/PaperInformation/FundingDisclosure>.

## 12. References

- Section heading: unnumbered first-level (`\section*{References}`).
- Any consistent style is acceptable.
- May be set in `\small` (9 pt).
- Does **not** count against the 9-page limit.

## 13. Appendix / supplemental

- Mark with `\appendix`; use `\section{...}` afterwards for each
  appendix item.
- All supplemental material **should be in the main submission**
  (appended after references).

---

## 14. NeurIPS Paper Checklist

The checklist is **mandatory for the conference submission** — papers
without it are desk-rejected. Place it after references and any
supplemental material. It does not count against the page limit. Use the
provided macros: `\answerYes{}`, `\answerNo{}`, `\answerNA{}`,
`\answerTODO{}`, `\justificationTODO{}`. Provide a 1–2 sentence
justification after each answer (even for NA).

> An arXiv preprint does not require this checklist — it's a
> conference-submission artifact.

### Checklist questions

1. **Claims** — Do the main claims made in the abstract and introduction
   accurately reflect the paper's contributions and scope?
2. **Limitations** — Does the paper discuss the limitations of the work
   performed by the authors?
3. **Theory Assumptions and Proofs** — For each theoretical result, does
   the paper provide the full set of assumptions and a complete (and
   correct) proof?
4. **Experimental Result Reproducibility** — Does the paper fully
   disclose all the information needed to reproduce the main
   experimental results, to the extent that it affects the main claims
   and/or conclusions?
5. **Open access to data and code** — Does the paper provide open access
   to the data and code, with sufficient instructions to faithfully
   reproduce the main experimental results?
6. **Experimental Setting/Details** — Does the paper specify all the
   training and test details (data splits, hyperparameters, choice
   procedure, optimizer type, …) needed to understand the results?
7. **Experiment Statistical Significance** — Does the paper report error
   bars suitably and correctly defined, or other appropriate information
   about statistical significance?
8. **Experiments Compute Resources** — Does the paper provide sufficient
   information on the compute resources (worker type, memory, runtime)
   needed to reproduce each experiment?
9. **Code of Ethics** — Does the research conform with the NeurIPS Code
   of Ethics (<https://neurips.cc/public/EthicsGuidelines>)?
10. **Broader Impacts** — Does the paper discuss both potential positive
    and negative societal impacts of the work?
11. **Safeguards** — Does the paper describe safeguards for responsible
    release of high-misuse-risk data or models?
12. **Licenses for existing assets** — Are creators / original owners of
    used assets credited and is licensing properly mentioned and
    respected?
13. **New Assets** — Are new assets introduced in the paper well
    documented and is documentation provided alongside the assets?
14. **Crowdsourcing and Research with Human Subjects** — Does the paper
    include the full text of participant instructions, screenshots (if
    applicable), and compensation details?
15. **IRB Approvals (or equivalent) for Research with Human Subjects** —
    Does the paper describe risks to participants, disclosure, and IRB
    approval status?

For each item, use best judgment in answering Yes / No / NA and add a
short justification. Answering "No" is acceptable when properly
justified — it is not, by itself, grounds for rejection. Reviewers are
asked **not** to penalize honest acknowledgement of limitations.

### Checklist macros (style file)

```latex
\answerYes{}            % point to the section(s) supporting the claim
\answerNo{}             % short justification required
\answerNA{}             % short justification required
\answerTODO{}           % placeholder
\justificationTODO{}    % placeholder
```

---

## 15. Style files

- `neurips_2024.sty` — the only supported style file. Do not tweak it
  (grounds for rejection).
- `neurips_2024.pdf` — the original rendered formatting-instructions PDF
  (not kept in this repo; regenerate from the upstream zip if needed).
- Source of truth: <http://www.neurips.cc/>.
