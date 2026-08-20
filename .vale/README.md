# ASD-STE100 checker

[Vale](https://vale.sh/) rules that check markdown and Python comments against
the [ASD-STE100 writing rules](../docs/ste_writing_rules.md).

## Run it

```bash
brew install vale          # once; or https://vale.sh/docs/vale-cli/installation/

./.vale/ste-lint.sh                       # the files that CI checks
./.vale/ste-lint.sh --all                 # add the warnings
./.vale/ste-lint.sh docs/trainer.md       # one file
./.vale/ste-lint.sh --changed origin/development
./.vale/ste-lint.sh --list                # show the files, check nothing
```

With no path, the script checks every `README.md`, every file with the same
name as its directory, every markdown file in `docs/`, and every `*.py`.

## Which report files are checked

The rule for a new report is the shape that the agents make, and nothing else:

```
reports/<YYYY-MM-DD>_<name>/<name>.md
```

See [`../reports/REPORT_STANDARD.md`](../reports/REPORT_STANDARD.md). Use this
shape for every new experiment.

Two older shapes stay in the set for the files that came before the convention:
`<dir>/<dir>.md`, and `report.md` in an experiment directory. Do not use them
for a new report.

## The rules

| Rule | Checks | Level |
|---|---|---|
| `NotApprovedWords` | 1,180 words that ASD-STE100 does not approve, with the approved alternatives | warning |
| `SentenceLength` | rules 4.1 and 6.3: a sentence of more than 25 words | warning |
| `Semicolon` | rule 8.1: the semicolon | error |
| `LatinAbbreviations` | GR-6: `e.g.`, `i.e.`, `etc.` | error |

## What CI does

[`.github/workflows/ste-lint.yml`](../.github/workflows/ste-lint.yml):

1. Checks all the files and puts the counts in the job summary. This step never
   fails the build.
2. Fails when a file **that your PR changed** has an error.

The second step is the gate. It looks only at the files in your diff, so a rule
that the older files break cannot stop your PR. Clean the files that you touch.

Vale ignores code blocks and inline code. In a Python file it reads only the
comments and the docstrings. Code and string literals are not checked.

## Where these files come from

`styles/ASD-STE100/NotApprovedWords.yml` is generated from the ASD-STE100
dictionary. The source of truth is the `agentic-documentation/asd-ste100/vale`
directory in the workspace, together with the script that makes it. Do not edit
the generated file here. Copy a new version in when the dictionary changes.
