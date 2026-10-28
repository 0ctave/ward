# Changelog fragments

Every change that users should hear about adds **its own file** here instead of
editing `CHANGELOG.md` directly. Two PRs then never touch the same lines, so a
changelog edit can't cause a merge conflict. At release time
`tools/release.py` assembles these fragments into a dated `CHANGELOG.md`
section and deletes them.

## Adding an entry (one command)

```
python3 tools/changelog.py add -t fixed -m "night unlock no longer stalls on a slow TPM"
```

`-t` is the change **type**; the message becomes the bullet (and, by default,
the file slug). Pass `--slug name` to choose the file name yourself.

## Fragment files

Name: **`<slug>.<type>.md`** — e.g. `tpm-stall.fixed.md`.

`<type>` is one of (each maps to a `###` heading in `CHANGELOG.md`):

| type        | heading     |
|-------------|-------------|
| `added`     | Added       |
| `changed`   | Changed     |
| `fixed`     | Fixed       |
| `security`  | Security    |
| `docs`      | Docs        |
| `packaging` | Packaging   |

Content: one or more markdown bullet lines. A leading `-` or `*` is optional —
the tool normalises every non-blank line to `- ...`.

## Preview the pending section

```
python3 tools/changelog.py preview
```

`.gitkeep` and this `README.md` are ignored by the tooling and survive a
release that empties the directory.

See [RELEASING.md](../RELEASING.md) for the full maintainer runbook.
