# View specs

A view spec is the exact form a description compiles to (see
[Descriptions and specs](../guide/specs.md)). This page is section 4 of
[`spectacular-specifications.md`](https://github.com/fullscreen-triangle/spectacular/blob/rebuild/spectacular-specifications.md),
the full guide for programs and agents, included here as it stands in the repository.
Its § numbers refer to that guide: §7 is embedding with `mount()`, §8 driving the
commands, §8.6 Python.

The contract itself is `packages/viz/src/spec.ts`, and `sp.schema()` returns its JSON
Schema.

--8<-- "spectacular-specifications.md:spec"
