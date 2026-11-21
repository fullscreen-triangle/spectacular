# Description keys

Every key a description accepts, with its other names. Keys are compared after removing
case, spaces, dashes and underscores, so `x-axis`, `X axis`, `x_axis` and `xAxis` are all
`x`. Writing two names for one key in the same object is a problem.

These tables are checked against the compiler (`crates/spectacular/src/describe.rs`) by
the package's tests, so they list exactly what it accepts.

## The whole description

<!-- aliases: TOP -->
| Key | Also | Value |
|---|---|---|
| `title` | `heading` | text above the charts |
| `description` | `about`, `notes` | text kept with the spec, not drawn |
| `data` | `file`, `csv` | a file path or URL, used when no table is passed; or a spec `data` object |
| `theme` | `style` | `"auto"`, `"light"`, `"dark"` or `"paper"` |
| `columns` | `grid`, `per-column`, `charts-per-row`, `per-row` | charts per row, 1 to 12 |
| `layout` | | a spec `layout` object, `{"columns": n}` |
| `labels` | `titles`, `axis-labels`, `names` | column → axis label |
| `categorical` | `nominal`, `categories`, `categorical-columns`, `groups` | columns to treat as categories |
| `domains` | `domain`, `levels`, `category-order`, `orders` | column → its categories in order |
| `palettes` | `field-colors`, `field-colours`, `category-colors`, `category-colours` | column → a list of colours, or category → colour |
| `key` | `row-key`, `key-column`, `id-column` | the column whose values name rows |
| `figure` | | `{"format", "width", "unit", "dpi"}`, the size `render` uses; `format` is required |
| `output` | | a spec `output` object |
| `fields` | | spec `fields`, merged with what the other keys declare |
| `version` | | `"spectacular/v0"` |
| `$schema` | `schema` | ignored; for editors |

The list of charts goes under one of these keys:

<!-- aliases: LISTS -->
| Key | Also | Value |
|---|---|---|
| `charts` | `views`, `plots`, `graphs`, `figures`, `panels` | a list of charts, or one chart object |

Without one of them, the description is a single chart (its keys beside the ones above),
or a list of charts at the top.

## A chart

<!-- aliases: CHART -->
| Key | Also | Value |
|---|---|---|
| `x` | `x-axis`, `horizontal`, `across` | a column |
| `y` | `y-axis`, `vertical`, `up` | a column |
| `z` | `z-axis`, `depth` | a column |
| `field` | `column`, `value`, `values` | a column, for one-column charts (the same as `x`) |
| `from` | `source`, `start` | the column of edge sources |
| `to` | `target`, `end` | the column of edge targets |
| `low` | `lower`, `lower-bound`, `low-end`, `min`, `minimum` | the column of interval low ends |
| `high` | `upper`, `upper-bound`, `high-end`, `max`, `maximum` | the column of interval high ends |
| `color` | `colour`, `color-by`, `colour-by`, `hue`, `fill` | a column, a colour, or a list of colours |
| `palette` | `colors`, `colours` | a list of colours for the categories |
| `title` | `name`, `label`, `caption` | the chart's title |
| `type` | `chart`, `kind`, `as`, `chart-type`, `plot`, `mark` | a [chart type](#chart-types) |
| `bins` | `buckets` | a histogram's bin count; a density's `[x, y]` grid |
| `size` | `point-size`, `dot-size` | point diameter in pixels |
| `height` | | plot height in pixels |
| `span` | | grid columns the chart spans |
| `wide` | `full-width` | `true`: span the whole row |
| `brush` | `select`, `selectable`, `interactive` | `false` (or `"none"`) to stop the chart filtering others |
| `order` | `sort` | an arc chart's node order: `"count"`, `"name"` or `"data"` |
| `scale` | `scales`, `axis` | an interval's scale: [shared or per band](#interval-scale) |
| `id` | | the chart's id in selections; by default made from its title |

## Chart types

`type` takes any of these names:

<!-- aliases: CHART_TYPES -->
| Type | Also |  |
|---|---|---|
| `scatter` | `points`, `point`, `dots`, `xy`, `scatterplot` | |
| `histogram` | `hist`, `distribution` | |
| `bars` | `bar`, `bar-chart`, `counts`, `count`, `categories` | |
| `density` | `heatmap`, `heat`, `hexbin`, `2d-histogram`, `histogram-2d` | |
| `scatter3d` | `3d`, `points3d`, `cloud`, `point-cloud` | |
| `strip` | `strip-plot`, `jitter`, `beeswarm`, `swarm` | |
| `matrix` | `grid`, `crosstab`, `contingency`, `table` | |
| `interval` | `intervals`, `range`, `ranges`, `error-bar`, `error-bars`, `forest` | |
| `arc` | `arcs`, `arc-diagram` | |
| `sankey` | `flow`, `flows` | |
| `network3d` | `network`, `graph`, `graph3d` | |
| `layers` | `layered`, `dag`, `pipeline`, `steps`, `flowchart` | |

## Interval scale

| `scale` | Also | Means |
|---|---|---|
| `"shared"` | `"one"`, `"common"`, `"same"` | one axis for every band (the default); the chart brushes |
| `"each"` | `"label"`, `"own"`, `"separate"`, `"per-label"`, `"per-band"`, `"band"` | each band on its own scale; no brush |

## How the chart type is chosen

When a chart has no `type`, the first rule that matches decides:

1. `low` or `high` → `interval`
2. `from` or `to` → `sankey`
3. `x`, `y` and `z` → `scatter3d`
4. `x` and `y`, both numbers → `scatter`; `density` instead above 200,000 rows when not coloured by a column
5. `x` and `y`, both categories → `matrix`
6. `x` and `y`, one of each → `strip`
7. one of `field`, `x`, `y`: a number → `histogram`; a category → `bars`

A column is a category when it is not numeric, or is named in `categorical`, `domains`
or `palettes`.
