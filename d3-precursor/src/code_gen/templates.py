from string import Template
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import os


class CodeTemplate:
    """Base class for code templates."""
    
    def __init__(self, template: str, name: str, description: str):
        """Initialize the code template.
        
        Args:
            template: The template string with placeholders.
            name: The name of the template.
            description: The description of the template.
        """
        self.template_str = template
        self.template = Template(template)
        self.name = name
        self.description = description
    
    def apply(self, customizations: Dict[str, Any]) -> str:
        """Apply the template with the given customizations.
        
        Args:
            customizations: Dictionary of values for placeholders.
            
        Returns:
            The filled template.
        """
        # Prepare values for all possible placeholders
        values = {}
        
        # Apply customizations
        values.update(customizations)
        
        # Fill the template
        return self.template.safe_substitute(values)


class VisualizationTemplate(CodeTemplate):
    """Template for D3.js visualizations."""
    
    def __init__(
        self,
        template: str,
        name: str,
        description: str,
        sample_data: str,
        sample_time_data: Optional[str] = None,
        sample_categorical_data: Optional[str] = None,
        sample_network_data: Optional[str] = None,
        sample_geo_data: Optional[str] = None
    ):
        """Initialize the visualization template.
        
        Args:
            template: The template string with placeholders.
            name: The name of the template.
            description: The description of the template.
            sample_data: Sample data for the visualization.
            sample_time_data: Sample time series data.
            sample_categorical_data: Sample categorical data.
            sample_network_data: Sample network data.
            sample_geo_data: Sample geographic data.
        """
        super().__init__(template, name, description)
        self.sample_data = sample_data
        self.sample_time_data = sample_time_data or sample_data
        self.sample_categorical_data = sample_categorical_data or sample_data
        self.sample_network_data = sample_network_data or sample_data
        self.sample_geo_data = sample_geo_data or sample_data
    
    def apply(self, data_variable: str, customizations: Dict[str, Any] = None) -> str:
        """Apply the template with the given data and customizations.
        
        Args:
            data_variable: The data variable definition.
            customizations: Additional customizations for the template.
            
        Returns:
            The filled template.
        """
        customizations = customizations or {}
        customizations["data"] = data_variable
        
        # Set default values for common placeholders
        if "width" not in customizations:
            customizations["width"] = 800
        if "height" not in customizations:
            customizations["height"] = 500
        if "margin" not in customizations:
            customizations["margin"] = {"top": 20, "right": 20, "bottom": 30, "left": 40}
        if "colorScale" not in customizations:
            customizations["colorScale"] = "d3.scaleOrdinal(d3.schemeCategory10)"
        if "title" not in customizations:
            customizations["title"] = ""
        
        # Format margin as JavaScript object if it's a dict
        if isinstance(customizations["margin"], dict):
            margin_dict = customizations["margin"]
            customizations["margin"] = (
                f"{{top: {margin_dict['top']}, right: {margin_dict['right']}, "
                f"bottom: {margin_dict['bottom']}, left: {margin_dict['left']}}}"
            )
        
        return super().apply(customizations)


class ComponentTemplate(CodeTemplate):
    """Template for D3.js components."""
    
    def __init__(
        self,
        template: str,
        name: str,
        description: str,
        default_function_name: str,
        dependencies: List[str] = None
    ):
        """Initialize the component template.
        
        Args:
            template: The template string with placeholders.
            name: The name of the template.
            description: The description of the template.
            default_function_name: The default function name for the component.
            dependencies: List of dependencies for the component.
        """
        super().__init__(template, name, description)
        self.default_function_name = default_function_name
        self.dependencies = dependencies or []
    
    def apply(self, customizations: Dict[str, Any] = None) -> str:
        """Apply the template with the given customizations.
        
        Args:
            customizations: Customizations for the template.
            
        Returns:
            The filled template.
        """
        customizations = customizations or {}
        
        # Set default values for common placeholders
        if "functionName" not in customizations:
            customizations["functionName"] = self.default_function_name
        
        # Add dependencies as a comment
        if self.dependencies and "dependencies" not in customizations:
            dependencies_str = ", ".join(self.dependencies)
            customizations["dependencies"] = f"// Dependencies: {dependencies_str}"
        elif "dependencies" not in customizations:
            customizations["dependencies"] = ""
        
        return super().apply(customizations)


class TemplateLibrary:
    """Library of code templates for D3.js."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """Initialize the template library.
        
        Args:
            template_dir: Directory containing template files. If None, uses built-in templates.
        """
        self.visualization_templates = {}
        self.component_templates = {}
        
        # Load built-in templates
        self._load_built_in_templates()
        
        # Load templates from directory if provided
        if template_dir:
            self._load_templates_from_dir(Path(template_dir))
    
    def _load_built_in_templates(self):
        """Load the built-in templates."""
        # Bar chart template
        bar_chart_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .bar {
            fill: steelblue;
            transition: fill 0.3s;
        }
        .bar:hover {
            fill: #69b3a2;
        }
        .axis--x path {
            display: none;
        }
        .axis text {
            font-size: 12px;
        }
        .title {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
    </style>
</head>
<body>
    <div id="chart"></div>

    <script>
        // Data
        ${data}

        // Set up dimensions and margins
        const margin = ${margin};
        const width = ${width} - margin.left - margin.right;
        const height = ${height} - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(\${margin.left},\${margin.top})`);

        // Add title if provided
        if ("${title}") {
            svg.append("text")
                .attr("class", "title")
                .attr("x", width / 2)
                .attr("y", -margin.top / 2)
                .attr("text-anchor", "middle")
                .text("${title}");
        }

        // Set up scales
        const x = d3.scaleBand()
            .domain(data.map(d => d.name))
            .range([0, width])
            .padding(0.2);

        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .nice()
            .range([height, 0]);

        // Add x-axis
        svg.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", `translate(0,\${height})`)
            .call(d3.axisBottom(x));

        // Add y-axis
        svg.append("g")
            .attr("class", "axis axis--y")
            .call(d3.axisLeft(y).ticks(10));

        // Add bars
        svg.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", d => x(d.name))
            .attr("y", d => y(d.value))
            .attr("width", x.bandwidth())
            .attr("height", d => height - y(d.value));
    </script>
</body>
</html>"""

        self.visualization_templates["bar_chart"] = VisualizationTemplate(
            template=bar_chart_template,
            name="Bar Chart",
            description="A simple vertical bar chart for categorical data.",
            sample_data="""const data = [
  { name: "A", value: 10 },
  { name: "B", value: 20 },
  { name: "C", value: 30 },
  { name: "D", value: 40 },
  { name: "E", value: 25 }
];"""
        )
        
        # Line chart template
        line_chart_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .line {
            fill: none;
            stroke: steelblue;
            stroke-width: 2;
        }
        .dot {
            fill: steelblue;
            stroke: white;
        }
        .axis text {
            font-size: 12px;
        }
        .title {
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
    </style>
</head>
<body>
    <div id="chart"></div>

    <script>
        // Data
        ${data}

        // Parse dates
        const parseDate = d3.timeParse("%Y-%m-%d");
        data.forEach(d => {
            if (typeof d.date === 'string') {
                d.date = parseDate(d.date);
            }
        });

        // Set up dimensions and margins
        const margin = ${margin};
        const width = ${width} - margin.left - margin.right;
        const height = ${height} - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(\${margin.left},\${margin.top})`);

        // Add title if provided
        if ("${title}") {
            svg.append("text")
                .attr("class", "title")
                .attr("x", width / 2)
                .attr("y", -margin.top / 2)
                .attr("text-anchor", "middle")
                .text("${title}");
        }

        // Set up scales
        const x = d3.scaleTime()
            .domain(d3.extent(data, d => d.date))
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .nice()
            .range([height, 0]);

        // Add x-axis
        svg.append("g")
            .attr("class", "axis axis--x")
            .attr("transform", `translate(0,\${height})`)
            .call(d3.axisBottom(x));

        // Add y-axis
        svg.append("g")
            .attr("class", "axis axis--y")
            .call(d3.axisLeft(y));

        // Add line
        svg.append("path")
            .datum(data)
            .attr("class", "line")
            .attr("d", d3.line()
                .x(d => x(d.date))
                .y(d => y(d.value))
            );

        // Add dots
        svg.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .attr("cx", d => x(d.date))
            .attr("cy", d => y(d.value))
            .attr("r", 4);
    </script>
</body>
</html>"""

        self.visualization_templates["line_chart"] = VisualizationTemplate(
            template=line_chart_template,
            name="Line Chart",
            description="A line chart for time series data.",
            sample_data="""const data = [
  { date: "2023-01-01", value: 10 },
  { date: "2023-02-01", value: 20 },
  { date: "2023-03-01", value: 15 },
  { date: "2023-04-01", value: 25 },
  { date: "2023-05-01", value: 22 },
  { date: "2023-06-01", value: 30 }
];""",
            sample_time_data="""const data = [
  { date: "2023-01-01", value: 10 },
  { date: "2023-02-01", value: 20 },
  { date: "2023-03-01", value: 15 },
  { date: "2023-04-01", value: 25 },
  { date: "2023-05-01", value: 22 },
  { date: "2023-06-01", value: 30 }
];"""
        )
        
        # Tooltip component template
        tooltip_template = """${dependencies}

function ${functionName}(selection) {
    // Create tooltip div if it doesn't exist
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("background-color", "white")
        .style("border", "1px solid #ddd")
        .style("border-radius", "4px")
        .style("padding", "8px")
        .style("pointer-events", "none")
        .style("opacity", 0)
        .style("transition", "opacity 0.3s");
    
    // Apply to selected elements
    selection
        .on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", 0.9);
            
            // Format tooltip content based on data
            let content = "";
            if (typeof d === "object") {
                // Try to format based on common data patterns
                if (d.name && d.value !== undefined) {
                    content = `<strong>\${d.name}</strong>: \${d.value}`;
                } else if (d.date && d.value !== undefined) {
                    const dateStr = d.date instanceof Date 
                        ? d.date.toLocaleDateString() 
                        : d.date;
                    content = `<strong>\${dateStr}</strong>: \${d.value}`;
                } else {
                    // Generic object display
                    content = Object.entries(d)
                        .map(([key, value]) => `<strong>\${key}</strong>: \${value}`)
                        .join("<br>");
                }
            } else {
                // Simple value
                content = d;
            }
            
            tooltip.html(content)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mousemove", function(event) {
            tooltip
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });
    
    return selection;
}

// Usage example:
// d3.selectAll(".interactive-element").call(${functionName});"""

        self.component_templates["tooltip"] = ComponentTemplate(
            template=tooltip_template,
            name="Tooltip",
            description="A reusable tooltip component for D3.js visualizations.",
            default_function_name="tooltip",
            dependencies=["d3.js"]
        )
        
        # Legend component template
        legend_template = """${dependencies}

function ${functionName}(selection, {
    color,
    title = "",
    tickSize = 6,
    width = 320, 
    height = 44 + tickSize,
    marginTop = 18,
    marginRight = 0,
    marginBottom = 16 + tickSize,
    marginLeft = 0,
    ticks = width / 64,
    tickFormat,
    tickValues
} = {}) {
    
    function ramp(color, n = 256) {
        const canvas = document.createElement("canvas");
        canvas.width = n;
        canvas.height = 1;
        const context = canvas.getContext("2d");
        for (let i = 0; i < n; ++i) {
            context.fillStyle = color(i / (n - 1));
            context.fillRect(i, 0, 1, 1);
        }
        return canvas;
    }
    
    let svg = selection.selectAll("svg").data([null]);
    
    const g = svg.enter().append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("overflow", "visible")
        .style("display", "block")
      .append("g")
        .attr("transform", `translate(\${marginLeft}, \${marginTop})`)
      .merge(svg.select("g"));
    
    let x;
    
    // Continuous legend
    if (color.interpolate) {
        const n = Math.min(color.domain().length, color.range().length);
        
        x = color.copy().rangeRound(d3.quantize(d3.interpolate(0, width), n));
        
        g.append("image")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width)
            .attr("height", height - marginTop - marginBottom)
            .attr("preserveAspectRatio", "none")
            .attr("xlink:href", ramp(color.copy().domain(d3.quantize(d3.interpolate(0, 1), n))).toDataURL());
    } 
    // Sequential legend
    else if (color.interpolator) {
        x = Object.assign(color.copy()
            .interpolator(d3.interpolateRound(0, width)),
            {range() { return [0, width]; }});
        
        g.append("image")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width)
            .attr("height", height - marginTop - marginBottom)
            .attr("preserveAspectRatio", "none")
            .attr("xlink:href", ramp(color.interpolator()).toDataURL());
        
        // Ticks
        if (!tickValues) {
            const n = Math.round(ticks + 1);
            tickValues = d3.range(n).map(i => d3.quantile(color.domain(), i / (n - 1)));
        }
    } 
    // Threshold legend
    else if (color.invertExtent) {
        const thresholds = color.thresholds ? color.thresholds() 
            : color.quantiles ? color.quantiles() 
            : color.domain();
        
        const thresholdFormat = tickFormat || (d => d);
        
        x = d3.scaleLinear()
            .domain([-1, color.range().length - 1])
            .rangeRound([0, width]);
        
        g.append("g")
            .selectAll("rect")
            .data(color.range())
            .join("rect")
            .attr("x", (d, i) => x(i - 1))
            .attr("y", 0)
            .attr("width", (d, i) => x(i) - x(i - 1))
            .attr("height", height - marginTop - marginBottom)
            .attr("fill", d => d);
        
        tickValues = d3.range(thresholds.length);
        tickFormat = i => thresholdFormat(thresholds[i], i);
    } 
    // Ordinal legend
    else {
        x = d3.scaleBand()
            .domain(color.domain())
            .rangeRound([0, width]);
        
        g.append("g")
            .selectAll("rect")
            .data(color.domain())
            .join("rect")
            .attr("x", x)
            .attr("y", 0)
            .attr("width", Math.max(0, x.bandwidth() - 1))
            .attr("height", height - marginTop - marginBottom)
            .attr("fill", color);
    }
    
    // Add title
    if (title) {
        g.append("text")
            .attr("class", "title")
            .attr("x", width / 2)
            .attr("y", -6)
            .attr("text-anchor", "middle")
            .attr("fill", "currentColor")
            .text(title);
    }
    
    // Add ticks
    if (tickValues && x) {
        g.append("g")
            .attr("transform", `translate(0,\${height - marginTop - marginBottom})`)
            .call(d3.axisBottom(x)
                .ticks(ticks, typeof tickFormat === "string" ? tickFormat : undefined)
                .tickFormat(typeof tickFormat === "function" ? tickFormat : undefined)
                .tickSize(tickSize)
                .tickValues(tickValues))
            .call(g => g.select(".domain").remove())
            .call(g => g.select(".title").remove());
    }
    
    return selection;
}

// Usage example:
// d3.select("#legend")
//   .call(${functionName}, {
//     color: colorScale,
//     title: "Legend Title"
//   });"""

        self.component_templates["legend"] = ComponentTemplate(
            template=legend_template,
            name="Legend",
            description="A reusable legend component for D3.js visualizations.",
            default_function_name="legend",
            dependencies=["d3.js"]
        )
    
    def _load_templates_from_dir(self, template_dir: Path):
        """Load templates from a directory.
        
        Args:
            template_dir: Directory containing template files.
        """
        # Check if directory exists
        if not template_dir.exists() or not template_dir.is_dir():
            return
        
        # Load visualization templates
        vis_dir = template_dir / "visualizations"
        if vis_dir.exists() and vis_dir.is_dir():
            for file_path in vis_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        template_data = json.load(f)
                    
                    # Create visualization template
                    template = VisualizationTemplate(
                        template=template_data["template"],
                        name=template_data["name"],
                        description=template_data["description"],
                        sample_data=template_data["sample_data"],
                        sample_time_data=template_data.get("sample_time_data"),
                        sample_categorical_data=template_data.get("sample_categorical_data"),
                        sample_network_data=template_data.get("sample_network_data"),
                        sample_geo_data=template_data.get("sample_geo_data")
                    )
                    
                    # Add to library
                    self.visualization_templates[file_path.stem] = template
                except Exception as e:
                    print(f"Error loading template from {file_path}: {e}")
        
        # Load component templates
        comp_dir = template_dir / "components"
        if comp_dir.exists() and comp_dir.is_dir():
            for file_path in comp_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        template_data = json.load(f)
                    
                    # Create component template
                    template = ComponentTemplate(
                        template=template_data["template"],
                        name=template_data["name"],
                        description=template_data["description"],
                        default_function_name=template_data["default_function_name"],
                        dependencies=template_data.get("dependencies", [])
                    )
                    
                    # Add to library
                    self.component_templates[file_path.stem] = template
                except Exception as e:
                    print(f"Error loading template from {file_path}: {e}")
    
    def save_templates_to_dir(self, template_dir: Union[str, Path]):
        """Save all templates to a directory.
        
        Args:
            template_dir: Directory to save templates to.
        """
        template_dir = Path(template_dir)
        
        # Create directories
        vis_dir = template_dir / "visualizations"
        comp_dir = template_dir / "components"
        
        vis_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization templates
        for name, template in self.visualization_templates.items():
            template_data = {
                "template": template.template_str,
                "name": template.name,
                "description": template.description,
                "sample_data": template.sample_data,
                "sample_time_data": template.sample_time_data,
                "sample_categorical_data": template.sample_categorical_data,
                "sample_network_data": template.sample_network_data,
                "sample_geo_data": template.sample_geo_data
            }
            
            with open(vis_dir / f"{name}.json", 'w') as f:
                json.dump(template_data, f, indent=2)
        
        # Save component templates
        for name, template in self.component_templates.items():
            template_data = {
                "template": template.template_str,
                "name": template.name,
                "description": template.description,
                "default_function_name": template.default_function_name,
                "dependencies": template.dependencies
            }
            
            with open(comp_dir / f"{name}.json", 'w') as f:
                json.dump(template_data, f, indent=2) 