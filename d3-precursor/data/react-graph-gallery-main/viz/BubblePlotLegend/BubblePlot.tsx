import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import { BubbleLegend } from "./BubbleLegend";
import { AxisLeft } from "./AxisLeft";
import { AxisBottom } from "./AxisBottom";

const MARGIN = { top: 30, right: 30, bottom: 80, left: 100 };
const BUBBLE_MIN_SIZE = 4;
const BUBBLE_MAX_SIZE = 40;

type BubblePlotProps = {
  width: number;
  height: number;
  data: {
    lifeExp: number;
    gdpPercap: number;
    continent: string;
    pop: number;
  }[];
};

export const BubblePlot = ({ width, height, data }: BubblePlotProps) => {
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;

  //Scales
  const yScale = useMemo(() => {
    const [min, max] = d3.extent(data.map((d) => d.lifeExp)) as [
      number,
      number
    ];
    return d3.scaleLinear().domain([min, max]).range([boundsHeight, 0]).nice();
  }, [data, height]);

  const xScale = useMemo(() => {
    const [min, max] = d3.extent(data.map((d) => d.gdpPercap)) as [
      number,
      number
    ];
    return d3.scaleLinear().domain([min, max]).range([0, boundsWidth]).nice();
  }, [data, width]);

  const groups = data
    .map((d) => d.continent)
    .filter((x, i, a) => a.indexOf(x) == i);

  const colorScale = d3
    .scaleOrdinal<string>()
    .domain(groups)
    .range(["#e0ac2b", "#e85252", "#6689c6", "#9a6fb0", "#a53253"]);

  const sizeScale = useMemo(() => {
    const [min, max] = d3.extent(data.map((d) => d.pop)) as [number, number];
    return d3
      .scaleSqrt()
      .domain([min, max])
      .range([BUBBLE_MIN_SIZE, BUBBLE_MAX_SIZE]);
  }, [data, width]);

  // Build the shapes
  const allShapes = data
    .sort((a, b) => b.pop - a.pop)
    .map((d, i) => {
      return (
        <circle
          key={i}
          r={sizeScale(d.pop)}
          cx={xScale(d.gdpPercap)}
          cy={yScale(d.lifeExp)}
          opacity={1}
          stroke={colorScale(d.continent)}
          fill={colorScale(d.continent)}
          fillOpacity={0.4}
          strokeWidth={1}
        />
      );
    });

  return (
    <div style={{ position: "relative", width, height }}>
      <svg width={width} height={height}>
        <g
          width={boundsWidth}
          height={boundsHeight}
          transform={`translate(${[MARGIN.left, MARGIN.top].join(",")})`}
        >
          {/* Y axis */}
          <AxisLeft yScale={yScale} pixelsPerTick={40} width={boundsWidth} />

          {/* X axis, use an additional translation to appear at the bottom */}
          <g transform={`translate(0, ${boundsHeight})`}>
            <AxisBottom
              xScale={xScale}
              pixelsPerTick={40}
              height={boundsHeight}
            />
          </g>
          {allShapes}
        </g>
      </svg>
      {/* Legend */}
      <div style={{ position: "absolute", right: 0, bottom: 100 }}>
        <BubbleLegend scale={sizeScale} tickNumber={4} />
      </div>
    </div>
  );
};
