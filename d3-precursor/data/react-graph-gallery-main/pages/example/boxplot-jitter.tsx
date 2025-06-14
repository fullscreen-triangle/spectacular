import React from "react";
import { Layout } from "component/Layout";
import TitleAndDescription from "component/TitleAndDescription";
import ChartFamilySection from "component/ChartFamilySection";
import { CodeBlock } from "component/UI/CodeBlock";
import { ChartOrSandbox } from "component/ChartOrSandbox";
import Link from "next/link";
import { BoxplotJitterDemo } from "viz/BoxplotJitter/BoxplotJitterDemo";

const graphDescription = (
  <>
    <p>
      This tutorial is a variation around the general{" "}
      <Link href="/boxplot">introduction to boxplot with react</Link> and d3.js.
      You should probably understand the concepts described there before reading
      here.
    </p>
    <p>
      This example explains how to display all individual data points of the
      dataset on top of each box, using <b>jittering</b> to avoid overlaps.
    </p>
    <p>
      A code sandbox is provided for the final result, but explanations target
      what's different compared to an usual boxplot.
    </p>
  </>
);

export default function Home() {
  return (
    <Layout
      title="How to build a boxplot with individual data points displayed on top using jittering."
      seoDescription="A step-by-step guide to build your boxplot with jittering component. Comes with explanations, code sandboxes, and ready-to-use templates."
    >
      <TitleAndDescription
        title={
          <h1>
            Boxplot{" "}
            <span className="text-gray-600 font-light hidden sm:inline">
              with jittering
            </span>
          </h1>
        }
        description={graphDescription}
        chartType="boxplot"
      />
      {/*
      //
      // Plot and code
      //
      */}
      <h2 id="plot">Plot and code</h2>
      <p>If you are in a hurry, this is what we're trying to achieve here.🙇‍♂️</p>
      <p>
        The distribution of <b>several groups</b> is represented, with one box
        for each group. This kind of viz is called a boxplot and has its own{" "}
        <Link href="/boxplot">boxplot section</Link>.
      </p>
      <p>
        Even if powerful to summarize the distribution of a numeric variable,
        the boxplot{" "}
        <a href="https://www.data-to-viz.com/caveat/boxplot.html">has flaws</a>.
      </p>
      <p>
        It basically <b>hides the underlying distribution</b>. For instance, a
        low sample size or a bi-modal distribution are impossible to detect
        reading the boxes only.
      </p>
      <p>
        <b>Jittering</b> is a good workaround. Add all individual data points
        with low size, low opacity, and some random shift to the right or to the
        left (jitter). The underlying distribution becomes instantly available.
      </p>{" "}
      <ChartOrSandbox
        vizName={"BoxplotJitter"}
        VizComponent={BoxplotJitterDemo}
        maxWidth={600}
        height={400}
        caption={
          <span>
            Showing individual data points using jittering on top of your
            boxplot adds trust. Reader now knows you're not{" "}
            <a href="https://www.data-to-viz.com/caveat/boxplot.html">hiding</a>{" "}
            anything.
          </span>
        }
      />
      {/*
      //
      // Data
      //
      */}
      <h2 id="data">The Data</h2>
      <p>
        The dataset used here is exactly the same as the one used for the{" "}
        <Link href="/boxplot#data">simple boxplot</Link>.
      </p>
      {/*
      //
      // Jittering
      //
      */}
      <h2 id="jittering">Jittering</h2>
      <p>
        We want to show each individual data point on top of each box for
        accuracy. If the dataset is big, this would result in a <b>cluttered</b>{" "}
        figure with a lot of <b>overlap</b>.
      </p>
      <p>
        Jittering is a process that shifts each circle by a random value. It
        avoids the overlap, and allows to make the circle layer insightful.
      </p>
      <p>Here is a suggestion to implement it:</p>
      <CodeBlock code={snippetCircle} />
      <div className="full-bleed border-t h-0 bg-gray-100 mb-3 mt-24" />
      <ChartFamilySection chartFamily="distribution" />
      <div className="mt-20" />
    </Layout>
  );
}

const snippetSkeleton = `
// List of arbitrary colors
const COLORS = ["#e0ac2b", "#e85252", "#6689c6", "#9a6fb0", "#a53253"];

// List of all group names
const allGroups = data.map((group) => group.group);

// Color scale
const colorScale = d3.scaleOrdinal<string>()
  .domain(allGroups)
  .range(COLORS);
`.trim();

const snippetCircle = `
const allCircles = groupData.map((value, i) => (
  <circle
    key={i}
    cx={
      xScale.bandwidth() / 2 -
      JITTER_WIDTH / 2 +
      Math.random() * JITTER_WIDTH
    }
    cy={yScale(value)}
    r={2}
    fill="grey"
    fillOpacity={0.3}
  />
));
`.trim();
