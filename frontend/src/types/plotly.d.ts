// Prevents tsc -b from complaining about missing types for plotly.js-dist-min

declare module "plotly.js-dist-min" {
  const Plotly: {
    react: (
      root: HTMLElement,
      data: Record<string, unknown>[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ) => Promise<HTMLElement>;
    purge: (root: HTMLElement) => void;
  };
  export default Plotly;
}
