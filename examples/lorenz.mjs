// node examples/lorenz.mjs > examples/lorenz.csv
// The Lorenz system (sigma 10, rho 28, beta 8/3) integrated with RK4, sampled every
// 0.01 time units. "lobe" is which wing the trajectory is on (sign of x).
const [sigma, rho, beta, dt, steps] = [10, 28, 8 / 3, 0.002, 150_000];
const f = ([x, y, z]) => [sigma * (y - x), x * (rho - z) - y, x * y - beta * z];
let s = [1, 1, 1];
const lines = ["t,x,y,z,speed,lobe"];
for (let i = 1; i <= steps; i++) {
  const k1 = f(s);
  const k2 = f(s.map((v, j) => v + (dt / 2) * k1[j]));
  const k3 = f(s.map((v, j) => v + (dt / 2) * k2[j]));
  const k4 = f(s.map((v, j) => v + dt * k3[j]));
  s = s.map((v, j) => v + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
  if (i % 5 === 0 && i * dt > 1) {
    const d = f(s);
    const speed = Math.hypot(...d);
    lines.push([(i * dt).toFixed(2), ...s.map((v) => v.toFixed(4)), speed.toFixed(3), s[0] < 0 ? "left" : "right"].join(","));
  }
}
process.stdout.write(lines.join("\n") + "\n");
