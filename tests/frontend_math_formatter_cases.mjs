import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const start = html.indexOf("const MATH_FORMAT_MAX_DEPTH");
const end = html.indexOf("function renderNodeEquations", start);
assert.ok(start >= 0 && end > start, "Could not locate math formatter block in index.html");

const formatterSource = html.slice(start, end);
const context = {
  console,
  escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
};
vm.createContext(context);
vm.runInContext(`${formatterSource}\nglobalThis.formatMath = formatMath;`, context);

const cases = [
  {
    name: "negative term inside a product is not negated as a whole sum",
    expr: "2*(-G12**2 + (G11 + w1)*(Dabc + G22 + Grc + w2))/(Dabc + G22 + Grc + w2)",
    mustContain: [
      "-G<sub>12</sub><sup>2</sup>",
      '<span class="math-op">+</span>'
    ],
    mustNotContain: [
      "-(G<sub>12</sub><sup>2</sup>"
    ]
  },
  {
    name: "explicit negated parenthesized sum remains a negated sum",
    expr: "-(a + b)",
    mustContain: ["-(a <span class=\"math-op\">+</span> b)"]
  },
  {
    name: "leading negative term plus another term stays additive",
    expr: "-a + b",
    mustContain: ["-a <span class=\"math-op\">+</span> b"],
    mustNotContain: ["-(a <span class=\"math-op\">+</span> b)"]
  },
  {
    name: "difference of squares keeps the minus between terms",
    expr: "x**2 - y**2",
    mustContain: ["x<sup>2</sup> <span class=\"math-op\">-</span> y<sup>2</sup>"]
  },
  {
    name: "fraction keeps denominator grouping",
    expr: "a**2/(b + c)",
    mustContain: [
      '<span class="math-frac">',
      "a<sup>2</sup>",
      'b <span class="math-op">+</span> c'
    ]
  },
  {
    name: "multiplication of sums keeps both parentheses",
    expr: "(a - b)*(c + d)",
    mustContain: [
      '(a <span class="math-op">-</span> b)',
      " · ",
      '(c <span class="math-op">+</span> d)'
    ]
  },
  {
    name: "sqrt expression is displayed without changing signs",
    expr: "sqrt(a - b) + c",
    mustContain: [
      "sqrt(a - b)",
      '<span class="math-op">+</span> c'
    ]
  },
  {
    name: "nested negative rational expression keeps inner additive sign",
    expr: "(-x**2 + y)/(z - w)",
    mustContain: [
      "-x<sup>2</sup>",
      '<span class="math-op">+</span> y',
      'z <span class="math-op">-</span> w'
    ],
    mustNotContain: [
      "-(x<sup>2</sup> <span class=\"math-op\">+</span> y)"
    ]
  },
  {
    name: "mixed exp log sqrt powers and arithmetic keep local signs",
    expr: "(exp(a) - log(b + c)*sqrt(d - e) + x**3/(y - z) - m*n)/(p + q**2)",
    mustContain: [
      "exp(a)",
      "log(b + c)",
      "sqrt(d - e)",
      "x<sup>3</sup>",
      '<span class="math-op">-</span> m · n',
      "q<sup>2</sup>"
    ],
    mustNotContain: [
      "-(exp(a)",
      "-(m · n)"
    ]
  }
];

for (const testCase of cases) {
  const rendered = context.formatMath(testCase.expr);
  for (const expected of testCase.mustContain || []) {
    assert.ok(
      rendered.includes(expected),
      `${testCase.name}: expected rendered HTML to include ${expected}\nRendered: ${rendered}`
    );
  }
  for (const forbidden of testCase.mustNotContain || []) {
    assert.ok(
      !rendered.includes(forbidden),
      `${testCase.name}: rendered HTML should not include ${forbidden}\nRendered: ${rendered}`
    );
  }
}

console.log(`math formatter cases passed: ${cases.length}`);
