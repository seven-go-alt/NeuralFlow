import js from "@eslint/js";
import nextVitals from "eslint-config-next/core-web-vitals";
import tseslint from "typescript-eslint";

const browserGlobals = {
  AbortSignal: "readonly",
  document: "readonly",
  crypto: "readonly",
  fetch: "readonly",
  FormData: "readonly",
  Intl: "readonly",
  navigator: "readonly",
  performance: "readonly",
  React: "readonly",
  ReadableStream: "readonly",
  TextDecoder: "readonly",
  window: "readonly",
};

export default tseslint.config(
  {
    ignores: [".next/**", "node_modules/**", "out/**", "next-env.d.ts"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...nextVitals,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      globals: {
        ...browserGlobals,
        process: "readonly",
      },
    },
    rules: {
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "no-undef": "off",
    },
  },
);
