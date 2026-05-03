import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif"
        ]
      },
      colors: {
        ink: "#20252b",
        fjord: "#315f72",
        moss: "#4f6f52",
        cloud: "#f6f7f4",
        paper: "#fffefa",
        line: "#d8ddd3",
        warning: "#9a5a20"
      }
    }
  },
  plugins: []
} satisfies Config;
