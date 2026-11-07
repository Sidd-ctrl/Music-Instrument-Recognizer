/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        void: "#070A07",
        surface: "#0F1710",
        "surface-2": "#121E13",
        "surface-3": "#1B2A1B",
        line: "#1E3520",
        "line-strong": "#2A4A2E",
        lime: "#C6FF2B",
        "lime-dim": "#A3D920",
        mint: "#00E599",
        "mint-dark": "#00B87A",
        paper: "#E8F0E6",
        muted: "#6E8A6E",
        faint: "#4A6650",
      },
      fontFamily: {
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
        sans: ["var(--font-sans)"],
      },
      borderRadius: {
        'xl2': '18px',
        '2xl': '22px',
      }
    },
  },
  plugins: [],
}
