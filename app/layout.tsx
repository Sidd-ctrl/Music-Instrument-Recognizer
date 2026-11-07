import "./globals.css";

export const metadata = {
  title: "TIMBRE — Instrument Intelligence Lab",
  description:
    "Explainable audio ML · KNN + Librosa · 26-D spectral fingerprints · IRMAS 11K+ · Built for researchers",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&family=Instrument+Serif:ital@0;1&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300&display=swap"
          rel="stylesheet"
        />
        <style>{`
          :root{ --font-display:"Space Grotesk", system-ui, sans-serif; --font-mono:"JetBrains Mono", ui-monospace, monospace; --font-sans:"Newsreader", Georgia, serif; }
          .font-display{ font-family: var(--font-display); }
          .font-mono{ font-family: var(--font-mono); }
          .font-serif{ font-family: var(--font-sans); }
        `}</style>
      </head>
      <body className="antialiased selection:bg-lime selection:text-void">
        {children}
      </body>
    </html>
  );
}
