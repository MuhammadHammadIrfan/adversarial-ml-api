import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FGSM Adversarial Attack Demo",
  description:
    "Demonstrate the Fast Gradient Sign Method (FGSM) adversarial attack on an MNIST CNN model.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
