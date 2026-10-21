import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NBA Game Predictions",
  description: "Daily NBA win probabilities from a logistic regression model, with a public track record.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
