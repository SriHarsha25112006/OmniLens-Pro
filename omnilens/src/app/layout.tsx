import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import NavBar from "@/components/NavBar";

import { ThemeProvider } from "@/components/ThemeContext";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "OmniLens Pro",
  description: "The Autonomous AI Shopping Agent",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased selection:bg-purple-500/30`}>
        <ThemeProvider>
          <div className="fixed inset-0 -z-50 overflow-hidden pointer-events-none transition-colors duration-500 bg-background">
            <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] rounded-full bg-purple-900/10 blur-[120px] mix-blend-screen opacity-50 dark:opacity-100" />
            <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] rounded-full bg-cyan-900/10 blur-[120px] mix-blend-screen opacity-50 dark:opacity-100" />
          </div>

          <div className="relative z-10 flex flex-col min-h-screen text-foreground">
            <NavBar />
            <main className="flex-1 pb-20 md:pb-0 md:pt-0">
              {children}
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
