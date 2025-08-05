"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Logging utility for Theme
const log = {
  info: (message: string, data?: any) => {
    console.log(`🎨 [Theme] ${message}`, data ? data : "");
  },
  debug: (message: string, data?: any) => {
    console.debug(`🟡 [Theme] ${message}`, data ? data : "");
  },
};

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    log.info("ThemeProvider initializing");

    // Check for saved theme in localStorage or use system preference
    const savedTheme = localStorage.getItem("theme") as Theme;
    if (savedTheme) {
      log.info(`Loading saved theme: ${savedTheme}`);
      setTheme(savedTheme);
    } else {
      // Check system preference
      const prefersDark = window.matchMedia(
        "(prefers-color-scheme: dark)"
      ).matches;
      const systemTheme = prefersDark ? "dark" : "light";
      log.info(`No saved theme found, using system preference: ${systemTheme}`);
      setTheme(systemTheme);
    }
  }, []);

  useEffect(() => {
    log.debug(`Applying theme: ${theme}`);

    // Apply theme to document
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);

    log.info(`Theme successfully applied and saved: ${theme}`);
  }, [theme]);

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    log.info(`Theme toggled: ${theme} → ${newTheme}`);
    setTheme(newTheme);
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
