"use client";

import { useTheme } from "./ThemeProvider";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex items-center space-x-2">
      <span className="text-sm text-gray-600 dark:text-gray-300">
        {theme === "light" ? "☀️" : "🌙"}
      </span>
      <button
        onClick={toggleTheme}
        className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-200 dark:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        role="switch"
        aria-checked={theme === "dark"}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-lg transition-transform ${
            theme === "dark" ? "translate-x-6" : "translate-x-1"
          }`}
        />
        <span className="sr-only">Toggle theme</span>
      </button>
      <span className="text-sm text-gray-600 dark:text-gray-300">
        {theme === "light" ? "Light" : "Dark"}
      </span>
    </div>
  );
}
